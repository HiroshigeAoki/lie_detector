import pickle
import logging
import os
import sys
import argparse
import pandas as pd
from tqdm import trange
sys.path.append('./src/')
import torch
import numpy as np
from transformers import AutoTokenizer, RobertaForSequenceClassification
from multiprocessing import Pool
from utils.logger import OperationEndNotifier
import traceback



class DataProcessor:

    def __init__(self, save_dir, bbs_dir='../../corpus/BBSjsons',
                 role2label={"人狼": 1, "狂人": 1, "村人": 0, "占い師": 0, "霊能者": 0, "狩人": 0, "共有者": 0, "ハムスター人間": 0},
                 used_role=["人狼", "狂人", "村人", "占い師", "霊能者", "狩人"],
                 min_len_char=10,
                 min_num_utter=10,
                 train_valid_test_ratio=[0.8, 0.1, 0.1],
                 BBS_model_path='../../corpus/dataset_for_fine-tuning/categorized_level-0_with_others/254-model-epoch-3',
                 BBS_tokenizer_path='itsunoda/wolfbbsRoBERTa-small',
                 num_cpus=4,
                 save_format='parquet',
                 sample=False):
        
        self.save_dir = save_dir
        self.bbs_dir = bbs_dir
        self.role2label = role2label
        self.used_role = used_role
        self.min_len_char = min_len_char
        self.min_num_utter = min_num_utter
        self.train_valid_test_ratio = train_valid_test_ratio
        self.num_cpus = num_cpus
        self.save_format = save_format
        
        self.sample = sample
        
        self.logger = logging.getLogger(__name__)
        if self.sample:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)
        
        # 取り除かれた発話確認用
        self.excluded_utters_logger = logging.getLogger('excluded_utters')
        self.excluded_utters_logger.setLevel(logging.DEBUG)
        self.excluded_utters_logger.addHandler(logging.FileHandler(os.path.join(save_dir, 'excluded_utters.txt'), mode='w'))
        
        self.notifier = OperationEndNotifier(subject="extract_bbs.py")
        
        self.BBS_model_path = BBS_model_path
        self.BBS_tokenizer_path = BBS_tokenizer_path


    def save_to_pickle(self, X, y, users, name):
        X_dfs, num_utters = [], []
        with trange(len(X), desc=f"pickling({name})...") as t:
            for _, X in zip(t, X):
                X_df = pd.DataFrame({'raw_nested_utters': X})
                X_dfs.append(X_df)
                num_utters.append(len(X))
        df = pd.DataFrame({'nested_utters': X_dfs, 'num_utters': num_utters, 'labels': y, 'users': users})
        
        with open(f'{self.save_dir}/{name}.pkl', 'wb') as f:
            pickle.dump(df, f, protocol=5)
            
    
    def exclude_werewolf_specific_utterances(self, df: pd.DataFrame, mode: str):
        total_rows = len(df)
        num_batches = self.num_cpus  # 例としてCPU数をバッチ数としています。
        self.logger.info(f"Splitting {total_rows} rows of DataFrame into {num_batches} batches for mode: {mode}")

        df_split = np.array_split(df, num_batches)

        # 各バッチにmodeを追加
        batch_with_mode = [(index, batch, mode) for index, batch in enumerate(df_split)]

        with Pool(processes=self.num_cpus) as pool:
            self.logger.info(f"Initiating multiprocessing with {self.num_cpus} CPUs for mode: {mode}")
            results = pool.starmap(self.process_batch, batch_with_mode)

        self.logger.info(f"Completed multiprocessing for mode: {mode}. Now sorting and concatenating batches.")
        results.sort(key=lambda x: x[0])  # インデックスでソート
        sorted_dfs = [batch for index, batch in results]  # ソートされたDataFrameのリスト
        self.logger.info(f"Successfully sorted and concatenated {num_batches} batches for mode: {mode}")
        return pd.concat(sorted_dfs, ignore_index=True)


    def process_batch(self, index, batch_df, mode):
        num_rows = len(batch_df)
        self.logger.info(f"Starting to process batch {index} containing {num_rows} rows for mode: {mode}")

        # Initialize tokenizer and model for each batch here
        self.logger.info(f"Initializing BBS tokenizer and model for batch {index} for mode: {mode}")
        BBStokenizer = AutoTokenizer.from_pretrained(self.BBS_tokenizer_path)
        BBSmodel = RobertaForSequenceClassification.from_pretrained(self.BBS_model_path)
        self.logger.info(f"Successfully initialized BBS tokenizer and model for batch {index} for mode: {mode}")

        # Initialize a counter to keep track of the number of rows processed
        rows_processed = 0

        for row in batch_df.itertuples():
            nested_utters = row.nested_utters
            nested_utters = self._exclude_werewolf_specific_utterances(nested_utters['raw_nested_utters'].tolist(), BBStokenizer, BBSmodel)
            batch_df.at[row.Index, 'nested_utters'] = pd.DataFrame(nested_utters, columns=['raw_nested_utters'])
            rows_processed += 1
            self.logger.info(f"Processed {rows_processed}/{num_rows} rows in batch {index} for mode: {mode}")

        self.logger.info(f"Finished processing all {num_rows} rows in batch {index} for mode: {mode}")
        return index, batch_df  # インデックスと処理済みのDataFrameをタプルで返却


    def _exclude_werewolf_specific_utterances(self, nested_utterances: list, BBStokenizer, BBSmodel):
        """CO発話、占い発話、霊能発話、狩人発話をモデルを使って削除"""
            # STSモデルのlabel内訳: 0: CO, 1: 護衛, 2: 占い・霊能結果発表発話, 3: その他の発話
            # 今回は人狼特有の発話(0,1,2)を削除する。
        if len(nested_utterances) == 0:
            return nested_utterances
        try:
            inputs = BBStokenizer(nested_utterances, padding='max_length', truncation=True, return_tensors='pt', max_length=128)
        except Exception as e:
            print(e)
            return nested_utterances
        with torch.no_grad():
            outputs = BBSmodel(**inputs)
        # 3以外が出た場合でデバッグするようにする。
        predicts = list(np.argmax(outputs[0].cpu().numpy(), axis=1))
        
        utters = []
        for utter, pred in zip(nested_utterances, predicts):
            if pred==3:
                utters.append(utter)
            else:
                if pred==0:
                    self.excluded_utters_logger.debug(f"CO utterance: {utter}")
                elif pred==1:
                    self.excluded_utters_logger.debug(f"guard utterance: {utter}")
                elif pred==2:
                    self.excluded_utters_logger.debug(f"divine utterance: {utter}")
        return utters
    

    def run(self):
        if self.sample:
            data_dir = './data/nested_sample'
        else:
            data_dir = './data/nested'
        
        try:
            for mode in ['train', 'valid', 'test']:
                self.logger.info(f"Started pricessing {mode} df")
                df = pd.read_pickle(os.path.join(data_dir, f'{mode}.pkl'))
                self.logger.debug(df.head())
                self.exclude_werewolf_specific_utterances(df, mode)
                self.logger.debug(f"{df.head()}")
                self.logger.debug(f"{df['nested_utters'].apply(lambda x: len(x)).describe()}")
                self.logger.debug(f"{df.iloc[0]['nested_utters'][:5]}")
                self.save_to_pickle(df.nested_utters, df.labels, df.users, mode)
        except Exception as e:
            self.logger.error(traceback.format_exc())
            self.notifier.notify(operation='extract_bbs.py', status='failed', message=traceback.format_exc())
            return
        self.logger.info('done!')
        self.notifier.notify(operation='extract_bbs.py', status='done', message='done!')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_cpus", default=4)
    parser.add_argument("--sample", action='store_true')
    args = parser.parse_args()
    
    if args.sample:
        save_dir = 'data/exclude_bbs_nested_sample'
    else:
        save_dir = 'data/exclude_bbs_nested'
    
    os.makedirs(save_dir, exist_ok=True)

    BBS_model_path = '/home/haoki/Documents/vscode-workplaces/sotuken/corpus/dataset_for_fine-tuning/categorized_level-0_with_others/254-model-epoch-3/'

    num_cpus = os.cpu_count() if int(args.num_cpus)==-1 else int(args.num_cpus)

    processor = DataProcessor(save_dir=save_dir, BBS_model_path=BBS_model_path, save_format='pickle', num_cpus=num_cpus, sample=args.sample)
    processor.run()
