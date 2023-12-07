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
                 BBS_model_path='../../corpus/dataset_for_fine-tuning/categorized_level-0_with_others/254-model-epoch-3',
                 BBS_tokenizer_path='itsunoda/wolfbbsRoBERTa-small',
                 num_cpus=4,
                 sample=False):
        
        self.save_dir = save_dir
        self.bbs_dir = bbs_dir
        self.num_cpus = num_cpus
        
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


    def save_to_pickle(self, raw_nested_utters, labels, users, mode):
        try:
            self.logger.debug(f"raw_nested_utters: {raw_nested_utters[:5]}")
            self.logger.debug(f"labels: {labels[:5]}")
            self.logger.debug(f"users: {users[:5]}")
            
            num_utters = []
            with trange(len(raw_nested_utters), desc=f"pickling({mode})...") as t:
                for _, raw_nested_utter in zip(t, raw_nested_utters):
                    num_utters.append(len(raw_nested_utter))
            df = pd.DataFrame({'nested_utters': raw_nested_utters, 'num_utters': num_utters, 'labels': labels, 'users': users})
            
            with open(f'{self.save_dir}/{mode}.pkl', 'wb') as f:
                pickle.dump(df, f, protocol=5)
        except Exception as e:
            self.logger.error(f"Error in save_to_pickle: {e}")
            self.notifier.notify(operation=f'save_to_pickle({mode})', status='failed', message=str(traceback.format_exc()))
            raise
    

    def exclude_werewolf_specific_utterances(self, df: pd.DataFrame, mode: str) -> pd.DataFrame:
        try:
            # Check if df is None or empty
            assert df is not None, "DataFrame df is None."
            assert not df.empty, "DataFrame df is empty."

            total_rows = len(df)
            num_batches = self.num_cpus 
            self.logger.info(f"- {mode} - Splitting {total_rows} rows of DataFrame into {num_batches} batches ")

            df_split = np.array_split(df, num_batches)
            
            # Check if df_split is valid
            assert all(isinstance(batch, pd.DataFrame) for batch in df_split), "df_split contains non-DataFrame elements."
            assert all(not batch.empty for batch in df_split), "df_split contains empty DataFrame."

            batch_with_mode = [(index, batch, mode) for index, batch in enumerate(df_split)]
            
            with Pool(processes=self.num_cpus) as pool:
                results = pool.starmap(self.process_batch, batch_with_mode)
            
            # Check if results are valid
            assert results is not None, "Results from pool.starmap is None."
            assert all(isinstance(result, tuple) for result in results), "Results contain non-tuple elements."

            results.sort(key=lambda x: x[0])

            # Check if sorted results are valid
            assert all(isinstance(batch, pd.DataFrame) for index, batch in results), "Sorted results contain non-DataFrame elements."

            sorted_dfs = [batch for index, batch in results]

            # Check if sorted_dfs are valid
            assert all(isinstance(batch, pd.DataFrame) for batch in sorted_dfs), "sorted_dfs contains non-DataFrame elements."

            final_df = pd.concat(sorted_dfs, ignore_index=True)
            
            # Check if the final DataFrame is valid
            assert final_df is not None, "Final DataFrame is None."
            assert not final_df.empty, "Final DataFrame is empty."
            expected_columns = [ "nested_utters", "num_utters", "labels", "users"]
            for col in expected_columns:
                assert col in final_df.columns, f"Final DataFrame is missing expected column {col}"

            return final_df
       
        except Exception as e:
            self.logger.error(f"Error in exclude_werewolf_specific_utterances: {e}")
            raise


    def process_batch(self, index, batch_df, mode):
        try:
            batch_size = len(batch_df)
            self.logger.info(f"- {mode} - Starting to process batch {index} with {batch_size} rows")

            try:
                # Initialize tokenizer and model for each batch here
                self.logger.info(f"- {mode} - Loading BBS tokenizer and model for batch {index}")
                BBStokenizer = AutoTokenizer.from_pretrained(self.BBS_tokenizer_path)
                BBSmodel = RobertaForSequenceClassification.from_pretrained(self.BBS_model_path)
                self.logger.info(f"- {mode} - Loaded BBS tokenizer and model for batch {index}")
            except Exception as e:
                self.logger.error(f"Failed to load BBS tokenizer and model: {e}")
                raise

            for row in batch_df.itertuples():
                try:
                    nested_utters = row.nested_utters
                    nested_utters = self._exclude_werewolf_specific_utterances(nested_utters['raw_nested_utters'].tolist(), BBStokenizer, BBSmodel)
                    batch_df.at[row.Index, 'nested_utters'] = pd.DataFrame(nested_utters, columns=['raw_nested_utters'])
                    self.logger.info(f"- {mode} - Processed row {row.Index + 1} in batch {index}")
                except Exception as e:
                    self.logger.warning(f"Failed to process row {row.Index + 1} in batch {index}: {e}")

            self.logger.info(f"- {mode} - Finished processing batch {index}")
            
            return index, batch_df

        except Exception as e:
            self.logger.error(f"Failed to process batch {index} in {mode}: {e}")
            self.notifier.notify(operation=f'batch {index} processing in {mode}', status='failed', message=str(e))
            raise


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
    

    def run(self, data_dir='data/nested'):
        if self.sample:
            data_dir = 'data/nested_sample'
            limit_row = 100
        else:
            data_dir = data_dir
        
        try:
            for mode in ['train', 'valid', 'test']:
                self.logger.info(f"Started processing {mode} df")
                try:
                    df = pd.read_pickle(os.path.join(data_dir, f'{mode}.pkl'))
                    if self.sample:
                        df = df[:limit_row]
                except Exception as e:
                    self.logger.error(f"Failed to read {mode} DataFrame: {e}")
                    continue
                self.logger.debug(df.head())
                df = self.exclude_werewolf_specific_utterances(df, mode)
                self.logger.debug(f"{df.head()}")
                self.logger.debug(f"{df['nested_utters'].apply(lambda x: len(x)).describe()}")
                self.logger.debug(f"{df.iloc[0]['nested_utters'][:5]}")
                self.save_to_pickle(df.nested_utters, df.labels, df.users, mode)
        except Exception as e:
            self.logger.error(f"Error in run: {e}")
            self.notifier.notify(operation='extract_bbs.py', status='failed', message=str(e))
            return
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_cpus", default=3)
    parser.add_argument("--sample", action='store_true')
    parser.add_argument("--data_dir", default='nested')
    args = parser.parse_args()
    
    if args.sample:
        save_dir = 'data/exclude_bbs_nested_sample'
    else:
        save_dir = f'data/exclude_bbs_{args.data_dir}'
        
    data_dir = f'data/{args.data_dir}'
    
    os.makedirs(save_dir, exist_ok=True)

    BBS_model_path = '/home/haoki/Documents/vscode-workplaces/sotuken/corpus/dataset_for_fine-tuning/categorized_level-0_with_others/254-model-epoch-3/'

    num_cpus = int(args.num_cpus)

    processor = DataProcessor(save_dir=save_dir, BBS_model_path=BBS_model_path, num_cpus=num_cpus, sample=args.sample)
    processor.run(data_dir=data_dir)
