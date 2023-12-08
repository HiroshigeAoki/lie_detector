import json
import glob
import pickle
import shutil
import logging
import os
import sys
import argparse
import joblib
import pandas as pd
from tqdm import trange, tqdm
from collections import Counter
sys.path.append('./src/')
import torch
import numpy as np
from preprocess.cleaner import clean_sent, replace_term
from transformers import AutoTokenizer, RobertaForSequenceClassification


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
                 save_format='parquet'):
        
        self.save_dir = save_dir
        self.bbs_dir = bbs_dir
        self.role2label = role2label
        self.used_role = used_role
        self.min_len_char = min_len_char
        self.min_num_utter = min_num_utter
        self.train_valid_test_ratio = train_valid_test_ratio
        self.num_cpus = num_cpus
        self.save_format = save_format
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(logging.StreamHandler())
        self.logger.addHandler(logging.FileHandler(os.path.join(save_dir, 'log.txt')))      
        
        # load BBS model
        self.logger.debug('loading BBS model')
        self.BBStokenizer = AutoTokenizer.from_pretrained(BBS_tokenizer_path)
        self.BBSmodel = RobertaForSequenceClassification.from_pretrained(BBS_model_path)
        self.logger.debug('loading done!')


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


    def save_to_parquet(self, X, y, users, name):
        """Save data in parquet format."""
        X_dfs, num_utters = [], []
        with trange(len(X), desc=f"saving to parquet({name})...") as t:
            for _, X in zip(t, X):
                X_df = pd.DataFrame({'raw_nested_utters': X})
                X_dfs.append(X_df)
                num_utters.append(len(X))
        df = pd.DataFrame({'nested_utters': X_dfs, 'num_utters': num_utters, 'labels': y, 'users': users})

        df.to_parquet(f'{self.save_dir}/{name}.parquet', engine='pyarrow')


    def split_data(self, nested_utterances, labels, users):
        """データをユーザーが被らないように分割"""
        grouped = pd.DataFrame(
            dict(
                nested_utterances=nested_utterances,
                labels=labels,
                users=users
            )
        ).groupby('users')
        groups = [grouped.get_group(x) for x in grouped.groups]

        self.logger.debug('split data')
        # train, valid, testに同じユーザが跨らないように分割
        train_ratio, valid_ratio, test_ratio = self.train_valid_test_ratio
        i = 0
        train, train_size = [groups[i]], len(groups[i])
        while train_size < len(labels) * train_ratio:
            i += 1
            train_size += len(groups[i])
            train.append(groups[i])

        valid, valid_size = [groups[i]], len(groups[i])
        while valid_size < len(labels) * valid_ratio:
            i += 1
            valid_size += len(groups[i])
            valid.append(groups[i])

        test, test_size = [groups[i]], len(groups[i])
        while test_size < len(labels)* test_ratio:
            i += 1
            test_size += len(groups[i])
            test.append(groups[i])

        train.extend(groups[i+1:])
        train, valid, test = pd.concat(train), pd.concat(valid), pd.concat(test)
        return train, valid, test


    def extract(self, filePaths):
        """ファイルから必要なところを取り出してリストにまとめます"""
        self.logger.debug('extracting...')
        save_dir = self.save_dir
        nested_utterances = []
        labels = []
        users = []
        deleted = []

        outputs = joblib.Parallel(n_jobs=self.num_cpus)(
            joblib.delayed(self.extract_loop)(
                filepath,
            ) for filepath in tqdm(filePaths, desc='extracting...')
        )

        for _nested_utterances, _labels, _users, _deleted in outputs:
            # CO発話、占い発話、霊能発話、狩人発話をモデルを使って削除
            nested_utterances += self.exclude_werewolf_specific_utterances(_nested_utterances)
            labels += _labels
            users += _users
            deleted += _deleted

        # データ分割
        train, valid, test = self.split_data(nested_utterances, labels, users)
        
        X_train, y_train, users_train = train['nested_utterances'].tolist(), train['labels'].tolist(), train['users'].tolist()
        X_valid, y_valid, users_valid = valid['nested_utterances'].tolist(), valid['labels'].tolist(), valid['users'].tolist()
        X_test, y_test, users_test = test['nested_utterances'].tolist(), test['labels'].tolist(), test['users'].tolist()

        # 統計量計算
        train_stats, valid_stats, test_stats = Counter(y_train), Counter(y_valid), Counter(y_test)
        all_stats = train_stats + valid_stats + test_stats
        train_stats, valid_stats, test_stats, all_stats = dict(train_stats), dict(valid_stats), dict(test_stats), dict(all_stats)

        train_stats.update({'users_num': len(Counter(users_train))})
        valid_stats.update({'users_num': len(Counter(users_valid))})
        test_stats.update({'users_num': len(Counter(users_test))})
        all_stats.update({'users_num': len(Counter(users))})

        stats_df = pd.DataFrame((train_stats, valid_stats, test_stats, all_stats), index=['train', 'valid', 'test', 'all'])
        stats_df.to_csv(os.path.join(save_dir, 'raw_stats.csv'))
        stats_df.to_latex(os.path.join(save_dir, 'raw_stats.tex'))

        # sentence piece学習用
        train_for_tapt = []
        for train in X_train:
            train_for_tapt.extend(train)
            train_for_tapt.append("")

        bbs_data = train_for_tapt + deleted

        with open(f'{save_dir}/bbs.txt', 'w') as f: # for tapt pretraining of RoBERTa.
            for utterance in tqdm(bbs_data, desc="making bbs.txt"):
                f.write(utterance + '\n')

        # duplicate werewolf data
        self.logger.debug("duplicate werewolf dataset")
        X_train, y_train, users_train = self.duplicate_werewolves(X_train, y_train, users_train)
        X_valid, y_valid, users_valid = self.duplicate_werewolves(X_valid, y_valid, users_valid)
        X_test, y_test, users_test = self.duplicate_werewolves(X_test, y_test, users_test)

        # save data
        if self.save_format == 'pickle':
            self.save_to_pickle(X_train, y_train, users_train, 'train', save_dir)
            self.save_to_pickle(X_valid, y_valid, users_valid, 'valid', save_dir)
            self.save_to_pickle(X_test, y_test, users_test, 'test', save_dir)
        elif self.save_format == 'parquet':
            self.save_to_parquet(X_train, y_train, users_train, 'train', save_dir)
            self.save_to_parquet(X_valid, y_valid, users_valid, 'valid', save_dir)
            self.save_to_parquet(X_test, y_test, users_test, 'test', save_dir)
        
    
    def exclude_werewolf_specific_utterances(self, nested_utterances):
        """CO発話、占い発話、霊能発話、狩人発話をモデルを使って削除"""
            # STSモデルのlabel内訳: 0: CO, 1: 護衛, 2: 占い・霊能結果発表発話, 3: その他の発話
            # 今回は人狼特有の発話(0,1,2)を削除する。
        if len(nested_utterances) == 0:
            return nested_utterances
        try:
            inputs = self.BBStokenizer(nested_utterances, padding='max_length', truncation=True, return_tensors='pt', max_length=128)
        except Exception as e:
            print(e)
            print(nested_utterances)
            return nested_utterances
        with torch.no_grad():
            outputs = self.BBSmodel(**inputs)
        # 3以外が出た場合でデバッグするようにする。
        predicts = list(np.argmax(outputs[0].cpu().numpy(), axis=1))
        
        utters = []
        for utter, pred in zip(nested_utterances, predicts):
            if pred==3:
                utters.append(utter)
            else:
                if pred==0:
                    self.logger.debug(f"CO utterance: {utter}")
                elif pred==1:
                    self.logger.debug(f"guard utterance: {utter}")
                elif pred==2:
                    self.logger.debug(f"divine utterance: {utter}")
        return utters
    
    
    def extract_loop(self, filePath):
        deleted = []
        delete_nested_utterances = []
        participants = {}
        users_dict = {}

        with open(filePath, encoding='utf-8') as f:
            data = json.load(f)
            p = data.get('participants')
            u = data.get('users')
            participants.update(p)
            users_dict.update(u)

        """processing each day"""
        days = data['days'][1:-1] # exclude prologue and epilogue
        epilogue = data['days'][0]
        prologue = data['days'][-1]

        nested_utterances, labels, users, _deleted = self.preprocess(days=days, participants=participants, users_dict=users_dict)
        deleted.extend(_deleted)
        _delete_nested_utterances, _, _, _deleted = self.preprocess(days=[epilogue, prologue], participants=participants, users_dict=users_dict)
        for utterances in _delete_nested_utterances:
            delete_nested_utterances.extend(utterances)
            delete_nested_utterances.append("")
        deleted.extend(delete_nested_utterances + deleted)

        return nested_utterances, labels, users, deleted


    def preprocess(self, days, participants, users_dict):
        nested_utterances = [] #(user_num, utterance_num)
        labels = []
        users = []
        deleted = []
        """Aggregate all utterances of each player, respectively."""
        for participant in participants.keys():
            if participant == '楽天家 ゲルト': # exclude a bot player.
                continue
            participant_role = participants[participant]
            if participant_role not in self.used_role:
                continue
            _nested_utterance = []
            _deleted = []
            for i, day in enumerate(days):
                for utterance_inf in day['utterances']:
                    if utterance_inf.get('subject') == participant:
                        utterance = clean_sent(utterance_inf['utterance'])
                        utterance = replace_term(utterance)
                        if len(utterance) <= self.min_len_char:
                            continue
                        if utterance_inf.get('u_type') == 'say':
                            _nested_utterance.append(utterance)
                        else:
                            _deleted.append(utterance)
            _nested_utterance = sorted(set(_nested_utterance), key=_nested_utterance.index) # remove duplications
            
            # CO発話、占い発話、霊能発話、狩人発話をモデルを使って削除
            #_nested_utterance = self.exclude_werewolf_specific_utterances(_nested_utterance)
            
            _deleted = sorted(set(_deleted), key=_deleted.index)
            _deleted.append("")
            deleted.extend(_deleted)
            if len(_nested_utterance) > self.min_num_utter:
                nested_utterances.append(_nested_utterance)
                labels.append(self.role2label[participant_role])
                users.extend(self.get_keys_from_value(users_dict, participant))

        return nested_utterances, labels, users, deleted


    def get_keys_from_value(self, d, val):
        return [k for k, v in d.items() if v == val]
    

    def duplicate_werewolves(self, nested_utterances, labels, users):
        """oversample werewolf class and change ratio 50:50"""
        werewolf_indices = [i for i, label in enumerate(labels) if label == 1]
        civil_num = len(labels) - len(werewolf_indices)
        difference = civil_num - len(werewolf_indices)

        for i in range(difference - len(werewolf_indices)):
            werewolf_indices.append(werewolf_indices[i])

        for i in werewolf_indices:
            nested_utterances.append(nested_utterances[i])
            labels.append(1)
            users.append(users[i])

        return nested_utterances, labels, users
    

    def save_split_train_for_fasttext(self, tokenizer, path, data):
        """save split train data for fasttext training"""
        with open(path, 'w') as f:
            for utterance in data:
                f.write(' '.join(tokenizer.tokenize(utterance + '\n')))


    def run(self, sample):
        files = sorted(glob.glob(self.bbs_dir + '/*/*.json'))
        save_dir = self.save_dir

        if sample:
            files = files[:100]

        self.extract(files)
        self.logger.debug('done!')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default='data/nested_bbs')
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--num_cpus", default=4)
    args = parser.parse_args()
    if args.sample:
        args.save_dir = args.save_dir.replace("nested_bbs", "nested_bbs_sample")
    
    os.makedirs(args.save_dir, exist_ok=True)

    BBS_model_path = '/home/haoki/Documents/vscode-workplaces/sotuken/corpus/dataset_for_fine-tuning/categorized_level-0_with_others/254-model-epoch-3/'

    processor = DataProcessor(save_dir=args.save_dir, BBS_model_path=BBS_model_path, save_format='pickle', num_cpus=int(args.num_cpus))
    processor.run(sample=args.sample)
