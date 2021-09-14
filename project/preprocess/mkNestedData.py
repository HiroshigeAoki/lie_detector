# -*- coding: utf-8 -*-///
import json
import glob
import pickle
from pathlib import Path
from tqdm import trange, tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
import os, sys
sys.path.append(os.pardir)
from preprocess.cleaner import clean_sent, replace_term, auto_exclude_sent
from utils.parser import parse_args
from utils.gmail_send import Gmailsender
import MeCab
from preprocess.custom_mecab_tagger import CustomMeCabTagger


def duplicate_werewolves(nested_utterances, labels):
    """oversample werewolf class and change ratio 50:50

    Args:
        nested_utterances (list): [description]
        labels (list): [description]

    Returns:
        nested_utterances (list): [description]
        labels (list): [description]
    """
    werewolf_indices = [i for i, label in enumerate(labels) if label == 1]
    civil_num = len(labels) - len(werewolf_indices)
    difference = civil_num - len(werewolf_indices)

    if len(werewolf_indices) >= difference:
        werewolf_indices[:civil_num]
    else:
        for i in range(difference - len(werewolf_indices)):
            werewolf_indices.append(werewolf_indices[i])

    for i in werewolf_indices:
        nested_utterances.append(nested_utterances[i])
        labels.append(1)

    return nested_utterances, labels


def print_stats(df, type, save_dir):
    print("calculating stats...")
    player_num = len(df)
    villager_num = df['labels'].value_counts()[0]
    werewolf_num = df['labels'].value_counts()[1]

    utters_num = df['num_utters'].sum()
    ave_utters = df['num_utters'].mean()
    max_utters = df['num_utters'].max()
    min_utters = df['num_utters'].min()

    utter_len_per_player = [utters_df['num_morphemes'].sum() for utters_df in df['nested_utters']]
    mor_num = sum(utter_len_per_player)
    ave_mor_per_player = mor_num / player_num
    max_mor_per_player = max(utter_len_per_player)
    min_mor_per_player = min(utter_len_per_player)
    utter_len_per_utter = [_num for utters_df in df['nested_utters'] for _num in utters_df['num_morphemes'].to_list()]
    ave_mor_per_utter = mor_num / utters_num
    max_mor_per_utter = max(utter_len_per_utter)
    min_mor_per_utter = min(utter_len_per_utter)

    # TODO: werewolfとvillagerで発話長、発話数等違いがあるか検証する。（別のプログラムでやった方がいいかも。）
    with open(f'{save_dir}/stats.txt', 'a') as f:
        print(f'--- {type} ---', file=f)
        print(f'- player(total:{player_num:,})', file=f)
        print(f'    | werewolf num:{werewolf_num:,}, {(werewolf_num / player_num)*100:.2f}%', file=f)
        print(f'    | villager num:{villager_num:,}, {(villager_num / player_num)*100:.2f}%', file=f)
        print(f'- utterances(total:{utters_num:,})', file=f)
        print(f'    | ave utters(/player):{ave_utters:,.2f}', file=f)
        print(f'    | max utters(/player):{max_utters:,}', file=f)
        print(f'    | min utters(/player):{min_utters:,}', file=f)
        print(f'- morpheme(total:{mor_num:,})', file=f)
        print(f'    - /player', file=f)
        print(f'        | ave morphemes(/player):{ave_mor_per_player:,.2f}', file=f)
        print(f'        | max morphemes(/player):{max_mor_per_player:,}', file=f)
        print(f'        | min morphemes(/player):{min_mor_per_player:,}', file=f)
        print(f'    - /utter', file=f)
        print(f'        | ave morphemes(/utter):{ave_mor_per_utter:,.2f}', file=f)
        print(f'        | max morphemes(/utter):{max_mor_per_utter:,}', file=f)
        print(f'        | min morphemes(/utter):{min_mor_per_utter:,}', file=f)
        print(f'\n', file=f)


def each_extract(filePath, ROLES_USED, exclude=True, auto=True, classifier=None):
    labels = []
    nested_utterances = [] #(user_num, utterance_num)
    participants = {}
    deleted_utters = []
    roles_and_tags_b = {"人狼": 1, "狂人": 1, "村人": 0, "占い師": 0, "霊能者": 0, "狩人": 0, "共有者": 0, "ハムスター人間": 0}
    min_len_char = 10
    min_num_utter = 10

    with open(filePath, encoding='utf-8') as f:
        data = json.load(f)
        p = data.get('participants')
        participants.update(p)

    """processing each day"""
    days = data['days'][1:-1]

    """split utterances for each character"""
    for participant in participants.keys():
        if participant == '楽天家 ゲルト':
            continue
        participant_role = participants[participant]
        if participant_role not in ROLES_USED:
            continue
        _nested_utterance = []
        for day in days:
            for _utterance in day['utterances']:
                if _utterance.get('subject') == participant \
                        and _utterance.get('u_type') == 'say':
                    _utterance = clean_sent(_utterance['utterance'])
                    _utterance = replace_term(_utterance)

                    # CO発話、人狼用語を削除
                    if exclude and auto:
                        BBS_term, type = classifier(_utterance)
                        if BBS_term:
                            deleted_utters.append(f"type: {type}\n{_utterance}")
                            continue

                    if len(_utterance) <= min_len_char:
                        continue
                    else:
                        _nested_utterance.append(_utterance)

        _nested_utterance = sorted(set(_nested_utterance), key=_nested_utterance.index) # remove duplications

        if len(_nested_utterance) <= min_num_utter:
            continue
        else:
            nested_utterances.append(_nested_utterance)
            labels.append(roles_and_tags_b[participant_role])

    return nested_utterances, labels, deleted_utters


def extract(filePaths, save_dir, train_txt_dir, ROLES_USED, exclude=True, auto=True):
    """ファイルから必要なところを取り出してリストにまとめます"""
    nested_utterances = []
    labels = []
    deleted_utters = [] # 一応

    with trange(len(filePaths), desc="extracting... ") as t:
        for _, filePath in zip(t, filePaths):
            _nested_utterances, _labels, _deleted_utters\
                = each_extract(filePath, ROLES_USED, exclude, auto)
            nested_utterances += _nested_utterances
            labels += _labels
            deleted_utters += _deleted_utters

    print("split data into train, valid and test")
    X_train, X_valid_test, y_train, y_valid_test = train_test_split(
        nested_utterances,labels, test_size=0.2, stratify=labels, random_state=0
    )

    X_valid, X_test, y_valid, y_test = train_test_split(
        X_valid_test, y_valid_test, test_size=.5, stratify=y_valid_test, random_state=0
    )

    with open(f'{train_txt_dir}/train.txt', 'w') as f:
        for _nested_utterances in tqdm(X_train, desc="outputting training data to train.txt"):
            for _utterance in _nested_utterances:
                f.write(_utterance)
                f.write('\n')

    wakati = MeCab.Tagger("-O wakati -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")
    with open(f'{train_txt_dir}/split_train.txt', 'w') as f:
        for _nested_utterances in tqdm(X_train, desc="outputting split training data to split_train.txt"):
            for _utterance in _nested_utterances:
                f.write(wakati.parse(_utterance))
                f.write('\n')

    print("duplicating werewolf dataset")
    X_train, y_train = duplicate_werewolves(X_train, y_train)
    X_valid, y_valid = duplicate_werewolves(X_valid, y_valid)
    X_test, y_test = duplicate_werewolves(X_test, y_test)

    tagger = CustomMeCabTagger(option="-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")

    for _nested_utterances, _labels, type in zip([X_train, X_valid, X_test], [y_train, y_valid, y_test], ['train', 'valid', 'test']):
        X_dfs, num_utters, parsed_dfs = [], [], []
        with trange(len(_nested_utterances), desc=f"parsing({type})...") as t:
            for _, _utterances in zip(t, _nested_utterances):
                _parsed_dfs = []
                num_morphemes = []
                for _utterance in _utterances:
                    parsed_df = tagger.parseToDataFrame(_utterance) # mecabで形態素解析
                    num_morphemes.append(len(parsed_df))
                    _parsed_dfs.append(parsed_df)
                X_df = pd.DataFrame({'raw_nested_utters': _utterances, 'num_morphemes': num_morphemes})
                X_dfs.append(X_df)
                parsed_df = pd.DataFrame({'parsed_nested_utters': _parsed_dfs})
                parsed_dfs.append(parsed_df)
                num_utters.append(len(_utterances))
        df = pd.DataFrame({'nested_utters': X_dfs, 'num_utters': num_utters, 'labels': _labels})
        print_stats(df, type, save_dir)

        with open(f'{save_dir}/{type}.pkl', 'wb') as f:
            pickle.dump(df,  f, protocol=5)

        with open(f'{save_dir}/{type}_parsed_df.pkl', 'wb') as f:
            pickle.dump(pd.DataFrame({'parsed_dfs': parsed_dfs}), f, protocol=5)


def main():
    args = parse_args()
    ROLES_USED = args.m_role_dict.values()

    files = glob.glob("../../../../corpus/BBSjsons/*/*.json")  # 7249 files

    if args.sample:
        files = files[:3]
    train_txt_dir = f"../tokenizer/" if not args.sample else "../model/data/nested_sample"
    save_dir = f"../model/data/nested" if not args.sample else "../model/data/nested_sample"
    os.makedirs(save_dir, exist_ok=True)
    shutil.rmtree(save_dir)
    os.mkdir(save_dir)

    extract(files, save_dir, train_txt_dir, ROLES_USED, exclude=args.exclude, auto=args.auto)
    print('done!')
    if not args.sample:
        gmail_sender = Gmailsender(subject="実行終了通知")
        gmail_sender.send(body="mkNested.py終了")

if __name__ == '__main__':
    main()