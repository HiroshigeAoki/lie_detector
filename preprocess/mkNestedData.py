# -*- coding: utf-8 -*-///
import json
import glob
import pickle
from pathlib import Path
from tqdm import trange
import pandas as pd
from sklearn.model_selection import train_test_split
import os, sys

sys.path.append(os.pardir)
from preprocess.cleaner import clean_sent, replace_term, auto_exclude_sent
from utils.parser import parse_args
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
    villager_num = df['label'].value_counts()[0]
    werewolf_num = df['label'].value_counts()[1]

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
        utterances = []
        for day in days:
            for utterance in day['utterances']:
                if utterance.get('subject') == participant \
                        and utterance.get('u_type') == 'say':
                    utter = clean_sent(utterance['utterance'])
                    utter = replace_term(utter)

                    if exclude and auto:
                        BBS_term, type = classifier(utter)
                        if BBS_term:
                            deleted_utters.append(f"type: {type}\n{utter}")
                            continue

                    if len(utter) == 0:
                        continue
                    else:
                        utterances.append(utter)

        if len(utterances) == 0:
            continue
        else:
            utterances = list(set(utterances, key=utterances.index)) # remove duplications
            nested_utterances.append(utterances)
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

    X_train, X_valid_test, y_train, y_valid_test = train_test_split(
        nested_utterances,labels, test_size=0.2, stratify=labels, random_state=0
    )

    X_valid, X_test, y_valid, y_test = train_test_split(
        X_valid_test, y_valid_test, test_size=.5, stratify=y_valid_test, random_state=0
    )

    with open(f'{train_txt_dir}/train.txt', 'w') as f:
        for _nested_utterances in X_train:
            for _utterance in _nested_utterances:
                f.write(_utterance)
                f.write('\n')

    wakati = MeCab.Tagger("-O wakati -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")
    with open(f'{train_txt_dir}/split_train.txt', 'w') as f:
        for _nested_utterances in X_train:
            for _utterance in _nested_utterances:
                f.write(wakati.parse(_utterance))
                f.write('\n')

    X_train, y_train = duplicate_werewolves(X_train, y_train)
    X_valid, y_valid = duplicate_werewolves(X_valid, y_valid)
    X_test, y_test = duplicate_werewolves(X_test, y_test)

    tagger = CustomMeCabTagger("-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")

    for _nested_utterances, _labels, type in zip([X_train, X_valid, X_test], [y_train, y_valid, y_test], ['train', 'valid', 'test']):
        X_dfs, num_utters = [], []
        with trange(len(_nested_utterances), desc=f"parsing({type})...") as t:
            for _, _utterances in zip(t, _nested_utterances):
                parsed_dfs = []
                num_morphemes = []
                for _utterance in _utterances:
                    parsed_df = tagger.parseToDataFrame(_utterance) # mecabで形態素解析
                    num_morphemes.append(len(parsed_df))
                    parsed_dfs.append(parsed_df)
                X_df = pd.DataFrame({'raw_nested_utters': _utterances, 'parsed_nested_utters': parsed_dfs, 'num_morphemes': num_morphemes})
                X_dfs.append(X_df)
                num_utters.append(len(_utterances))
        df = pd.DataFrame({'nested_utters': X_dfs, 'num_utters': num_utters, 'label': _labels})
        print_stats(df, type, save_dir)

        with open(f'{save_dir}/{type}.pkl', 'wb') as f:
            pickle.dump(df,  f, protocol=5)


def main():
    args = parse_args()
    ROLES_USED = args.m_role_dict.values()

    files = glob.glob("../../../corpus/BBSjsons/A/*.json")  # 7249 files
    save_dir = Path(f"../model/data/nested")
    os.makedirs(save_dir, exist_ok=True)
    train_txt_dir = f"../tokenizer/" # for fasttext

    if args.sample:
        files = files[:10]
        save_dir = save_dir / f"{'auto' if args.auto else 'man'}"

    extract(files, save_dir, train_txt_dir, ROLES_USED, exclude=args.exclude, auto=args.auto)
    print('done!')


if __name__ == '__main__':
    main()