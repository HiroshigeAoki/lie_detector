import json
import glob
import pickle
from tqdm import trange, tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertJapaneseTokenizer
import shutil
import argparse
import os, sys
sys.path.append(os.pardir)
from preprocess.cleaner import clean_sent, replace_term
from utils.gmail_send import Gmailsender
from preprocess.custom_mecab_tagger import CustomMeCabTagger


def extract(filePaths, save_dir, ROLES_USED):
    """ファイルから必要なところを取り出してリストにまとめます"""
    nested_utterances = []
    labels = []

    with trange(len(filePaths), desc="extracting... ") as t:
        for _, filePath in zip(t, filePaths):
            _nested_utterances, _labels = extract_loop(filePath, ROLES_USED)
            nested_utterances += _nested_utterances
            labels += _labels

    print("split data into train, valid and test")
    X_train, X_valid_test, y_train, y_valid_test = train_test_split(
        nested_utterances,labels, test_size=0.2, stratify=labels, random_state=0
    )

    X_valid, X_test, y_valid, y_test = train_test_split(
        X_valid_test, y_valid_test, test_size=.5, stratify=y_valid_test, random_state=0
    )

    tokenizer = CustomMeCabTagger("-O wakati -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd -r /home/haoki/Documents/vscode-workplaces/lie_detector/project/tokenizer/mecab_userdic/mecabrc")
    make_split_train(save_dir, X_train, tokenizer, file_name='split-train-mecab.txt') # mecab version

    tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-large-japanese', additional_special_tokens=['<person>'])
    make_split_train(save_dir, X_train, tokenizer, file_name='split-train-mecab-wordpiece.txt') # mecab wordpiece version

    print("duplicating werewolf dataset")
    X_train, y_train = duplicate_werewolves(X_train, y_train)
    X_valid, y_valid = duplicate_werewolves(X_valid, y_valid)
    X_test, y_test = duplicate_werewolves(X_test, y_test)

    for _nested_utterances, _labels, type in zip([X_train, X_valid, X_test], [y_train, y_valid, y_test], ['train', 'valid', 'test']):
        X_dfs, num_utters = [], []
        with trange(len(_nested_utterances), desc=f"pickling({type})...") as t:
            for _, _utterances in zip(t, _nested_utterances):
                X_df = pd.DataFrame({'raw_nested_utters': _utterances})
                X_dfs.append(X_df)
                num_utters.append(len(_utterances))
        df = pd.DataFrame({'nested_utters': X_dfs, 'num_utters': num_utters, 'labels': _labels})

        with open(f'{save_dir}/{type}.pkl', 'wb') as f:
            pickle.dump(df,  f, protocol=5)

def extract_loop(filePath, ROLES_USED):
    labels = []
    nested_utterances = [] #(user_num, utterance_num)
    participants = {}
    roles_label = {"人狼": 1, "狂人": 1, "村人": 0, "占い師": 0, "霊能者": 0, "狩人": 0, "共有者": 0, "ハムスター人間": 0}
    min_len_char = 10
    min_num_utter = 10

    with open(filePath, encoding='utf-8') as f:
        data = json.load(f)
        p = data.get('participants')
        participants.update(p)

    """processing each day"""
    days = data['days'][1:-1] # exclude prologue and epilogue

    """Aggregate all utterances of each player, respectively."""
    for participant in participants.keys():
        if participant == '楽天家 ゲルト': # exclude a bot player.
            continue
        participant_role = participants[participant]
        if participant_role not in ROLES_USED:
            continue
        _nested_utterance = []
        for day in days:
            for _utterance in day['utterances']:
                if _utterance.get('subject') == participant and _utterance.get('u_type') == 'say':
                    _utterance = clean_sent(_utterance['utterance'])
                    _utterance = replace_term(_utterance)

                    if len(_utterance) <= min_len_char:
                        continue
                    else:
                        _nested_utterance.append(_utterance)

        _nested_utterance = sorted(set(_nested_utterance), key=_nested_utterance.index) # remove duplications

        if len(_nested_utterance) <= min_num_utter:
            continue
        else:
            nested_utterances.append(_nested_utterance)
            labels.append(roles_label[participant_role])

    return nested_utterances, labels


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


def make_split_train(save_dir, X_train , tokenizer, file_name):
    """save split train data for fasttext training"""
    with open(f'{save_dir}/{file_name}', 'w') as f:
        for _nested_utterances in tqdm(X_train, desc=f"making {file_name}"):
            for _utterance in _nested_utterances:
                f.write(' '.join(tokenizer.tokenize(_utterance)))
                f.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--ROLES_USED", type=list, default={"人狼", "狂人", "村人", "占い師", "霊能者", "狩人"})

    args = parser.parse_args()

    files = sorted(glob.glob("../../../../corpus/BBSjsons/*/*.json"))  # 7249 files

    save_dir = f"../../data/nested"

    if args.sample:
        files = files[:3]
        save_dir = save_dir.replace("nested", "nested_sample")

    os.makedirs(save_dir, exist_ok=True)
    shutil.rmtree(save_dir)
    os.mkdir(save_dir)

    extract(files, save_dir, args.ROLES_USED)

    print('done!')
    if not args.sample:
        gmail_sender = Gmailsender(subject="実行終了通知")
        gmail_sender.send(body="mkNested.py終了")

if __name__ == '__main__':
    main()