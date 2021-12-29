import json
import glob
import pickle
from tqdm import trange, tqdm
import pandas as pd
from transformers import BertJapaneseTokenizer
import shutil
import argparse
import os, sys
import joblib
sys.path.append('./src/')
from preprocess.cleaner import clean_sent, replace_term
from utils.gmail_send import Gmailsender
from preprocess.custom_mecab_tagger import CustomMeCabTagger


def extract(filePaths, save_dir, kwargs):
    """ファイルから必要なところを取り出してリストにまとめます"""
    nested_utterances = []
    labels = []
    users = []
    deleted = []

    outputs = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(extract_loop)(
            filepath,
            kwargs
        ) for filepath in tqdm(filePaths, desc='extracting...')
    )

    for _nested_utterances, _labels, _users, _deleted in outputs:
        nested_utterances += _nested_utterances
        labels += _labels
        users += _users
        deleted += _deleted

    grouped = pd.DataFrame(
        dict(
            nested_utterances=nested_utterances,
            labels=labels,
            users=users
        )
    ).groupby('users')
    groups = [grouped.get_group(x) for x in grouped.groups]

    print('split data')
    i = 0
    train, train_size = [groups[i]], len(groups[i])
    while train_size < len(labels) * 0.8:
        i += 1
        train_size += len(groups[i])
        train.append(groups[i])

    valid, valid_size = [groups[i]], len(groups[i])
    while valid_size < len(labels) * 0.1:
        i += 1
        valid_size += len(groups[i])
        valid.append(groups[i])

    test, test_size = [groups[i]], len(groups[i])
    while test_size < len(labels)* 0.1:
        i += 1
        test_size += len(groups[i])
        test.append(groups[i])

    train.extend(groups[i+1:])

    train, valid, test = pd.concat(train), pd.concat(valid), pd.concat(test)
    X_train, y_train, users_train = train['nested_utterances'].tolist(), train['labels'].tolist(), train['users'].tolist()
    X_valid, y_valid, users_valid = valid['nested_utterances'].tolist(), valid['labels'].tolist(), valid['users'].tolist()
    X_test, y_test, users_test = test['nested_utterances'].tolist(), test['labels'].tolist(), test['users'].tolist()

    train_for_tapt = []
    for train in X_train:
        train_for_tapt.extend(train)
        train_for_tapt.append("")

    bbs_data = train_for_tapt + deleted

    with open(f'{save_dir}/bbs.txt', 'w') as f: # for tapt pretraining of RoBERTa.
        for utterance in tqdm(bbs_data, desc="making bbs.txt"):
            f.write(utterance + '\n')

    tokenizer, path, _data = []

    tokenizer.append(CustomMeCabTagger("-O wakati -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd -r /home/haoki/Documents/vscode-workplaces/lie_detector/src/tokenizer/mecab_userdic/mecabrc"))
    path.append(os.path.join(save_dir, 'split-train-mecab.txt'))
    _data.append(bbs_data)

    tokenizer.append(BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-large-japanese', additional_special_tokens=['<person>']))
    path.append(os.path.join(save_dir, 'split-train-mecab-wordpiece.txt'))
    _data.append(bbs_data)

    list_args = [a for a in zip(tokenizer, path, _data)]

    joblib.Parallel(n_jobs=2)(
        joblib.delayed(make_split_train)(
            *args,
        ) for args in tqdm(list_args, desc='making split train data...')
    )

    print("duplicating werewolf dataset")
    X_train, y_train, users_train = duplicate_werewolves(X_train, y_train, users_train)
    X_valid, y_valid, users_valid = duplicate_werewolves(X_valid, y_valid, users_valid)
    X_test, y_test, users_test = duplicate_werewolves(X_test, y_test, users_test)

    for _nested_utterances, _labels, _users, type in zip([X_train, X_valid, X_test], [y_train, y_valid, y_test], [users_train, users_valid, users_test], ['train', 'valid', 'test']):
        X_dfs, num_utters = [], []
        with trange(len(_nested_utterances), desc=f"pickling({type})...") as t:
            for _, _utterances in zip(t, _nested_utterances):
                X_df = pd.DataFrame({'raw_nested_utters': _utterances})
                X_dfs.append(X_df)
                num_utters.append(len(_utterances))
        df = pd.DataFrame({'nested_utters': X_dfs, 'num_utters': num_utters, 'labels': _labels, 'users': _users})

        with open(f'{save_dir}/{type}.pkl', 'wb') as f:
            pickle.dump(df,  f, protocol=5)


def extract_loop(filePath, kwargs):
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

    nested_utterances, labels, users, _deleted = preprocess(days=days, participants=participants, users_dict=users_dict, **kwargs)
    deleted.extend(_deleted)
    _delete_nested_utterances, _, _, _deleted = preprocess(days=[epilogue, prologue], participants=participants, users_dict=users_dict, **kwargs)
    for utterances in _delete_nested_utterances:
        delete_nested_utterances.extend(utterances)
        delete_nested_utterances.append("")
    deleted.extend(delete_nested_utterances + deleted)

    return nested_utterances, labels, users, deleted


def preprocess(days, participants, users_dict, role2label , used_role, min_len_char, min_num_utter):
    nested_utterances = [] #(user_num, utterance_num)
    labels = []
    users = []
    deleted = []
    """Aggregate all utterances of each player, respectively."""
    for participant in participants.keys():
        if participant == '楽天家 ゲルト': # exclude a bot player.
            continue
        participant_role = participants[participant]
        if participant_role not in used_role:
            continue
        _nested_utterance = []
        _deleted = []
        for i, day in enumerate(days):
            for utterance_inf in day['utterances']:
                if utterance_inf.get('subject') == participant:
                    utterance = clean_sent(utterance_inf['utterance'])
                    utterance = replace_term(utterance)
                    if len(utterance) <= min_len_char:
                        continue
                    if utterance_inf.get('u_type') == 'say':
                        _nested_utterance.append(utterance)
                    else:
                        _deleted.append(utterance)
        _nested_utterance = sorted(set(_nested_utterance), key=_nested_utterance.index) # remove duplications
        _deleted = sorted(set(_deleted), key=_deleted.index)
        _deleted.append("")
        deleted.extend(_deleted)
        if len(_nested_utterance) > min_num_utter:
            nested_utterances.append(_nested_utterance)
            labels.append(role2label[participant_role])
            users.extend(get_keys_from_value(users_dict, participant))

    return nested_utterances, labels, users, deleted


def get_keys_from_value(d, val):
    return [k for k, v in d.items() if v == val]


def duplicate_werewolves(nested_utterances, labels, users):
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

    for i in range(difference - len(werewolf_indices)):
        werewolf_indices.append(werewolf_indices[i])

    for i in werewolf_indices:
        nested_utterances.append(nested_utterances[i])
        labels.append(1)
        users.append(users[i])

    return nested_utterances, labels, users


def make_split_train(tokenizer, path, data):
    """save split train data for fasttext training"""
    with open(path, 'w') as f:
        for utterance in data:
            f.write(' '.join(tokenizer.tokenize(utterance + '\n')))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true")

    args = parser.parse_args()

    files = sorted(glob.glob("../../corpus/BBSjsons/*/*.json"))  # 7249 files

    save_dir = f"data/nested_separate_users"

    if args.sample:
        files = files[:100]
        save_dir = save_dir.replace("nested", "nested_sample")

    os.makedirs(save_dir, exist_ok=True)
    shutil.rmtree(save_dir)
    os.mkdir(save_dir)

    kwargs = dict(
        role2label={"人狼": 1, "狂人": 1, "村人": 0, "占い師": 0, "霊能者": 0, "狩人": 0, "共有者": 0, "ハムスター人間": 0},
        used_role=["人狼", "狂人", "村人", "占い師", "霊能者", "狩人"],
        min_len_char=10,
        min_num_utter=10
    )

    extract(files, save_dir, kwargs)

    print('done!')
    if not args.sample:
        gmail_sender = Gmailsender(subject="実行終了通知")
        gmail_sender.send(body="mkNested.py終了")

if __name__ == '__main__':
    main()
