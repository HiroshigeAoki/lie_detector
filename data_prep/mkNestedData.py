# -*- coding: utf-8 -*-///
import json
import glob
from pathlib import Path
from tqdm import trange
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import os, sys

sys.path.append(os.pardir)
from data_prep.cleaner import clean_sent, replace_term, auto_exclude_sent
from utils.parser import parse_args
from utils.unix_command import mkdirs

# TODO: ランダムアンダーサンプリングして、バランスデータを作る
# TODO: 統計量をきちんと出す。

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def random_under_sampling(X, y):
    rus = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    return pd.concat([X_resampled, y_resampled], axis=1)

def output(input, outputPath, flag):
    if flag == 'json':  # list comes
        for type in ["train", "valid", "test"]:
            with open(outputPath / f"{type}.json", "w") as outfile:
                for dict in input[type]:
                    json.dump(dict, outfile, ensure_ascii=False, cls=MyEncoder)
                    outfile.write("\n")

    elif flag == 'text':
        """for fasttext"""
        with open(outputPath, "w") as outfile:
            for utterance in input:
                outfile.write(utterance + '\n')


def splitter(d, total, train_rate=.6, dev_rate=.2):
    dicts = {}
    train = []
    valid = []
    test = []

    dicts.update({'train': train, 'valid': valid, 'test': test})

    train_end_point = int(total * train_rate)
    valid_end_point = int(total * dev_rate) + train_end_point

    train += d[:train_end_point]
    valid += d[train_end_point:valid_end_point]
    test += d[valid_end_point:]
    return dicts


def each_extract(filePath, ROLES_USED, exclude=True, auto=True, classifier=None):
    utter_ans_dict = []
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
        list_utterance = []
        for day in days:
            for utterance in day['utterances']:
                if utterance.get('subject') == participant \
                        and utterance.get('u_type') == 'say':
                    utter = clean_sent(utterance['utterance'])
                    utter = replace_term(utter)

                    if exclude and auto:
                        # If True then skip.
                        BBS_term, type = classifier(utter)
                        if BBS_term:
                            deleted_utters.append(f"type: {type}\n{utter}")
                            continue

                    if len(utter) == 0:
                        continue
                    else:
                        list_utterance.append(utter)

        if len(list_utterance) == 0:
            continue
        else:
            nested_b = list(np.array(list_utterance).reshape(-1, 1))
            dict_b = {'utterances': nested_b, 'answer': roles_and_tags_b[participant_role]}

            """adding"""
            utter_ans_dict.append(dict_b)

    return utter_ans_dict, deleted_utters


def extract(filePaths, path, textPath, ROLES_USED, exclude=True, auto=True):
    """ファイルから必要なところを取り出してリストにまとめます"""
    utter_ans_dicts = []
    train_utterances = []
    deleted_utters = []

    if exclude and auto:
        classifier = auto_exclude_sent()

    with trange(len(filePaths), desc="extracting... ") as t:
        for _, filePath in zip(t, filePaths):
            utter_ans_dict, deleted_utter\
                = each_extract(filePath, ROLES_USED, exclude, auto, classifier)
            utter_ans_dicts += utter_ans_dict
            deleted_utters += deleted_utter

    train_valid_test_dict = splitter(utter_ans_dicts, len(utter_ans_dicts), train_rate=.8, dev_rate=.1)

    for dict in train_valid_test_dict["train"]:
        utter = [utter for utters in dict["utterances"] for utter in utters]
        train_utterances += utter

    output(train_valid_test_dict, path, flag='json')
    output(train_utterances, textPath, flag='text')
    output(deleted_utters, path / "deleted_utters.txt", flag='text')


def main():
    args = parse_args()
    ROLES_USED = args.m_role_dict.values()

    files = glob.glob("../../../corpus/BBSjsons/*/*.json")  # 7249 files
    path = Path(f"../model/data")
    textPath = f"../tokenizer/train.txt"

    if args.sample:
        files = files[:10]
        path = path / f"{'auto' if args.auto else 'man'}"

    extract(files, path, textPath, ROLES_USED, exclude=args.exclude, auto=args.auto)


if __name__ == '__main__':
    main()