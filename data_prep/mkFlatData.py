import json
import glob
from pathlib import Path
from tqdm import trange
import os, sys
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

sys.path.append(os.pardir)
from cleaner import clean_sent
from utils.parser import parse_args
from utils.unix_command import mkdir, mkdirs


def random_under_sampling(X, y):
    rus = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    return pd.concat([X_resampled, y_resampled], axis=1)


def each_extract(filePath, ROLES_USED):
    """１つのファイルから必要なところを前処理して取り出します。"""
    utterances = []
    labels = []
    participants = {}
    roles_and_tags_b = {"人狼": 1, "狂人": 1, "村人": 0, "占い師": 0, "霊能者": 0, "狩人": 0, "共有者": 0, "ハムスター人間": 0}

    with open(filePath, encoding='utf-8') as f:
        data = json.load(f)
        p = data.get('participants')
        participants.update(p)

    days = data['days'][1:-1] # omit epilogue and ending

    """omit monologue, wisper and etc"""
    for day in days:
        for utterance in day['utterances']:
            subject = utterance.get('subject')
            role = participants.get(subject)
            if utterance.get('u_type') == 'say'\
                    and len(utterance['utterance']) != 0\
                    and subject != '楽天家 ゲルト'\
                    and role in ROLES_USED:
                utter = clean_sent(utterance['utterance'])
                if len(utter) != 0:
                    utterances.append(utter)
                    labels.append(roles_and_tags_b[role])
                else:
                    continue
    return utterances, labels


def extract(filePaths, path, ROLES_USED):
    """ファイルから必要なところを取り出してリストにまとめます"""
    utterances = []
    labels = []

    with trange(len(filePaths), desc="extracting... ") as t:
        for num, filePath in zip(t, filePaths):
            file_utterances, file_labels = each_extract(filePath, ROLES_USED)
            utterances.extend(file_utterances)
            labels.extend(file_labels)

    X_train, X_valid_test, y_train, y_valid_test = train_test_split(
        utterances,labels, test_size=0.2, stratify=labels, random_state=0
    )

    X_valid, X_test, y_valid, y_test = train_test_split(
        X_valid_test, y_valid_test, test_size=.5, stratify=y_valid_test, random_state=0
    )

    train_df = pd.DataFrame([X_train, y_train], index=['text', 'label']).T
    valid_df = pd.DataFrame([X_valid, y_valid], index=['text', 'label']).T
    test_df = pd.DataFrame([X_test, y_test], index=['text', 'label']).T

    balanced_train_df = random_under_sampling(pd.DataFrame({'text':X_train}), pd.DataFrame({'label':y_train}))
    balanced_valid_df = random_under_sampling(pd.DataFrame({'text':X_valid}), pd.DataFrame({'label':y_valid}))
    balanced_test_df = random_under_sampling(pd.DataFrame({'text':X_test}), pd.DataFrame({'label':y_test}))

    unbalance_dir = path / 'unbalance'
    balance_dir = path / 'balance'

    mkdirs(unbalance_dir)
    mkdirs(balance_dir)

    train_df.to_pickle(unbalance_dir / 'train.pkl')
    valid_df.to_pickle(unbalance_dir / 'valid.pkl')
    test_df.to_pickle(unbalance_dir / 'test.pkl')

    balanced_train_df.to_pickle(balance_dir / 'train.pkl')
    balanced_valid_df.to_pickle(balance_dir / 'valid.pkl')
    balanced_test_df.to_pickle(balance_dir / 'test.pkl')

    # for sentencepiece
    text_dir = Path('../tokenizer/flat')
    mkdir(text_dir)
    with open(text_dir / 'train.txt', 'w') as f:
        for row in X_train:
            f.write(f"{row}\n")


    with open(path / 'stats.txt', 'w') as f:
        print(f"総発話数:{len(utterances):,}", file=f)
        print(f"市民発話数:{labels.count(0):,}, {(labels.count(0) / len(labels)) * 100:.2f}%", file=f)
        print(f"人狼陣営総発話数:{labels.count(1):,}, {(labels.count(1) / len(labels)) * 100:.2f}%", file=f)

        print('---'*10+'unbalanced data'+'---'*10, file=f)
        print(f"train:{len(train_df.index):,}発話, {(len(train_df.index) / len(labels)) * 100:.2f}%  \
                \n\t市民{train_df['label'].value_counts()[0]:,}, {train_df['label'].value_counts()[0] / len(train_df.index) * 100:.2f}% \
                \n\t人狼{train_df['label'].value_counts()[1]:,}, {train_df['label'].value_counts()[1] / len(train_df.index) * 100:.2f}%"
                , file=f
        )
        print(f"valid:{len(valid_df.index):,}発話, {(len(valid_df.index) / len(labels)) * 100:.2f}% \
                \n\t市民{valid_df['label'].value_counts()[0]:,}, {valid_df['label'].value_counts()[0] / len(valid_df.index) * 100:.2f}% \
                \n\t人狼{valid_df['label'].value_counts()[1]:,}, {valid_df['label'].value_counts()[1] / len(valid_df.index) * 100:.2f}%"
                , file=f
        )
        print(f"test:{len(test_df.index):,}発話, {(len(test_df.index) / len(labels)) * 100:.2f}% \
                \n\t市民{test_df['label'].value_counts()[0]:,}, {test_df['label'].value_counts()[0] / len(test_df.index) * 100:.2f}% \
                \n\t人狼{test_df['label'].value_counts()[1]:,}, {test_df['label'].value_counts()[1] / len(test_df.index) * 100:.2f}%"
                , file=f
        )

        print('---'*10+'balanced data'+'---'*10, file=f)
        n_balanced = len(balanced_train_df.index) + len(balanced_valid_df.index) + len(balanced_test_df.index)

        print(f"train:{len(balanced_train_df.index):,}発話, {(len(balanced_train_df.index) / n_balanced) * 100:.2f}%  \
                \n\t市民{balanced_train_df['label'].value_counts()[0]:,}, {balanced_train_df['label'].value_counts()[0] / len(balanced_train_df.index) * 100:.2f}% \
                \n\t人狼{balanced_train_df['label'].value_counts()[1]:,}, {balanced_train_df['label'].value_counts()[1] / len(balanced_train_df.index) * 100:.2f}%"
                , file=f
        )
        print(f"valid:{len(balanced_valid_df.index):,}発話, {(len(balanced_valid_df.index) / n_balanced) * 100:.2f}% \
                \n\t市民{balanced_valid_df['label'].value_counts()[0]:,}, {balanced_valid_df['label'].value_counts()[0] / len(balanced_valid_df.index) * 100:.2f}% \
                \n\t人狼{balanced_valid_df['label'].value_counts()[1]:,}, {balanced_valid_df['label'].value_counts()[1] / len(balanced_valid_df.index) * 100:.2f}%"
                , file=f
        )
        print(f"test:{len(balanced_test_df.index):,}発話, {(len(balanced_test_df.index) / n_balanced) * 100:.2f}% \
                \n\t市民{balanced_test_df['label'].value_counts()[0]:,}, {balanced_test_df['label'].value_counts()[0] / len(balanced_test_df.index) * 100:.2f}% \
                \n\t人狼{balanced_test_df['label'].value_counts()[1]:,}, {balanced_test_df['label'].value_counts()[1] / len(balanced_test_df.index) * 100:.2f}%"
                , file=f
        )


def main(args):
    ROLES_USED = args.m_role_dict.values()
    files = glob.glob("../../../corpus/BBSjsons/*/*.json")  # 7249 files
    dataPath = Path(f"../model/data/flat")
    extract(files, dataPath, ROLES_USED)


if __name__ == "__main__":
    args = parse_args()
    main(args)