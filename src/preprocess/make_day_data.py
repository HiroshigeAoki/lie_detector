import json
import glob
import pickle
from tqdm import trange, tqdm
import pandas as pd
import shutil
import argparse
import os, sys
import joblib
from collections import Counter
sys.path.append('./src/')
from preprocess.cleaner import clean_sent, replace_term


def extract(filePaths, save_dir, kwargs):
    """ファイルから必要なところを取り出してリストにまとめます"""
    save_dir = save_dir
    nested_utterances = []
    labels = []
    users = []
    deleted = []
    

    outputs = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(extract_loop)(
            filepath,
            kwargs,
        ) for filepath in tqdm(filePaths, desc='extracting...')
    )

    aggregated_replacement_track_dict = {}

    for _nested_utterances, _labels, _users, _deleted, partial_replacement_track_dict in tqdm(outputs, desc='aggregating joblib outputs...'):
        nested_utterances += _nested_utterances
        labels += _labels
        users += _users
        deleted += _deleted
        
        for key, value in partial_replacement_track_dict.items():
            if key not in aggregated_replacement_track_dict:
                aggregated_replacement_track_dict[key] = value
            else:
                aggregated_replacement_track_dict[key]['count'] += value['count']

    grouped = pd.DataFrame(
        dict(
            nested_utterances=nested_utterances,
            labels=labels,
            users=users
        )
    ).groupby('users')
    groups = [grouped.get_group(x) for x in grouped.groups]

    print('split data')
    # train, valid, testに同じユーザが跨らないように分割
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

    save_to_pickle(X_train, y_train, users_train, 'train', save_dir)
    save_to_pickle(X_valid, y_valid, users_valid, 'valid', save_dir)
    save_to_pickle(X_test, y_test, users_test, 'test', save_dir)
    
    for key in tqdm(aggregated_replacement_track_dict, desc='save replacement_track_dict'):
        aggregated_replacement_track_dict[key]['examples'] = list(aggregated_replacement_track_dict[key]['examples'])
    sorted_replacement_track_dict = dict(sorted(aggregated_replacement_track_dict.items(), key=lambda item: item[1]['count'], reverse=True))
    with open(os.path.join(save_dir, 'replacements.json'), 'w', encoding='utf-8') as f:
        json.dump(sorted_replacement_track_dict, f, ensure_ascii=False, indent=4)



def save_to_pickle(X, y, users, name, save_dir):
    X_dfs, num_utters = [], []
    with trange(len(X), desc=f"pickling({name})...") as t:
        for _, X in zip(t, X):
            X_df = pd.DataFrame({'raw_nested_utters': X})
            X_dfs.append(X_df)
            num_utters.append(len(X))
    df = pd.DataFrame({'nested_utters': X_dfs, 'num_utters': num_utters, 'labels': y, 'users': users})

    with open(f'{save_dir}/{name}.pkl', 'wb') as f:
        pickle.dump(df, f, protocol=5)


def extract_loop(filePath, kwargs):
    deleted = []
    delete_nested_utterances = []
    participants = {}
    users_dict = {}
    
    replacement_track_dict = {}

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

    nested_utterances, labels, users, _deleted = preprocess(days=days, participants=participants, users_dict=users_dict, replacement_track_dict=replacement_track_dict, **kwargs)
    deleted.extend(_deleted)
    _delete_nested_utterances, _, _, _deleted = preprocess(days=[epilogue, prologue], participants=participants, users_dict=users_dict, replacement_track_dict=replacement_track_dict, **kwargs)
    for utterances in _delete_nested_utterances:
        delete_nested_utterances.extend(utterances)
        delete_nested_utterances.append("")
    deleted.extend(delete_nested_utterances + deleted)

    return nested_utterances, labels, users, deleted, replacement_track_dict


def preprocess(days, participants, users_dict, role2label , used_role, min_len_char, min_num_utter, replacement_track_dict):
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
        _deleted = []
        for i, day in enumerate(days):
            _nested_utterance = []
            for utterance_inf in day['utterances']:
                if utterance_inf.get('subject') == participant:
                    utterance = clean_sent(utterance_inf['utterance'], replacement_track_dict)
                    utterance = replace_term(utterance, replacement_track_dict)
                    if len(utterance) <= min_len_char:
                        continue
                    if utterance_inf.get('u_type') == 'say':
                        _nested_utterance.append(utterance)
                    else:
                        _deleted.append(utterance)
            _nested_utterance = sorted(set(_nested_utterance), key=_nested_utterance.index) # remove duplications
            if len(_nested_utterance) > min_num_utter:
                nested_utterances.append(_nested_utterance)
                labels.append(role2label[participant_role])
                users.extend(get_keys_from_value(users_dict, participant))

        _deleted = sorted(set(_deleted), key=_deleted.index)
        _deleted.append("")
        deleted.extend(_deleted)

    return nested_utterances, labels, users, deleted


def get_keys_from_value(d, val):
    return [k for k, v in d.items() if v == val]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default='data/nested_day_unbalance')
    parser.add_argument("--sample", action="store_true")

    args = parser.parse_args()

    files = sorted(glob.glob("../..//Documents/corpus/BBSjsons/*/*.json"))  # 7249 files

    save_dir = args.save_dir

    if args.sample:
        files = files[:100]
        save_dir = save_dir.replace("nested_day", "nested_sample")

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


if __name__ == '__main__':
    main()