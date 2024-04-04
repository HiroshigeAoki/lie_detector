import argparse
import pandas as pd
import os
import pickle
from tqdm import tqdm
from joblib import Parallel, delayed
from collections import Counter

import os, sys

sys.path.append("./src")
from src.preprocess.ngram.utils import (
    generate_ngrams_counter_per_user,
    tokenize_group,
)


def process_user_group(group):
    return pd.concat(
        [row["raw_nested_utters"] for row in group[1]["nested_utters"].tolist()]
    ).tolist()


def process_all_groups(groups):
    return Parallel(n_jobs=-1)(
        delayed(process_user_group)(group)
        for group in tqdm(groups, desc="Processing groups")
    )


def _calculate_vocab_count(ngrams_counter):
    return Counter(next(iter(ngrams_counter.values())).keys())


def calculate_vocab_count(ngrams_counters):
    combined_counter = Counter()
    for ngrams_counter in tqdm(ngrams_counters, desc="Calculating vocab count"):
        partial_counter = _calculate_vocab_count(ngrams_counter)
        combined_counter.update(partial_counter)
    return combined_counter


def main(args):
    data_dir = os.path.join("data", args.data)
    output_dir = os.path.join("data", args.data, args.tokenizer_name, "ngram")
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data...")
    if args.test:
        data = pd.read_pickle(os.path.join(data_dir, "test.pkl"))
    else:
        data = pd.read_pickle(os.path.join(data_dir, "train.pkl"))

    label_0 = data[data["labels"] == 0]
    label_1 = data[data["labels"] == 1]

    label_0_grouped = list(label_0.groupby("users"))
    label_1_grouped = list(label_1.groupby("users"))

    label_0_user_names = [group[0] for group in label_0_grouped]
    label_1_user_names = [group[0] for group in label_1_grouped]

    print("Processing label 0 utters...")
    label_0_utters_per_user = process_all_groups(label_0_grouped)
    print("Processing label 1 utters...")
    label_1_utters_per_user = process_all_groups(label_1_grouped)

    exclude_tokens = [
        "＜",
        "＞",
        "person",
    ]  # , '＞_＜', '＞_', '_＜', 'ジムソン', '＞_※＜', '⑥', 'ーダ', 'kwsk', '?＞＞＜', '片片', '服そ', 'ゃら', 'ゲス', '※', 'ワクテカ', 'ーー',
    # '時分時', '上部', 'wk', 'ーダー', '*:', 'やけん', '真神', 'na', 'マル', '白斑', '継が', '検分', '偶発', '＞_＼＜', '宿帳', '④', 'ソーダ', 'プギャー', '恥さらし',
    # 'エルニャ', '徳', 'ダイス', '霊媒', '独白', '*＜', '⑦', '①②', '譲渡', 'ささやけ', '亀裂', '当確', '出そろわ', '退学', '買い被ら', '号外', '@-', '締め切る',
    # 'アナウンス', 'パックンチョ', '※:＜', '!:\\', '①②③', '＞_*＜', '噛先', '発表', '更新', "ororor", "rl", "＞_②＜", "トリックオアトリー", "ぃぃぃいいい", "キッコリス",
    # "＞』『", "弐兵", "ミイラ", "=⇒＜", "dotch", "①⑤", "＞＞＜＜", "!②", "①④", "ージムソン", "?⇒＜", "①③", "①②③", "①②④", "①②⑤", "①②⑥", "①②⑦", "①②⑧", "rom",
    # "。:\\", "ーーー", "::\\", "===", "_＼", "@;", "?>>\\", "--", ":＼", "パーメーラー", "_:", "hage", "pta", "_:\\", "mjd", "_※_", "ds", "rom", "rd",
    # "ro", "ーッッ", "!=", "パーメーラー", "*_", "?+?", "クラムチャウダー", "ウコン", "キャットファイト", "ぉっふぉっふぉっ",
    # "ジャンピング", "嬢", "ジャミング", "アンチ", "弐霊", "暫白", "書抜き", "チョコチップクッキー", "ed", "ep", "テヘペロ", ",(", "()。", "ジャンピング", "白白", "三つ巴", "nd"]
    print("Tokenizing label 0 data...")

    label_0_tokens_list = tokenize_group(
        label_0_utters_per_user,
        exclude_tokens=exclude_tokens,
        full_width_person_token=True,
        tokenizer_name=args.tokenizer_name,
    )

    print("Tokenizing label 1 data...")
    label_1_tokens_list = tokenize_group(
        label_1_utters_per_user,
        exclude_tokens=exclude_tokens,
        full_width_person_token=True,
        tokenizer_name=args.tokenizer_name,
    )

    # Saving token data
    pd.DataFrame(
        {"users": label_0_user_names, "tokens": label_0_tokens_list}
    ).to_pickle(os.path.join(data_dir, f"label_0_tokens_{args.tokenizer_name}.pkl"))
    pd.DataFrame(
        {"users": label_1_user_names, "tokens": label_1_tokens_list}
    ).to_pickle(os.path.join(data_dir, f"label_1_tokens_{args.tokenizer_name}.pkl"))

    for n_gram in [1, 2, 3, 4, 5]:
        print(f"Processing {n_gram}-gram for label 0...")
        label_0_ngrams_counters = generate_ngrams_counter_per_user(
            label_0_tokens_list, label_0_user_names, n_gram
        )

        print(f"Processing {n_gram}-gram for label 1...")
        label_1_ngrams_counters = generate_ngrams_counter_per_user(
            label_1_tokens_list, label_1_user_names, n_gram
        )

        for i, counts in enumerate(
            tqdm(
                [label_0_ngrams_counters, label_1_ngrams_counters],
                desc=f"Saving {n_gram}-gram data",
            )
        ):
            file_path = os.path.join(output_dir, f"{n_gram}_gram_{i}.pkl")
            try:
                with open(file_path, "wb") as f:
                    pickle.dump(counts, f)
            except Exception as e:
                print(f"ファイルの保存に失敗しました: {file_path}")
                print(f"エラー詳細: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="nested_day_twitter")
    parser.add_argument("--tokenizer_name", type=str, default="hierbert")
    parser.add_argument("--test", action="store_true")
    # parser.add_argument("--tokenizer_name", type=str, default="mecab")
    # parser.add_argument('--data', type=str, default="nested_day/under_sampling_160000")

    args = parser.parse_args()
    main(args)
