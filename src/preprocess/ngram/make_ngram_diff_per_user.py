import argparse
import pandas as pd
import os
from collections import Counter
import pickle
from tqdm import tqdm
import joblib

import os, sys

sys.path.append("./src")
from src.preprocess.ngram.utils import (
    calc_diff_ngrams,
    calculate_vocab_count,
)


def process_ngram(n_gram, data_dir, output_dir, label_0_user_num, label_1_user_num):
    with open(os.path.join(output_dir, f"{n_gram}_gram_0.pkl"), "rb") as f:
        label_0_ngrams_counters = pickle.load(f)
    with open(os.path.join(output_dir, f"{n_gram}_gram_1.pkl"), "rb") as f:
        label_1_ngrams_counters = pickle.load(f)

        # ユーザーごとに各トークン1回ずつカウント
        print(f"Calculating vocab count for {n_gram}-gram...")
        label_0_vocab_count = calculate_vocab_count(label_0_ngrams_counters)
        label_1_vocab_count = calculate_vocab_count(label_1_ngrams_counters)

        # 引き算の前を保存
        os.makedirs(os.path.join(output_dir, "raw"), exist_ok=True)
        pd.DataFrame(
            label_0_vocab_count.most_common(), columns=["ngram", "count"]
        ).to_csv(
            os.path.join(output_dir, "raw", f"{n_gram}_gram_count_0.csv"), index=False
        )
        pd.DataFrame(
            label_1_vocab_count.most_common(), columns=["ngram", "count"]
        ).to_csv(
            os.path.join(output_dir, "raw", f"{n_gram}_gram_count_1.csv"), index=False
        )
        print(f"Calculating diff count for {n_gram}-gram...")

        # プレイヤー数で正規化
        os.makedirs(os.path.join(output_dir, "normed_diff"), exist_ok=True)
        normed_0_count = Counter(
            dict(
                map(
                    lambda kv: (kv[0], kv[1] / label_0_user_num),
                    label_0_vocab_count.items(),
                )
            )
        )
        normed_1_count = Counter(
            dict(
                map(
                    lambda kv: (kv[0], kv[1] / label_1_user_num),
                    label_1_vocab_count.items(),
                )
            )
        )
        pd.DataFrame(
            calc_diff_ngrams(normed_0_count, normed_1_count),
            columns=["ngram", "difference"],
        ).to_csv(
            os.path.join(output_dir, "normed_diff", f"{n_gram}_gram_count_0.csv"),
            index=False,
        )
        pd.DataFrame(
            calc_diff_ngrams(normed_1_count, normed_0_count),
            columns=["ngram", "difference"],
        ).to_csv(
            os.path.join(output_dir, "normed_diff", f"{n_gram}_gram_count_1.csv"),
            index=False,
        )

        # 正規化無し
        os.makedirs(os.path.join(output_dir, "diff"), exist_ok=True)
        pd.DataFrame(
            calc_diff_ngrams(label_0_vocab_count, label_1_vocab_count),
            columns=["ngram", "difference"],
        ).to_csv(
            os.path.join(output_dir, "diff", f"{n_gram}_gram_count_0.csv"), index=False
        )
        pd.DataFrame(
            calc_diff_ngrams(label_1_vocab_count, label_0_vocab_count),
            columns=["ngram", "difference"],
        ).to_csv(
            os.path.join(output_dir, "diff", f"{n_gram}_gram_count_1.csv"), index=False
        )


def main(args):
    data_dir = os.path.join("data", args.data)
    output_dir = os.path.join("data", args.data, args.tokenizer_name, "ngram")

    os.makedirs(output_dir, exist_ok=True)

    parsed_0 = pd.read_pickle(
        os.path.join(data_dir, f"label_0_tokens_{args.tokenizer_name}.pkl")
    )
    parsed_1 = pd.read_pickle(
        os.path.join(data_dir, f"label_1_tokens_{args.tokenizer_name}.pkl")
    )

    label_0_user_num = len(parsed_0)
    label_1_user_num = len(parsed_1)

    joblib.Parallel(n_jobs=5)(
        joblib.delayed(process_ngram)(
            n_gram, data_dir, output_dir, label_0_user_num, label_1_user_num
        )
        for n_gram in range(1, 6)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--tokenizer_name", type=str, default="mecab")
    # parser.add_argument('--data', type=str, default="nested_day/under_sampling_160000")
    parser.add_argument("--data", type=str, default="nested_day_twitter")
    parser.add_argument("--tokenizer_name", type=str, default="hierbert")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    main(args)
