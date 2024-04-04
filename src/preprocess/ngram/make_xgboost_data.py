import os
import argparse
import pandas as pd
import json
from tqdm import tqdm
from pandarallel import pandarallel
import joblib

import sys

sys.path.append("./src")
from trashbin.add_mecab_parse import add_mecab_parsed
from src.preprocess.ngram.utils import generate_ngrams_for_sentence


pandarallel.initialize(progress_bar=True)


def create_ngram_features(nested_utters, ngram_dict):
    ngram_features = {}
    for n_gram in range(1, 6):
        total_ngram = 0
        ngrams_list = []
        for tokens in nested_utters["parsed_nested_utters"].tolist():
            ngrams = generate_ngrams_for_sentence(tokens, n_gram)
            ngrams_list.extend(ngrams)
            total_ngram += len(ngrams)
        normed_counts = {
            key: ngrams_list.count(key) / total_ngram
            for key in ngram_dict[f"{n_gram}_gram"]
        }
        ngram_features.update(normed_counts)
    return ngram_features


def create_ngram_dict(count_dir, n_gram, top_n):
    ngram_label_0 = pd.read_csv(
        f"{count_dir}/ngram/normed_diff/{n_gram}_gram_count_0.csv"
    )["ngram"][:top_n].tolist()
    ngram_label_1 = pd.read_csv(
        f"{count_dir}/ngram/normed_diff/{n_gram}_gram_count_1.csv"
    )["ngram"][:top_n].tolist()
    ngrams = ngram_label_0 + ngram_label_1
    return {f"{n_gram}_gram": list(map(lambda n: n.replace(" ", ""), ngrams))}


def create_ngram_features_for_df(df, ngram_dict):
    ngram_features = pd.DataFrame(
        df["nested_utters"]
        .parallel_apply(create_ngram_features, args=(ngram_dict,))
        .tolist()
    )
    return pd.concat((ngram_features, df["labels"]), axis=1)


def main(args):
    data = args.data
    count = args.count
    top_n = args.top_n

    data_name = data.split("/")[0]
    count_data_name = count.split("/")[0]

    data_dir = os.path.join("data", data)
    count_dir = os.path.join("data", count)

    if data_name != count_data_name:
        output_dir = os.path.join("data", count, "xgboost", f"top_{top_n}", data)
    elif data_name == count_data_name:
        output_dir = os.path.join("data", data, "xgboost", f"top_{top_n}")
    os.makedirs(output_dir, exist_ok=True)

    datasets = []
    if os.path.exists(os.path.join(data_dir, "train.pkl")):
        train = pd.read_pickle(os.path.join(data_dir, "train.pkl")).drop(
            ["users"], axis=1
        )
        datasets.append((train, "train"))
    if os.path.exists(os.path.join(data_dir, "valid.pkl")):
        valid = pd.read_pickle(os.path.join(data_dir, "valid.pkl")).drop(
            ["users"], axis=1
        )
        datasets.append((valid, "valid"))
    if os.path.exists(os.path.join(data_dir, "test.pkl")):
        test = pd.read_pickle(os.path.join(data_dir, "test.pkl")).drop(
            ["users"], axis=1
        )
        datasets.append((test, "test"))

    print("creating ngram dict...")
    ngram_dicts = joblib.Parallel(n_jobs=5)(
        joblib.delayed(create_ngram_dict)(count_dir, n_gram, top_n)
        for n_gram in range(1, 6)
    )
    ngram_dict = {}
    for ngram_dict_ in ngram_dicts:
        ngram_dict.update(ngram_dict_)

    for df, mode in tqdm(datasets, desc="Creating ngram features"):
        if not data_dir.endswith("mecab_parsed"):
            df = add_mecab_parsed(df)
        df = create_ngram_features_for_df(df, ngram_dict)
        df.to_pickle(os.path.join(output_dir, f"{mode}.pkl"))
        print(f"Finish {mode} data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="nested_day/parsed",
        help="XGBoost用のデータを作りたいデータ(parsedを作らなくてもいい)",
    )
    parser.add_argument(
        "--count",
        type=str,
        default="nested_day/under_sampling_160000",
        help="ngramのカウントデータがあるディレクトリ。このカウントの内、各n_gramの各ラベル毎に上位top_nを特徴量として使う。なので、学習データとテストデータのcountディレクトリは同じものを使う必要がある。",
    )
    parser.add_argument("--top_n", type=int, default=100)

    args = parser.parse_args()
    main(args)
