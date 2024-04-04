import argparse
import os
from tqdm import tqdm
import joblib


import os, sys

sys.path.append("./src")
from src.preprocess.ngram.utils import save_ngram_tfidf_diff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="nested_day")
    args = parser.parse_args()

    data_dir = os.path.join("data", args.data)
    output_dir = os.path.join("data", "nested_day", "twitter")
    os.makedirs(output_dir, exist_ok=True)
    twitter_path = "/home/share/corpus/twitter/dentsu/tweet_wakati1.txt"

    def process_row(row):
        return (
            row.replace("<person>", "")
            .replace("0", "")
            .replace(",", "")
            .replace(".", "")
            .replace("。", "")
            .replace("、", "")
            .split()
        )

    print("start loading werewolf data")

    with open(os.path.join(data_dir, "train-mecab-parsed.txt"), "r") as file:
        werewolf_tokens = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(process_row)(line) for line in tqdm(file)
        )
    print("done")

    print("start loading twitter data")
    with open(twitter_path, "r") as file:
        twitter_tokens = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(process_row)(line) for line in tqdm(file)
        )
    print("done")

    for n_gram in [1, 2, 3, 4, 5]:
        save_ngram_tfidf_diff(output_dir, werewolf_tokens, twitter_tokens, n_gram)


if __name__ == "__main__":
    main()
