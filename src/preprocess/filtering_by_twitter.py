import pickle
import pandas as pd
from pandarallel import pandarallel
import itertools
from collections import Counter

import os, sys
sys.path.append("./src")
from src.preprocess.ngram_utils import tokenize

pandarallel.initialize(progress_bar=True)


def tokenize_nested_utters(nested_utters, token_set):
    excluded_tokens = []
    df = pd.DataFrame(dict(raw_nested_utters=nested_utters["raw_nested_utters"].apply(tokenize, token_set=token_set, full_width_person_token=False, excluded_tokens=excluded_tokens).apply(lambda x: "".join(x))))
    return df, excluded_tokens
    
def main():
    output_dir = f"data/nested_day_twitter"
    os.makedirs(output_dir, exist_ok=True)
    
    excluded_tokens_counter = Counter()
    
    data_dir = "data/nested_day"
    
    with open(f"data/twitter/token_counter_filtered.pkl", "rb") as pickle_file:
        token_counter_filtered = pickle.load(pickle_file)
    token_set = set(token_counter_filtered.keys())
    token_set.add("<person>")
    
    data = []
    if os.path.exists(os.path.join(data_dir, "train.pkl")):
        train = pd.read_pickle(os.path.join(data_dir, "train.pkl"))
        data.append((train, "train"))
    if os.path.exists(os.path.join(data_dir, "valid.pkl")):
        valid = pd.read_pickle(os.path.join(data_dir, "valid.pkl"))
        data.append((valid, "valid"))
    test = pd.read_pickle(os.path.join(data_dir, "test.pkl"))
    data.append((test, "test"))
    
    for df, mode in data:
        results = df["nested_utters"].parallel_apply(lambda x: tokenize_nested_utters(x, token_set))
        df["nested_utters"] = results.apply(lambda x: x[0])
        excluded_tokens_lists = results.apply(lambda x: x[1]).tolist()
        excluded_tokens_counter.update(itertools.chain(*excluded_tokens_lists))
        df.to_pickle(os.path.join(output_dir, f"{mode}.pkl"))
        print(f"Finish {mode} data")
    
    with open(f"data/nested_day_twitter/excluded_tokens_counter.pkl", "wb") as pickle_file:
        pickle.dump(excluded_tokens_counter, pickle_file)
        
    print("\n".join(list(map(lambda x: f"{x[0]}: {x[1]}",  excluded_tokens_counter.most_common()))), file=open("data/nested_day_twitter/excluded_tokens_count.txt", "w"))

if __name__ == "__main__":
    main()
