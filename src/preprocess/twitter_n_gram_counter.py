import pickle
from glob import glob
import re
import os
from tqdm import tqdm
import argparse
import joblib
from collections import Counter
import neologdn
import itertools


import os, sys
sys.path.append("./src")
from src.preprocess.ngram_utils import generate_ngrams_for_sentence

def generate_ngrams_from_file(file, n_gram):
    with open(file, "r") as f:
        lines = f.readlines()
    ngrams = []
    for line in lines:
        ngrams.extend(generate_ngrams_for_sentence([neologdn.normalize(token.lower()) for token in line.split()], n_gram=n_gram))
    return ngrams

def main(args):
    path = "/home/share/corpus/twitter/dentsu/"
    files = glob(path + "*.txt")
    os.makedirs("data/twitter", exist_ok=True)

    n_gram = args.n_gram

    n_gram_counter = Counter(list(itertools.chain(*joblib.Parallel(n_jobs=len(files))(joblib.delayed(generate_ngrams_from_file)(file, n_gram) for file in tqdm(files, desc="Splitting files")))))
    
    with open(f"data/twitter/{n_gram}_gram_counter.pkl", "wb") as pickle_file:
        pickle.dump(n_gram_counter, pickle_file)
    
    print("a")

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--n_gram", type=int, default=2)
    args = argparse.parse_args()
    main(args)
