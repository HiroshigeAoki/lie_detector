import argparse
import MeCab
import pandas as pd
import os
from tqdm import tqdm
import joblib
from collections import Counter


def tokenizer(text):
    text = text.replace('0', '').replace('<person>', '').replace(',', '').replace('.', '').replace('。', '').replace('、', '')
    mecab = MeCab.Tagger("-Owakati")
    parsed_text = mecab.parse(text)
    if parsed_text is None:
        return []
    return parsed_text.strip().split()


def flatten_docs(docs):
    _flatten_docs = []
    for doc in docs["nested_utters"]:
        _flatten_docs.extend(doc["raw_nested_utters"].tolist())
    return _flatten_docs


def tokenize(nested_doc):
    flattened_docs = flatten_docs(nested_doc)
    tokens = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(tokenizer)(utter) for utter in tqdm(flattened_docs)
    )
    return tokens


def generate_ngram_counts(tokens_list, n_gram):
    ngram_counts = Counter()
    for tokens in tokens_list:
        ngrams = generate_ngrams_for_sentence(tokens, n_gram)
        ngram_counts.update(ngrams)
    return ngram_counts


def generate_ngrams_for_sentence(words, n_gram):
    ngrams = set(map(lambda x: " ".join(x), zip(*[words[i:] for i in range(n_gram)])))
    return ngrams


def calc_diff_ngrams(target_label_ngram_counts, another_label_ngram_counts):
    diff_ngrams = {ngram: target_label_ngram_counts[ngram] - another_label_ngram_counts[ngram] for ngram in set(target_label_ngram_counts).union(set(another_label_ngram_counts))}
    sorted_ngrams = sorted(diff_ngrams.items(), key=lambda x: x[1], reverse=True)
    return sorted_ngrams


def main(args):
    data_dir = os.path.join("data", args.data)
    output_dir = os.path.join("data", args.data, "ngram")
    
    train = pd.read_pickle(os.path.join(data_dir, "test.pkl"))
    label_0 = train[train["labels"] == 0]
    label_1 = train[train["labels"] == 1]

    label_0_tokens = tokenize(label_0)
    label_1_tokens = tokenize(label_1)
    
    for n_gram in [1, 2, 3, 4, 5]:
        os.makedirs(os.path.join(os.path.join(output_dir, f"{n_gram}")), exist_ok=True)
        label_0_ngram_counts = generate_ngram_counts(label_0_tokens, n_gram)
        label_1_ngram_counts = generate_ngram_counts(label_1_tokens, n_gram)
        sorted_ngrams_0 = calc_diff_ngrams(label_0_ngram_counts, label_1_ngram_counts)
        pd.DataFrame(sorted_ngrams_0, columns=['ngram', 'difference']).to_csv(os.path.join(output_dir, f"{n_gram}", f"diff_{n_gram}_gram_0.csv"), index=False)
        sorted_ngrams_1 = calc_diff_ngrams(label_1_ngram_counts, label_0_ngram_counts)
        pd.DataFrame(sorted_ngrams_1, columns=['ngram', 'difference']).to_csv(os.path.join(output_dir, f"{n_gram}", f"diff_{n_gram}_gram_1.csv"), index=False)


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--data', type=str, default="exclude_bbs_nested_day_100")
    
    args = argparse.parse_args()
    main(args)



