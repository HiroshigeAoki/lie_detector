import argparse
import MeCab
import urllib3
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import os
import random


slothlib_path = 'http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt'
http = urllib3.PoolManager()
response = http.request('GET', slothlib_path)
text_data = response.data.decode('utf-8')
stop_words = [item for item in text_data.split('\r\n') if item != '']


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
        _flatten_docs.extend(doc["raw_nested_utters"])
    return _flatten_docs


def get_top_ngram_tfidf_scores(doc, n_gram, label, top_n=1000):
    sample_size = min(8000, len(doc))
    sample_doc = random.sample(doc, sample_size)
    vectorizer = TfidfVectorizer(
        tokenizer=tokenizer, stop_words=stop_words, ngram_range=(n_gram, n_gram))
    tfidf_matrix = vectorizer.fit_transform(["".join(sample_doc)])
    feature_array = np.array(vectorizer.get_feature_names())
    
    word_list, word_score = [], []
    for idx, tfidf in sorted(enumerate(tfidf_matrix.toarray()[0]), key=lambda x: x[1], reverse=True)[:top_n]:
        word_list.append(feature_array[idx])
        word_score.append(f"{tfidf: .2f}")
    return {
            f"word_{label}": word_list,
            f"score_{label}": word_score
        }


def main(args):
    data_dir = os.path.join("data", args.data)
    output_dir = os.path.join("data", args.data, "tfidf")
    
    train = pd.read_pickle(os.path.join(data_dir, "test.pkl"))
    label_0 = train[train["labels"] == 0]
    label_1 = train[train["labels"] == 1]

    label_0_doc = flatten_docs(label_0)
    label_1_doc = flatten_docs(label_1)
    
    for n_gram in [1, 2, 3, 4, 5]:
        os.makedirs(os.path.join(os.path.join(output_dir, f"{n_gram}")), exist_ok=True)
        label_0_dict = get_top_ngram_tfidf_scores(label_0_doc, n_gram, 0)
        label_1_dict = get_top_ngram_tfidf_scores(label_1_doc, n_gram, 1)
        label_0_dict.update(label_1_dict)
        pd.DataFrame(label_0_dict).to_csv(os.path.join(os.path.join(output_dir, f"{n_gram}"), "tfidf.csv"), index=False)
        
        label_0_unique_words = set(label_0_dict[f"word_0"]) - set(label_1_dict[f"word_1"])
        label_1_unique_words = set(label_1_dict[f"word_1"]) - set(label_0_dict[f"word_0"])
        pd.DataFrame({"unique_0": list(label_0_unique_words), "unique_1": list(label_1_unique_words)}).to_csv(os.path.join(os.path.join(output_dir, f"{n_gram}"), f"unique_words_{n_gram}.csv"), index=False)


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--data', type=str, default="exclude_bbs_nested_day")
    
    args = argparse.parse_args()
    main(args)
