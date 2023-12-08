import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import  sys
sys.path.append('./src/')
from tokenizer.SPTokenizer import SentencePieceTokenizer
from utils.wiki_extract import wiki_extract

class SPtokenizer():
    def __init__(self, model_file='model/sentencepiece/werewolf.model', do_lower_case=True):
        self.tokenizer = SentencePieceTokenizer(model_file=model_file, do_lower_case=do_lower_case)
    def tokenize(self, utter: str) -> list[str]:
        parsed_utter = self.tokenizer.tokenize(utter)
        return parsed_utter

def collapse(tokenized):
    corpus = []
    for utters in tokenized:
        joint_utters = ''
        for utter in utters:
            joint_utters += ' '.join(utter)
        corpus.append(joint_utters)
    return corpus

def main():
    train = pd.read_pickle('data/nested/train.pkl')
    tokenizer = SPtokenizer()
    tokenized = []

    for utters in train['nested_utters']:
        tokenized.append(utters['raw_nested_utters'].apply(tokenizer.tokenize))

    corpus = collapse(tokenized)

    wiki_text = wiki_extract(99999999)
    corpus.extend(collapse(wiki_text))

    # https://www.haya-programming.com/entry/2018/07/09/190819

    vectorizer = TfidfVectorizer(min_df=0.03)
    tfidf_X = vectorizer.fit_transform(corpus).toarray()  # ぜんぶで1万データくらいあるけど、そんなに要らないので1000件取っている

    index = tfidf_X.argsort(axis=1)[:,::-1]

    feature_names = np.array(vectorizer.get_feature_names())
    feature_words = feature_names[index]

    n = 20  # top何単語取るか
    m = len(train)  # 何記事サンプルとして抽出するか
    top_words = []
    for fwords in feature_words[:m,:n]:
        # 各文書ごとにtarget（ラベル）とtop nの重要語を表示
        top_words.extend(fwords)

    pd.to_pickle(Counter(top_words), 'data/top_words.pkl')

if __name__ == "__main__":
    main()