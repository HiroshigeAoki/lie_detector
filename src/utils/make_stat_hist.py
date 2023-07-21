import pandas as pd
from pandarallel import pandarallel
from transformers import BertJapaneseTokenizer
import os, sys
from src.tokenizer.SPTokenizer import SentencePieceTokenizer


def count_subword(df, tokenizer) -> int:
    count = []
    for _, row in df.iteritems():
        for _, utter in row.iteritems():
            tokenized = tokenizer.tokenize(utter)
            count.append(len(tokenized))
    return count


def main():
    train = pd.read_pickle("data/nested/train.pkl")
    valid = pd.read_pickle("data/nested/valid.pkl")
    test = pd.read_pickle("data/nested/test.pkl")

    all_num_utter = pd.concat((train.loc[:,"num_utters"], valid.loc[:,"num_utters"], test.loc[:,"num_utters"]), axis=0, ignore_index=True)
    fig_utter = all_num_utter.hist(bins=80, grid=True , xlabelsize=8, ylabelsize=8).set_xticks(ticks=list(range(0,210,10)))
    fig_utter[0].figure.savefig("data/nested/num_utter_hist.png")

    # tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-v2', additional_special_tokens=['<person>'])
    tokenizer = SentencePieceTokenizer(model_file="model/sentencepiece/werewolf.model", do_lower_case=True)
    all_utter = pd.concat((train.loc[:,"nested_utters"], valid.loc[:,"nested_utters"], test.loc[:,"nested_utters"]), axis=0, ignore_index=True)
    pandarallel.initialize()
    subword_count = all_utter.parallel_apply(count_subword, tokenizer=tokenizer).explode(ignore_index=True)
    fig_sub = subword_count.hist(bins=150, grid=True , xlabelsize=8, ylabelsize=8).set_xticks(ticks=list(range(0,600,50)))
    fig_sub[0].figure.savefig("data/nested/num_sub_hist.png")

if __name__ == "__main__":
    main()
