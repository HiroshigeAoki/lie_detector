import pandas as pd
from transformers import BertJapaneseTokenizer
import os, sys
sys.path.append(os.pardir)
from utils.gmail_send import Gmailsender


def count_subword(df, tokenizer) -> int:
    count = []
    for _, row in df.iteritems():
        for _, utter in row.iteritems():
            tokenized = tokenizer.tokenize(utter)
            count.append(len(tokenized))
    return sum(count)


def main():
    sender = Gmailsender()
    sender.send("処理開始")
    train = pd.read_pickle("../../data/nested/train.pkl")
    valid = pd.read_pickle("../../data/nested/valid.pkl")
    test = pd.read_pickle("../../data/nested/test.pkl")

    all_num_utter = pd.concat((train.loc[:,"num_utters"], valid.loc[:,"num_utters"], test.loc[:,"num_utters"]), axis=0, ignore_index=True)
    fig_utter = all_num_utter.hist(bins=80, grid=True , xlabelsize=8, ylabelsize=8).set_xticks(ticks=list(range(0,180,10)))
    fig_utter[0].figure.savefig("../../data/nested/num_utter_hist.png")

    tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-v2', additional_special_tokens=['<person>'])
    all_utter = pd.concat((train.loc[:,"nested_utters"], valid.loc[:,"nested_utters"], test.loc[:,"nested_utters"]), axis=0, ignore_index=True)
    subword_count = all_utter.apply(count_subword, tokenizer=tokenizer)
    fig_mor = subword_count.hist(bins=200, grid=True, xrot=20, xlabelsize=16, ylabelsize=16).set_xticks(ticks=list(range(0,150,2)))
    fig_mor[0].figure.savefig("../../data/nested/num_sub_hist.png")
    sender.send("histogram作成終了。")

if __name__ == "__main__":
    main()