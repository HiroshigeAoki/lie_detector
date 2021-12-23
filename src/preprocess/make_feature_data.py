import pandas as pd
import numpy as np
from collections import Counter
from pandarallel import pandarallel
import MeCab
import oseti
import itertools
import os, sys
sys.path.append('./src/')
from preprocess.cleaner import clean_sent
from utils.gmail_send import Gmailsender

mecab_args = "-r /home/haoki/Documents/vscode-workplaces/lie_detector/src/tokenizer/mecab_userdic/mecabrc -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd"

hedges = ['かも', 'かもしれん','そうかもしれない','かな','なのかな','あるかな','そうかな','っぽい','たぶん','多分','恐らく','めいた','ちょっと','すこし','少し']
self_references = ['私', '私自身','私的','私たち','私達','私見','私は羊','私について','私の主張','私事','私のように','私の部屋','実は私は','私ゃ','明日の私','私が死んでも','私どう','僕','僕自身','僕ら','僕たち','僕のこと','僕は思う','僕達','僕は君','下僕','僕が悪い','僕の話','老僕','ぼく','ぼくが','ぼくら','わたし','わたしゃ','わたしたち','俺','俺様','おれ','おれっち','あたし','あたしゃ','わし','わしゃ']
cognitive_words = ['おもう', '思う','思うに','考え','考える','かんがえる','かんがえ','かんがえなおす']

col_names = [
        'num_morphemes',
        'positive', 'negative', 'polarity',
        'self_ref', 'cognitive', 'hedges',
        'sign', 'noun', 'prefix', 'verb', 'particle', 'adjective', 'conjunction', 'auxiliary_verb'
]


def feature_counter(nested_df):
    pandarallel.initialize()
    nested_df.loc[:, 'nested_utters'] = nested_df.loc[:, 'nested_utters'].parallel_apply(_feature_counter)
    nested_df = player_feature_count(nested_df)
    nested_df = calc_TTR(nested_df)
    nested_df = nested_df.drop('nested_utters', axis=1)
    return nested_df


def _feature_counter(df: pd.DataFrame):
    tagger = MeCab.Tagger(mecab_args)
    analyzer = oseti.Analyzer(mecab_args)
    raw_nested_utters = df.loc[:, 'raw_nested_utters'].apply(tagger.parse)
    count = {col_name: [] for col_name in col_names}
    list_mors = []
    for (_, line), (_, parsed_line) in zip(df.loc[:, 'raw_nested_utters'].items(), raw_nested_utters.items()):
        line = clean_sent(line)
        polarity = analyzer.count_polarity(line) # 東北大の極性辞書を使って、polarityを計算
        count['positive'].append(sum([pol['positive'] for pol in polarity]))
        count['negative'].append(sum([pol['negative'] for pol in polarity]))
        matched_total = sum([pol['positive'] + pol['negative'] for pol in polarity])
        count['polarity'].append(sum([pol['positive'] - pol['negative'] for pol in polarity]) / matched_total if matched_total != 0 else 0) # average

        count['num_morphemes'].append(len(parsed_line.split('\n')) - 1) # exclude EOS

        mors = [mor.split("\t")[0] for mor in parsed_line.split("\n")[:-2]] # 形態素のリスト。EOSトークン除外
        count['self_ref'].append(len([mor for mor in mors if mor in self_references]))
        count['cognitive'].append(len([mor for mor in mors if mor in cognitive_words]))
        count['hedges'].append(len([mor for mor in mors if mor in hedges]))

        pos = [mor.split("\t")[1].split(",")[0] for mor in parsed_line.split("\n")[:-2]] # 品詞をカウント
        pos_count = Counter(pos)
        count['sign'].append(pos_count['記号'])
        count['noun'].append(pos_count['名詞'])
        count['prefix'].append(pos_count['接頭詞'])
        count['verb'].append(pos_count['動詞'])
        count['particle'].append(pos_count['助詞'])
        count['adjective'].append(pos_count['形容詞'])
        count['conjunction'].append(pos_count['接続詞'])
        count['auxiliary_verb'].append(pos_count['助動詞'])
        list_mors.append(mors) # To calculate TTR.
    return pd.concat((df, pd.DataFrame(count), pd.DataFrame({'mors': list_mors})), axis=1)


def player_feature_count(df):
    for col_name in col_names:
        df[f"{col_name}_sum"], df[f"{col_name}_mean"], df[f"{col_name}_std"] = sum_mean_std(df, col_name)
    return df


def sum_mean_std(df, col_name):
    sum, mean, std = [], [], []
    for nested_utters_df in df.loc[:,'nested_utters']:
        sum.append(np.sum(nested_utters_df.loc[:,col_name]))
        mean.append(np.mean(nested_utters_df.loc[:,col_name]))
        std.append(np.std(nested_utters_df.loc[:,col_name]))
    return sum, mean, std


def calc_TTR(df):
    TTR = []
    for row in df.itertuples():
        mors = list(itertools.chain.from_iterable(row.nested_utters.mors.tolist()))
        counter = Counter(mors)
        TTR.append(len(counter) / row.num_morphemes_sum)
    df['TTR'] = TTR
    return df


def main():
    data_dir = 'data/nested'
    print('loading..')
    train = pd.read_pickle(os.path.join(data_dir, 'train.pkl'))
    valid = pd.read_pickle(os.path.join(data_dir, 'valid.pkl'))
    test = pd.read_pickle(os.path.join(data_dir, 'test.pkl'))

    print('extracting..')
    train = feature_counter(train)
    print('train finish')
    valid = feature_counter(valid)
    print('valid finish')
    test = feature_counter(test)
    print('test finish')

    save_dir = '../../data/features'
    #save_dir = '../../data/features_sample'
    os.makedirs(save_dir, exist_ok=True)

    train.to_pickle(f'{save_dir}/train.pkl')
    valid.to_pickle(f'{save_dir}/valid.pkl')
    test.to_pickle(f'{save_dir}/test.pkl')
    gmail_sender = Gmailsender(subject="実行終了通知")
    gmail_sender.send(body="make_hierarchical_data.py終了")


if __name__ == "__main__":
    main()