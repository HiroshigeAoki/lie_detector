import argparse
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

hedges = ['かも', 'かもしれん','そうかもしれない','なのかな','あるかな','そうかな','っぽい','ぽい', 'たぶん','多分','恐らく', 'おそらく', 'めいた', 'ちょっと','すこし','少し', 'とか', 'のほう', 'みたい', 'みたいな', 'ていうか', 'よう', 'げ', 'そうだ', 'そう', 'けど', '結構', 'けっこう', 'だいたい', '大体', '大抵', 'たり', '感じ', 'かんじ'] # https://www.lang.nagoya-u.ac.jp/nichigen/issue/pdf/4/4-18.pdf
hesitations = ['あ', 'あー', 'ああ', 'あああ', 'う', 'ううう','うーん', 'え', 'ええ', 'えええ', 'えーと', 'お', 'おお', 'おおお']
self_references = ['私','わたし','わたしゃ','わたしたち', '私自身','私的','私たち','私達','私見','私について','私の主張','私事','私のように','私の部屋', '私ゃ','明日の私','私が死んでも','私どう','僕','僕自身','僕ら','僕たち','僕のこと','僕は思う','僕たち', '僕達','僕が悪い','僕の話','ぼく','ぼくが','ぼくら','俺','俺様','おれ','おれっち','あたし','あたしゃ','うち', 'あたい', 'あちき', 'わし','わしゃ', 'おいどん', '拙者', '某']
cognitive_words = ['思う', 'おもう', '思うに', '思い出す', 'おもいだす', '思い出し', 'おもいだし', '思いつく', 'おもいつく', '思いつい', 'おもいつい', '考え','かんがえ','考える','かんがえる','かんがえなおす', '信じ', 'しんじ', '信じる', 'しんじる', '覚える', 'おぼえる', '覚え', 'おぼえ', '称賛', '比較', '比べる', '対比', '批判', '守る', '判断', '違う', 'ちがう', '反対', 'はんたい', '支援', '賛成', 'さんせい', '確認', '了解', '確証', '分類', '編集', '作曲', '構成', '作る', 'つくる', '設計', '生み出す', '予測'] # https://courses.washington.edu/pharm439/Bloomstax.htm
exclusion_words = ['以外', '他', 'その他', '除い', '除く', '除外', '別']
negations = ['しない', 'ない', 'ぬ', 'ざる', 'ず', 'ません', 'ありません', '違う', '違い', '拒否', '却下', '断る'] # https://niwasaburoo.amebaownd.com/posts/5757774/
motion_words = ["行く", "いく", "開く", "ひらく", "閉じる", "とじる", "押す", "おす", "持つ", "もつ", "取る", "とる", "投げる", "なげる", "落とす", "落ち", "おとす", "打つ", "立つ", "たつ", "読む", "よむ", "書く", "描く", "歩く", "あるく", "走る", "はしる", "到着", "食べる", "たべる"]
sense_words = ["知覚", "触覚", "触感", "触る", "ふれる", "さわる", "感覚", "感じる", "かんじる", "感じ", "かんじ", "味覚", "食欲", "嗅覚", "匂う", "におう", "聴覚", "聞く", "きく", "聞こえた", "聞こえ", "きこえ", "聞こえる", "きこえる", "聞いた", "きいた","視覚", "見る", "観る", "みる"] # http://assets.flips.jp/files/users/ichimai-quiz/joyo.pdf

col_names = [
    'num_morphemes',
    'positive',
    'negative',
    'polarity',
    'self_ref',
    'cognitive',
    'hedges',
    'hesitations',
    'exlusion_words',
    'negations',
    'motion_words',
    'sense_words',
    'noun',
    'verb',
    'particle',
    'adjective',
    'conjunction',
    'connections',
]


def feature_counter(nested_df):
    pandarallel.initialize()
    nested_df.loc[:, 'nested_utters'] = nested_df.loc[:, 'nested_utters'].parallel_apply(_feature_counter)
    #nested_df.loc[:, 'nested_utters'] = nested_df.loc[:, 'nested_utters'].apply(_feature_counter)
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
        count['exlusion_words'].append(len([mor for mor in mors if mor in exclusion_words]))
        count['negations'].append(len([mor for mor in mors if mor in negations]))
        count['motion_words'].append(len([mor for mor in mors if mor in motion_words]))
        count['sense_words'].append(len([mor for mor in mors if mor in sense_words]))
        count['hesitations'].append(len([mor for mor in mors if mor in hesitations]))

        pos = [mor.split("\t")[1].split(",")[0] for mor in parsed_line.split("\n")[:-2]] # 品詞をカウント
        pos_count = Counter(pos)
        count['noun'].append(pos_count['名詞'])
        count['verb'].append(pos_count['動詞'])
        count['particle'].append(pos_count['助詞'])
        count['adjective'].append(pos_count['形容詞'])
        count['conjunction'].append(pos_count['接続詞'])

        count['connections'].append(pos_count['接続詞'] + pos_count['助詞'])
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', action='store_true')
    args = parser.parse_args()
    data_dir = 'data/nested_unbalanced'
    data_dir_JDC = 'data/JDC'
    save_dir = 'data/features'
    if args.sample:
        data_dir = 'data/nested_sample'
        save_dir = 'data/features_sample'
    os.makedirs(save_dir, exist_ok=True)

    print('loading..')
    train = pd.read_pickle(os.path.join(data_dir, 'train.pkl'))
    valid = pd.read_pickle(os.path.join(data_dir, 'valid.pkl'))
    test = pd.read_pickle(os.path.join(data_dir, 'test.pkl'))

    test_JDC = pd.read_pickle(os.path.join(data_dir_JDC, 'test.pkl'))

    print('extracting..')
    train = feature_counter(train)
    print('train finish')
    valid = feature_counter(valid)
    print('valid finish')
    test = feature_counter(test)
    print('test finish')
    test_JDC = feature_counter(test_JDC)
    print('test_JDC finished')

    train.to_pickle(os.path.join(save_dir, 'train.pkl'))
    valid.to_pickle(os.path.join(save_dir, 'valid.pkl'))
    test.to_pickle(os.path.join(save_dir, 'test.pkl'))
    test_JDC.to_pickle(os.path.join(save_dir, 'test_JDC.pkl'))
    gmail_sender = Gmailsender(subject="実行終了通知")
    gmail_sender.send(body="make_feature_data.py終了")


if __name__ == "__main__":
    main()