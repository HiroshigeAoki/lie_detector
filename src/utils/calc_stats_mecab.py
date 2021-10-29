import pandas as pd
import numpy as np
from collections import Counter
from pandarallel import pandarallel
import MeCab
import oseti

mecab_args = "-r /home/haoki/Documents/vscode-workplaces/lie_detector/project/tokenizer/mecab_userdic/mecabrc -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd"

hedges = ['かも', 'かもしれん','そうかもしれない','かな','なのかな','あるかな','そうかな','っぽい','たぶん','多分','恐らく','めいた','ちょっと','すこし','少し']
self_references = ['私', '私自身','私的','私たち','私達','私見','私は羊','私について','私の主張','私事','私のように','私の部屋','実は私は','私ゃ','明日の私','私が死んでも','私どう','僕','僕自身','僕ら','僕たち','僕のこと','僕は思う','僕達','僕は君','下僕','僕が悪い','僕の話','老僕','ぼく','ぼくが','ぼくら','わたし','わたしゃ','わたしたち','俺','俺様','おれ','おれっち','あたし','あたしゃ','わし','わしゃ']
cognitive_words = ['おもう', '思う','思うに','考え','考える','かんがえる','かんがえ','かんがえなおす']


def feature_counter(nested_df):
    pandarallel.initialize()
    nested_df.loc[:, 'nested_utters'] = nested_df.loc[:, 'nested_utters'].parallel_apply(_feature_counter)
    col_names = ['num_morphemes', 'positive', 'negative', 'polarity', 'self_ref', 'cognitive', 'hedges', 'noun', 'verb', 'adjective', 'conjunction', 'particle']
    nested_df = player_feature_count(nested_df, col_names)
    return nested_df

def _feature_counter(df: pd.DataFrame):
    tagger = MeCab.Tagger(mecab_args)
    analyzer = oseti.Analyzer(mecab_args)
    raw_nested_utters = df.loc[:, 'raw_nested_utters'].apply(tagger.parse)
    count = {'num_morphemes': [], 'positive': [], 'negative': [], 'polarity': [], 'self_ref': [], 'cognitive': [], 'hedges': [], 'noun': [], 'verb': [], 'adjective': [], 'conjunction': [], 'particle': []}
    for (_, line), (_, parsed_line) in zip(df.loc[:, 'raw_nested_utters'].items(), raw_nested_utters.items()):
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
        count['noun'].append(pos_count['名詞'])
        count['verb'].append(pos_count['動詞'])
        count['adjective'].append(pos_count['形容詞'])
        count['conjunction'].append(pos_count['接続詞'])
        count['particle'].append(pos_count['助詞'])
    return pd.concat((df, pd.DataFrame(count)), axis=1)

def sum_mean_std(df, col_name):
    return (
        [np.sum(nested_utters_df.loc[:,col_name]) for nested_utters_df in df.loc[:,'nested_utters']],
        [np.mean(nested_utters_df.loc[:,col_name]) for nested_utters_df in df.loc[:,'nested_utters']],
        [np.std(nested_utters_df.loc[:,col_name]) for nested_utters_df in df.loc[:,'nested_utters']]
    )

def player_feature_count(df, col_names):
    for col_name in col_names:
        df[f"{col_name}_sum"], df[f"{col_name}_mean"], df[f"{col_name}_std"] = sum_mean_std(df, col_name)
    return df

def calc_stats(nested_df):
    player_num = len(nested_df)
    civil_num = nested_df.loc[:,'labels'].value_counts()[0]
    werewolf_num = nested_df.loc[:,'labels'].value_counts()[1]
    sum_utter_num = nested_df.loc[:,'num_utters'].sum()
    ave_utter_num = nested_df.loc[:,'num_utters'].mean()
    max_utter_num = nested_df.loc[:,'num_utters'].max()
    min_utter_num = nested_df.loc[:,'num_utters'].min()
    sum_mor_num = nested_df.loc[:,'num_morphemes_sum'].sum()
    ave_mor_num_per_player = nested_df.loc[:,'num_morphemes_sum'].mean()
    max_mor_num_per_player = nested_df.loc[:,'num_morphemes_sum'].max()
    min_mor_num_per_player = nested_df.loc[:,'num_morphemes_sum'].min()
    ave_mor_num_per_utter = sum_mor_num / sum_utter_num
    max_mor_num_per_utter = max([df.loc[:,'num_morphemes'].max() for df in nested_df.loc[:,'nested_utters']])
    min_mor_num_per_utter = min([df.loc[:,'num_morphemes'].min() for df in nested_df.loc[:,'nested_utters']])

    return [f'{player_num:,}({civil_num:,}/{werewolf_num:,})', sum_utter_num, ave_utter_num, max_utter_num, min_utter_num, sum_mor_num, ave_mor_num_per_player, max_mor_num_per_player, min_mor_num_per_player, ave_mor_num_per_utter, max_mor_num_per_utter, min_mor_num_per_utter]


def make_stats_table(train, valid, test):
    index = ['プレイヤー数(人狼/市民)', '合計発話数', '平均発話数', '最大発話数', '最小発話数', '形態素数合計', '1プレイヤーにおける平均形態素数', '1プレイヤーにおける最大形態素数', '1プレイヤーにおける最小形態素数', '1発話における平均形態素数', '1発話における最大形態素数', '1発話における最小形態素数']
    train_row = calc_stats(train)
    valid_row = calc_stats(valid)
    test_row = calc_stats(test)
    stats_table = pd.DataFrame({'train': train_row, 'valid': valid_row, 'test': test_row}, index=index)
    stats_table.iloc[1:] = stats_table.iloc[1:].applymap(lambda x: f'{x:,.2f}')

    return stats_table


def main():
    data_dir = 'data/nested'
    #data_dir = 'data/nested_sample'
    train = pd.read_pickle(f'{data_dir}/train.pkl')
    valid = pd.read_pickle(f'{data_dir}/valid.pkl')
    test = pd.read_pickle(f'{data_dir}/test.pkl')

    train = feature_counter(train)
    valid = feature_counter(valid)
    test = feature_counter(test)

    stats_table = make_stats_table(train, valid, test)
    stats_table.to_csv(f'{data_dir}/stats_mecab.csv')

if __name__ == "__main__":
    main()