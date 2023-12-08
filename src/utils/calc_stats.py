import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandarallel import pandarallel
from transformers import BertJapaneseTokenizer
import argparse
import sys
sys.path.append('./src/')

tokenizer = BertJapaneseTokenizer.from_pretrained('ku-nlp/deberta-v2-large-japanese-char-wwm', additional_special_tokens=['<person>'])

def feature_counter(nested_df):
    pandarallel.initialize()
    nested_df.loc[:, 'nested_utters'] = nested_df.loc[:, 'nested_utters'].parallel_apply(_feature_counter)
    #nested_df.loc[:, 'nested_utters'] = nested_df.loc[:, 'nested_utters'].apply(_feature_counter)
    col_names = ['num_subwords']
    nested_df = player_feature_count(nested_df, col_names)
    return nested_df

def _feature_counter(df: pd.DataFrame):
    raw_nested_utters = df.loc[:, 'raw_nested_utters'].apply(tokenizer.tokenize)
    count = {'num_subwords': []}
    for utter in raw_nested_utters:
        count['num_subwords'].append(len(utter))
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
    sum_sub_num = nested_df.loc[:,'num_subwords_sum'].sum()
    ave_sub_num_per_player = nested_df.loc[:,'num_subwords_sum'].mean()
    max_sub_num_per_player = nested_df.loc[:,'num_subwords_sum'].max()
    min_sub_num_per_player = nested_df.loc[:,'num_subwords_sum'].min()
    ave_sub_num_per_utter = sum_sub_num / sum_utter_num
    max_sub_num_per_utter = max([df.loc[:,'num_subwords'].max() for df in nested_df.loc[:,'nested_utters']])
    min_sub_num_per_utter = min([df.loc[:,'num_subwords'].min() for df in nested_df.loc[:,'nested_utters']])
    unique_users_num = nested_df['users'].nunique()

    return [f'{player_num:,}({civil_num:,}/{werewolf_num:,})', sum_utter_num, ave_utter_num, max_utter_num, min_utter_num, sum_sub_num, ave_sub_num_per_player, max_sub_num_per_player, min_sub_num_per_player, ave_sub_num_per_utter, max_sub_num_per_utter, min_sub_num_per_utter, unique_users_num]


def make_stats_table(train, valid, test):
    index = ['プレイヤー数(人狼/市民)', '合計発話数', '平均発話数', '最大発話数', '最小発話数', 'サブワード数合計', '1プレイヤーにおける平均サブワード数', '1プレイヤーにおける最大サブワード数', '1プレイヤーにおける最小サブワード数', '1発話における平均サブワード数', '1発話における最大サブワード数', '1発話における最小サブワード数', '異なりユーザ数']
    train_row = calc_stats(train)
    valid_row = calc_stats(valid)
    test_row = calc_stats(test)
    all_row = calc_stats(pd.concat((train,valid,test),axis=0))
    stats_table = pd.DataFrame({'train': train_row, 'valid': valid_row, 'test': test_row, 'all': all_row}, index=index)
    stats_table.iloc[1:] = stats_table.iloc[1:].applymap(lambda x: f'{x:,.2f}')

    return stats_table


def create_histogram(df, feature_col, title, filename):
    plt.figure()
    plt.hist(df[feature_col], bins=20, edgecolor='black')
    plt.title(title)
    plt.xlabel(feature_col)
    plt.ylabel('Frequency')
    
    # 四分位数を計算
    q1 = df[feature_col].quantile(0.25)
    median = df[feature_col].quantile(0.5)
    q3 = df[feature_col].quantile(0.75)
    
    # 平均を計算
    mean = df[feature_col].mean()
    
    # 四分位数を赤線でプロット
    plt.axvline(q1, color='r', linestyle='--', label=f'Q1: {q1:.2f}')
    plt.axvline(median, color='r', linestyle='-', label=f'Median: {median:.2f}')
    plt.axvline(q3, color='r', linestyle='--', label=f'Q3: {q3:.2f}')
    
    # 平均を黄色線でプロット
    plt.axvline(mean, color='y', linestyle='-', label=f'Mean: {mean:.2f}')
    
    plt.legend()  # 凡例を表示
    plt.savefig(filename)  # 画像として保存



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='nested')
    args = parser.parse_args()
    data_dir = f'data/{args.data_dir}'
    train = pd.read_pickle(f'{data_dir}/train.pkl')
    valid = pd.read_pickle(f'{data_dir}/valid.pkl')
    test = pd.read_pickle(f'{data_dir}/test.pkl')

    train = feature_counter(train)
    valid = feature_counter(valid)
    test = feature_counter(test)
    
    # 発話数に関するヒストグラムの作成と保存
    create_histogram(train, 'num_utters', 'Train Utterances Histogram', f'{data_dir}/train_utterances_histogram.png')
    create_histogram(valid, 'num_utters', 'Valid Utterances Histogram', f'{data_dir}/valid_utterances_histogram.png')
    create_histogram(test, 'num_utters', 'Test Utterances Histogram', f'{data_dir}/test_utterances_histogram.png')

    # サブワード数に関するヒストグラムの作成と保存
    create_histogram(train, 'num_subwords_sum', 'Train Subwords Histogram', f'{data_dir}/train_subwords_histogram.png')
    create_histogram(valid, 'num_subwords_sum', 'Valid Subwords Histogram', f'{data_dir}/valid_subwords_histogram.png')
    create_histogram(test, 'num_subwords_sum', 'Test Subwords Histogram', f'{data_dir}/test_subwords_histogram.png')


    stats_table = make_stats_table(train, valid, test)
    stats_table = stats_table.to_latex()
    with open(f'{data_dir}/stats_mecab_wordpiece.tex', 'w') as f:
        f.write(stats_table)

if __name__ == "__main__":
    main()
