import pandas as pd
import numpy as np
from pandarallel import pandarallel
from transformers import BertJapaneseTokenizer
import argparse
import os, sys
sys.path.append('./src/')
from utils.gmail_send import Gmailsender

tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-large-japanese', additional_special_tokens=['<person>'])

def feature_counter(nested_df):
    pandarallel.initialize()
    nested_df.loc[:, 'nested_utters'] = nested_df.loc[:, 'nested_utters'].parallel_apply(_feature_counter)
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

    return [f'{player_num:,}({civil_num:,}/{werewolf_num:,})', sum_utter_num, ave_utter_num, max_utter_num, min_utter_num, sum_sub_num, ave_sub_num_per_player, max_sub_num_per_player, min_sub_num_per_player, ave_sub_num_per_utter, max_sub_num_per_utter, min_sub_num_per_utter]


def make_stats_table(train, valid, test):
    index = ['プレイヤー数(人狼/市民)', '合計発話数', '平均発話数', '最大発話数', '最小発話数', 'サブワード数合計', '1プレイヤーにおける平均サブワード数', '1プレイヤーにおける最大サブワード数', '1プレイヤーにおける最小サブワード数', '1発話における平均サブワード数', '1発話における最大サブワード数', '1発話における最小サブワード数']
    train_row = calc_stats(train)
    valid_row = calc_stats(valid)
    test_row = calc_stats(test)
    all_row = calc_stats(pd.concat((train,valid,test),axis=0))
    stats_table = pd.DataFrame({'train': train_row, 'valid': valid_row, 'test': test_row, 'all': all_row}, index=index)
    stats_table.iloc[1:] = stats_table.iloc[1:].applymap(lambda x: f'{x:,.2f}')

    return stats_table


def main():
    sender = Gmailsender()
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default='nested')
    data_dir = 'data/nested'
    train = pd.read_pickle(f'{data_dir}/train.pkl')
    valid = pd.read_pickle(f'{data_dir}/valid.pkl')
    test = pd.read_pickle(f'{data_dir}/test.pkl')

    train = feature_counter(train)
    valid = feature_counter(valid)
    test = feature_counter(test)

    stats_table = make_stats_table(train, valid, test)
    stats_table.to_csv(f'{data_dir}/stats_mecab_wordpiece.csv')
    sender.send('統計量計算完了')

if __name__ == "__main__":
    main()