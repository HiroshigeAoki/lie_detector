import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandarallel import pandarallel
from transformers import BertJapaneseTokenizer, AutoTokenizer
import argparse
import sys
import matplotlib
import os
from typing import List, Tuple
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
sys.path.append('./src/')
from src.preprocess.ngram_utils import tokenize as mecab_tokenizer
pandarallel.initialize()

bert_tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-v3', additional_special_tokens=['<person>'])
bigbird_tokenizer = AutoTokenizer.from_pretrained('nlp-waseda/bigbird-base-japanese', additional_special_tokens=['<person>'])


def feature_counter(nested_df):
    nested_df.loc[:, 'nested_utters'] = nested_df.loc[:, 'nested_utters'].parallel_apply(_feature_counter)
    #nested_df.loc[:, 'nested_utters'] = nested_df.loc[:, 'nested_utters'].apply(_feature_counter)
    col_names = ['num_subwords_bert', 'num_subwords_bigbird', 'num_morphemes']
    nested_df = player_feature_count(nested_df, col_names)
    return nested_df

def _feature_counter(df: pd.DataFrame):
    bert_subwords = df.loc[:, 'raw_nested_utters'].apply(bert_tokenizer.tokenize)
    bigbird_subwords = df.loc[:, 'raw_nested_utters'].apply(bigbird_tokenizer.tokenize)
    morphemes = df.loc[:, 'raw_nested_utters'].apply(mecab_tokenizer)
    count = {
        'num_subwords_bert': [],
        'num_subwords_bigbird': [],
        'num_morphemes': [],
    }
    for _bert_subwords, _bigbird_subwords , _morphemes in zip(bert_subwords, bigbird_subwords, morphemes):
        count['num_subwords_bert'].append(len(_bert_subwords))
        count['num_subwords_bigbird'].append(len(_bigbird_subwords))
        count['num_morphemes'].append(len(_morphemes))
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
    return_dict = {}
    player_num = len(nested_df)
    civil_num = nested_df.loc[:,'labels'].value_counts()[0]
    werewolf_num = nested_df.loc[:,'labels'].value_counts()[1]
    return_dict["プレイヤー数(人狼/市民)"] = f'{player_num:,}({civil_num:,}/{werewolf_num:,})'
    
    return_dict["合計発話数"] = nested_df.loc[:,'num_utters'].sum()
    return_dict["平均発話数"] = nested_df.loc[:,'num_utters'].mean()
    return_dict["最大発話数"] = nested_df.loc[:,'num_utters'].max()
    return_dict["最小発話数"] = nested_df.loc[:,'num_utters'].min()
    
    for tokenizer in ["bert", "bigbird"]:
        return_dict[f"1プレイヤーにおける平均サブワード数({tokenizer})"] = nested_df.loc[:,f'num_subwords_{tokenizer}_sum'].mean()
        return_dict[f"1プレイヤーにおける最大サブワード数({tokenizer})"] = nested_df.loc[:,f'num_subwords_{tokenizer}_sum'].max()
        return_dict[f"1プレイヤーにおける最小サブワード数({tokenizer})"] = nested_df.loc[:,f'num_subwords_{tokenizer}_sum'].min()
        return_dict[f"1発話における平均サブワード数({tokenizer})"] = nested_df.loc[:,f'num_subwords_{tokenizer}_sum'].sum() / nested_df.loc[:,'num_utters'].sum()
        return_dict[f"1発話における最大サブワード数({tokenizer})"] = max([df.loc[:,f'num_subwords_{tokenizer}'].max() for df in nested_df.loc[:,'nested_utters']])
        return_dict[f"1発話における最小サブワード数({tokenizer})"] = min([df.loc[:,f'num_subwords_{tokenizer}'].min() for df in nested_df.loc[:,'nested_utters']])
    
    return_dict["1プレイヤーにおける平均形態素数"] = nested_df.loc[:,'num_morphemes_sum'].mean()
    return_dict["1プレイヤーにおける最大形態素数"] = nested_df.loc[:,'num_morphemes_sum'].max()
    return_dict["1プレイヤーにおける最小形態素数"] = nested_df.loc[:,'num_morphemes_sum'].min()
    return_dict["1発話における平均形態素数"] = nested_df.loc[:,'num_morphemes_sum'].sum() / nested_df.loc[:,'num_utters'].sum()
    return_dict["1発話における最大形態素数"] = max([df.loc[:,'num_morphemes'].max() for df in nested_df.loc[:,'nested_utters']])
    return_dict["1発話における最小形態素数"] = min([df.loc[:,'num_morphemes'].min() for df in nested_df.loc[:,'nested_utters']])
    
    return_dict["異なりユーザ数"] = nested_df['users'].nunique()
    
    return return_dict


def make_stats_table(dataset: List[Tuple[str, pd.DataFrame]], all: pd.DataFrame):
    stats_table = {name: calc_stats(df) for name, df in dataset}
    stats_table['all'] = calc_stats(all)
    stats_table = pd.DataFrame(stats_table)
    stats_table.iloc[1:] = stats_table.iloc[1:].applymap(lambda x: f'{x:,.2f}')

    return stats_table


def create_histogram(df, feature_col, title, filename):
    if feature_col == 'num_utters':
        bins = 20
        feature_name = '発話数'
    elif 'num_subwords' in feature_col:
        bins = 50
        feature_name = 'サブワード数'
    elif 'num_morphemes' in feature_col:
        bins = 50
        feature_name = '形態素数'
    
    plt.figure()
    plt.hist(df[feature_col], bins=bins, edgecolor='black')
    plt.title(title)
    plt.xlabel(feature_name)
    plt.ylabel('カウント')
    
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
    parser.add_argument('--data_dir', default='exclude_bbs_nested_day_100')
    args = parser.parse_args()
    data_dir = f'data/{args.data_dir}'
    output_dir = f'{data_dir}/stats'
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = []
    if os.path.exists(f'{data_dir}/train.pkl'):
        train = pd.read_pickle(f'{data_dir}/train.pkl')
        dataset.append(('train', train))
    if os.path.exists(f'{data_dir}/valid.pkl'):
        valid = pd.read_pickle(f'{data_dir}/valid.pkl')
        dataset.append(('valid', valid))
    if os.path.exists(f'{data_dir}/test.pkl'):
        test = pd.read_pickle(f'{data_dir}/test.pkl')
        if args.data_dir == "murder_mystery":
            test["num_utters"] = test["nested_utters"].apply(lambda x: len(x))
        dataset.append(('test', test))
    
    all = []
    for name, df in dataset:
        print(f'{name} data shape: {df.shape}')
        df = feature_counter(df)
        all.append(df)
        create_histogram(df, 'num_utters', f'{name.capitalize()} Utterances Histogram', f'{output_dir}/{name}_utterances_histogram.png')
        for tokenizer in ["bert", "bigbird"]:
            create_histogram(df, f'num_subwords_{tokenizer}_sum', f'{name.capitalize()} Subwords Histogram {tokenizer}', f'{output_dir}/{name}_subwords_histogram_{tokenizer}.png')
        create_histogram(df, 'num_morphemes_sum', f'{name.capitalize()} Morphemes Histogram', f'{output_dir}/{name}_morphemes_histogram.png')
        
    all = pd.concat(all)
    create_histogram(all, 'num_utters', 'All Utterances Histogram', f'{output_dir}/all_utterances_histogram.png')
    
    for tokenizer in ["bert", "bigbird"]:
        create_histogram(all, f'num_subwords_{tokenizer}_sum', f'All Subwords Histogram {tokenizer}', f'{output_dir}/all_subwords_histogram_{tokenizer}.png')
    create_histogram(all, 'num_morphemes_sum', f'{name.capitalize()} Morphemes Histogram', f'{output_dir}/{name}_morphemes_histogram.png')
    
    stats_table = make_stats_table(dataset=dataset, all=all)    
    stats_table_latex = stats_table.to_latex()
    with open(f'{output_dir}/stats_table.tex', 'w') as f:
        f.write(stats_table_latex)

if __name__ == "__main__":
    main()
