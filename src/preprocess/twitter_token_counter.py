import pickle
from glob import glob
import os
from tqdm import tqdm
import joblib
from collections import Counter
import itertools
import neologdn
import numpy as np
import matplotlib.pyplot as plt


#japanese_pattern = re.compile(r'[\u3000-\u303F\u3040-\u309F\u30A0-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]')

def split_file(file):
    with open(file, "r") as f:
        lines = f.readlines()
    token_list = []
    for line in lines:
        token_list.extend([neologdn.normalize(token.lower()) for token in line.split()])
    return token_list



def main():
    path = "/home/share/corpus/twitter/dentsu/"
    files = glob(path + "*.txt")
    os.makedirs("data/twitter", exist_ok=True)
    token_counter = Counter(list(itertools.chain(*joblib.Parallel(n_jobs=1)(joblib.delayed(split_file)(file) for file in tqdm(files, desc="Splitting files")))))
    token_counter = Counter(list(itertools.chain(*joblib.Parallel(n_jobs=len(files))(joblib.delayed(split_file)(file) for file in tqdm(files, desc="Splitting files")))))
    
    with open("data/twitter/token_counter.pkl", "wb") as pickle_file:
        pickle.dump(token_counter, pickle_file)
    # with open("data/twitter/token_counter.pkl", "rb") as pickle_file:
    #     token_counter = pickle.load(pickle_file)
    
    filtered_counter = Counter({token: count for token, count in token_counter.items() if count > 10})
    all_counts = np.array(list(filtered_counter.values()))

    # 四分位数を計算
    quartiles = np.percentile(all_counts, [25, 50, 75])
    counts = Counter(filtered_counter.values())

    # ヒストグラムのためのデータとビンを準備
    values, frequencies = zip(*counts.items())

    # ヒストグラムを描画
    plt.figure(figsize=(10, 6))
    plt.bar(values, frequencies, width=np.log10(values), align='center', log=True)
    for quartile in quartiles:
        plt.axvline(x=quartile, color='r', linestyle='--')
        plt.text(quartile, plt.gca().get_ylim()[1], f'{int(quartile)}', horizontalalignment='right', color='red')

    plt.xlabel('Token Occurrence Counts (Log Scale)')
    plt.ylabel('Frequency')
    plt.xscale('log')  # x軸を対数スケールに設定
    plt.title('Histogram of Token Occurrence Counts')
    plt.savefig('data/twitter/counts_dist_log_scale.png')
    plt.show()
    
    filtered_counter = Counter({token: count for token, count in token_counter.items() if count > token_counter.get("人狼")})
    with open("data/twitter/token_counter_filtered.pkl", "wb") as pickle_file:
        pickle.dump(filtered_counter, pickle_file)
    
    print("\n".join(list(map(lambda x: f"{x[0]}: {x[1]}",  filtered_counter.most_common()))), file=open("data/twitter/token_counter_filtered.csv", "w"))

if __name__ == "__main__":
    main()