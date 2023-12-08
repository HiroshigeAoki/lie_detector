import argparse
import pandas as pd
import os


def under_sampling(data_dir, save_dir, num_train):
    num_vaild = num_train // 10
    num_test = num_train // 10
    train = pd.read_pickle(f"{data_dir}/train.pkl")
    valid = pd.read_pickle(f"{data_dir}/valid.pkl")
    test = pd.read_pickle(f"{data_dir}/test.pkl")
    
    sampling_data(train, num_train).to_pickle(f"{save_dir}/train.pkl")
    sampling_data(valid, num_vaild).to_pickle(f"{save_dir}/valid.pkl")
    sampling_data(test, num_test).to_pickle(f"{save_dir}/test.pkl")
    

def sampling_data(df: pd.DataFrame, num_sample: int):
    df_0 = df[df["labels"] == 0]
    df_1 = df[df["labels"] == 1]
    
    df_0_sample = df_0.sample(n=num_sample // 2 , random_state=0)
    df_1_sample = df_1.sample(n=num_sample // 2, random_state=0)
    
    df_sample = pd.concat([df_0_sample, df_1_sample])
    df_sample = df_sample.sample(frac=1, random_state=0).reset_index(drop=True)
    return df_sample
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_train", default=1000)
    parser.add_argument("--data_dir", default="nested_day_unbalance")
    args = parser.parse_args()
    
    data_dir = f"data/{args.data_dir}"
    save_dir = f"data/{args.data_dir}_{args.num_train}"
    os.makedirs(save_dir, exist_ok=True)
    under_sampling(data_dir, save_dir, int(args.num_train))

if __name__ == "__main__":
    main()
