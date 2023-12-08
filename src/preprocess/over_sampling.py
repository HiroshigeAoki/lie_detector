import argparse
import pandas as pd
import os
from tqdm import tqdm


def over_sampling(data_dir, save_dir):
    train = pd.read_pickle(f"{data_dir}/train.pkl")
    valid = pd.read_pickle(f"{data_dir}/valid.pkl")
    test = pd.read_pickle(f"{data_dir}/test.pkl")
    
    over_sample_werewolves(train).to_pickle(f"{save_dir}/train.pkl")
    over_sample_werewolves(valid).to_pickle(f"{save_dir}/valid.pkl")
    over_sample_werewolves(test).to_pickle(f"{save_dir}/test.pkl")
    

def over_sample_werewolves(df):
    nested_utterances = df["nested_utters"].tolist()
    num_utters = df["num_utters"].tolist()
    labels = df["labels"].tolist()
    users = df["users"].tolist()
    werewolf_indices = [i for i, label in enumerate(labels) if label == 1]
    civil_num = len(labels) - len(werewolf_indices)
    difference = civil_num - len(werewolf_indices)

    for i in tqdm(range(difference - len(werewolf_indices))):
        werewolf_indices.append(werewolf_indices[i])

    for i in tqdm(werewolf_indices):
        nested_utterances.append(nested_utterances[i])
        num_utters.append(num_utters[i])
        labels.append(1)
        users.append(users[i])

    return pd.DataFrame(
        dict(
            nested_utters=nested_utterances,
            num_utters=num_utters,
            labels=labels,
            users=users
        )
    )
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="nested_day_unbalance")
    args = parser.parse_args()
    
    data_dir = f"data/{args.data_dir}"
    if not data_dir.endswith("_unbalance"):
        raise ValueError(f"data_dir:{data_dir} is invalid.")
    
    save_dir = data_dir.replace("_unbalance", "")
    
    os.makedirs(save_dir, exist_ok=True)
    over_sampling(data_dir, save_dir)

if __name__ == "__main__":
    main()
