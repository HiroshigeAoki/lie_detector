import pandas as pd
import os
import argparse
from tqdm import tqdm

def save_texts_to_file(text_list, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for text in tqdm(text_list):
            file.write(text + '\n')


def save_txt(path, mode):
    if os.path.exists(path):
        df = pd.read_pickle(path)
        text_list = [text for row in tqdm(df["nested_utters"].tolist()) for text in row["raw_nested_utters"]]
        save_texts_to_file(text_list, path.replace(".pkl", ".txt"))
    
        label_0 = df[df["labels"] == 0]
        label_0_text_list = [text for row in tqdm(label_0["nested_utters"].tolist()) for text in row["raw_nested_utters"]]
        save_texts_to_file(label_0_text_list, path.replace(".pkl", "-0.txt"))
        
        label_1 = df[df["labels"] == 1]
        label_1_text_list = [text for row in tqdm(label_1["nested_utters"].tolist()) for text in row["raw_nested_utters"]]
        save_texts_to_file(label_1_text_list, path.replace(".pkl", "-1.txt"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/nested_day")
    args = parser.parse_args()
    data_dir = args.data_dir
    
    train_path = os.path.join(data_dir, "train.pkl")
    valid_path = os.path.join(data_dir, "valid.pkl")
    test_path = os.path.join(data_dir, "test.pkl")
    
    save_txt(train_path, "train")
    save_txt(valid_path, "valid")
    save_txt(test_path, "test")


if __name__ == "__main__":
    main()
