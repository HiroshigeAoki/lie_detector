import argparse
import pandas as pd
import joblib
from collections import Counter

import os, sys
sys.path.append("./src")
from preprocess.ngram_utils import (
    tokenize_group,
    generate_ngrams_counter_per_user,
    calc_diff_ngrams,
    calculate_vocab_count,
)


def concat_group(group):
    return pd.concat([row["raw_nested_utters"] for row in group[1]["nested_utters"].tolist()]).tolist()


def process_group(name, rows, tokenizer_name, output_dir):
    groups = list(rows.groupby('users'))
    groups_utters_list = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(concat_group)(group) for group in groups
    )
    user_names = [group[0] for group in groups]
    tokens_list = tokenize_group(groups_utters_list, tokenizer_name=tokenizer_name)
    counters = {}
    for ngram in range(1, 6):
        counter = generate_ngrams_counter_per_user(tokens_list, user_names, ngram)
        # ユーザーごとに各トークン1回ずつカウント
        vocab_count = calculate_vocab_count(counter)
        counters[f"{name}_{ngram}"] = vocab_count
        pd.DataFrame(vocab_count.most_common(), columns=['ngram', 'count']).to_csv(
            os.path.join(output_dir, f"{name}_{ngram}.csv"), index=False)
    return dict(counters=counters, user_num=len(user_names))


def get_model_data_dir_test(data: str, model: str):
    if data == "nested_day":
        test = pd.read_pickle("data/nested_day/test.pkl")
        if model == "bigbird":
            data_dir = "outputs/nested_day/hf_bigbird/baseline/2024-01-09_082240"
        elif model == "hierbert":
            data_dir = "outputs/nested_day/hf_hierbert/baseline/2024-01-07_040732"
        else:
            raise ValueError("model must be bigbird or hierbert")
    elif data == "nested_day_twitter":
        test = pd.read_pickle("data/nested_day_twitter/test.pkl")
        if model == "bigbird":
            data_dir = f"outputs/nested_day_twitter/hf_bigbird/baseline/2024-01-26_053837"
        elif model == "hierbert":
            data_dir = "outputs/nested_day_twitter/hf_hierbert/baseline/2024-01-25_051414"
        else:
            raise ValueError("model must be bigbird or hierbert")        
    elif data == "murder_mystery":
        test = pd.read_pickle("data/murder_mystery/test.pkl")
        if model == "bigbird":
            data_dir = "outputs/murder_mystery/hf_bigbird/baseline/2024-02-06_120218"
        elif model == "hierbert":
            data_dir = "outputs/murder_mystery/hf_hierbert/baseline/2024-01-29_091844"
        else:
            raise ValueError("model must be bigbird or hierbert")
    else:
        raise ValueError(f"data {data} not valid")
    return data_dir, test


def diff_each_cm(model, data):
    data_dir, test = get_model_data_dir_test(data, model)
    confidences = pd.read_csv(os.path.join(data_dir, "confidence.csv"))
    
    tp_index = confidences[(confidences['true label'] == 1) & (confidences['pred label'] == 1)].index
    tn_index = confidences[(confidences['true label'] == 0) & (confidences['pred label'] == 0)].index
    fp_index = confidences[(confidences['true label'] == 0) & (confidences['pred label'] == 1)].index
    fn_index = confidences[(confidences['true label'] == 1) & (confidences['pred label'] == 0)].index

    tp_rows = test.iloc[tp_index]
    tn_rows = test.iloc[tn_index]
    fp_rows = test.iloc[fp_index]
    fn_rows = test.iloc[fn_index]
    
    cm_name = ["TP", "TN", "FP", "FN"]
    output_dir = os.path.join(data_dir, "raw")
    os.makedirs(output_dir, exist_ok=True)
    results = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(process_group)(name, rows, tokenizer_name=model, output_dir=output_dir)
        for name, rows in zip(cm_name, [tp_rows, tn_rows, fp_rows, fn_rows])
    )
    diff_output_dir = os.path.join(data_dir, "diff")
    os.makedirs(diff_output_dir, exist_ok=True)
    normed_diff_output_dir = os.path.join(data_dir, "normed_diff")
    os.makedirs(normed_diff_output_dir, exist_ok=True)
    for i, name in enumerate(cm_name):
        other_cm = [(i, n) for i, n in enumerate(cm_name) if n != name]
        for ngram in range(1, 6):
            for j, another_cm in other_cm:
                target = Counter(results[i]["counters"][f"{name}_{ngram}"])
                another = Counter(results[j]["counters"][f"{another_cm}_{ngram}"])
                pd.DataFrame(
                    calc_diff_ngrams(target, another), columns=['ngram', 'difference']).to_csv(
                            os.path.join(diff_output_dir, f"{name}-{another_cm}_{ngram}_gram.csv"), index=False
                    )
                normed_target = Counter(dict(map(lambda kv: (kv[0], kv[1] / results[i]["user_num"]), target.items())))
                normed_another = Counter(dict(map(lambda kv: (kv[0], kv[1] / results[j]["user_num"]), another.items())))
                pd.DataFrame(
                    calc_diff_ngrams(normed_target, normed_another), columns=['ngram', 'difference']).to_csv(
                            os.path.join(normed_diff_output_dir, f"{name}-{another_cm}_{ngram}_gram.csv"), index=False
                )


def check_if_diff_cm_exists(model, data):
    data_dir, _ = get_model_data_dir_test(data, model)
    cm_name = ["TP", "TN", "FP", "FN"]
    for name in cm_name:
        other_cm = [(i, n) for i, n in enumerate(cm_name) if n != name]
        for ngram in range(1, 6):
            if not os.path.exists(os.path.join(data_dir, "raw", f"{name}_{ngram}.csv")):
                return False
            for another_cm in other_cm:
                if not os.path.exists(os.path.join(data_dir, "diff", f"{name}-{another_cm}_{ngram}_gram.csv")):
                    return False
                elif not os.path.exists(os.path.join(data_dir, "normed_diff", f"{name}-{another_cm}_{ngram}_gram.csv")):
                    return False
    return True


def diff_nested_day():
    cm_name = ["TP", "TN", "FP", "FN"]
    for model in ["bigbird", "hierbert"]:
        output_dir_raw, _ = get_model_data_dir_test("nested_day", model)
        output_dir_filtered, _ = get_model_data_dir_test("nested_day_twitter", model)
        os.makedirs(os.path.join(output_dir_filtered, "raw-filtered"), exist_ok=True)
        os.makedirs(os.path.join(output_dir_filtered, "filtered-raw"), exist_ok=True)
        for name in cm_name:
            for ngram in range(1, 6):
                raw = pd.read_csv(os.path.join(output_dir_raw, "raw", f"{name}_{ngram}.csv"))
                raw_counter = Counter(dict(zip(raw["ngram"], raw["count"])))
                filtered = pd.read_csv(os.path.join(output_dir_filtered, "raw", f"{name}_{ngram}.csv"))
                filtered_counter = Counter(dict(zip(filtered["ngram"], filtered["count"])))
                
                diff = calc_diff_ngrams(raw_counter, filtered_counter)
                pd.DataFrame(diff, columns=['ngram', 'difference']).to_csv(
                    os.path.join(output_dir_filtered, "raw-filtered", f"{name}_{ngram}.csv"), index=False
                )
                diff = calc_diff_ngrams(filtered_counter, raw_counter)
                pd.DataFrame(diff, columns=['ngram', 'difference']).to_csv(
                    os.path.join(output_dir_filtered, "filtered-raw", f"{name}_{ngram}.csv"), index=False
                )


if __name__ == "__main__":
    for data in ["nested_day", "nested_day_twitter", "murder_mystery"]:
        for model in ["bigbird", "hierbert"]:
            if not check_if_diff_cm_exists(model, data):
                diff_each_cm(model, data)
    
    diff_nested_day()
