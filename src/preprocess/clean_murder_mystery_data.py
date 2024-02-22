import pandas as pd
import json
import sys

sys.path.append("./src")
from preprocess.cleaner import clean_murder_mystery_data

replacement_track_dict = {}

def clean_utters(utters):
    cleaned = []
    for utter in utters["raw_nested_utters"].tolist():
        cleaned.append(clean_murder_mystery_data(utter, replacement_track_dict))
    return pd.DataFrame({"raw_nested_utters": cleaned})

def main():
    global replacement_track_dict
    test = pd.read_pickle("data/murder_mystery/test.pkl")
    test["nested_utters"] = test["nested_utters"].apply(clean_utters)
    test.to_pickle("data/murder_mystery/test.pkl")
    for key in replacement_track_dict:
        replacement_track_dict[key]['examples'] = list(replacement_track_dict[key]['examples'])
    sorted_replaced = dict(sorted(replacement_track_dict.items(), key=lambda item: item[1]['count'], reverse=True))
    with open("data/murder_mystery/replaced.json", "w", encoding="utf-8") as f:
        json.dump(sorted_replaced, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
