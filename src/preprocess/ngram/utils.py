import MeCab
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import joblib
from collections import Counter
from pandarallel import pandarallel
from typing import List, Set
from transformers import AutoTokenizer

# Pandarallelの初期化
pandarallel.initialize(progress_bar=True)
bigbird_tokenizer = AutoTokenizer.from_pretrained('nlp-waseda/bigbird-base-japanese', additional_special_tokens=['<person>'])
bert_tokenizer = AutoTokenizer.from_pretrained('tohoku-nlp/bert-base-japanese-v3', additional_special_tokens=['<person>'])


def tokenize(
    text, 
    exclude_tokens: List[str]=[], 
    full_width_person_token: bool=True, 
    token_set: Set[str]=None, 
    excluded_tokens: List[str]=None,
    tokenizer_name="mecab"):
    # text = text.replace('0', '').replace(',', '').replace('.', '').replace('。', '').replace('、', '')#.replace('<person>', '')
    if tokenizer_name == "mecab":
        text = text.replace("<person>", "＠<person>＠")
        for num in ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩", "⑪", "⑫", "⑬", "⑭", "⑮", "⑯", "⑰", "⑱", "⑲", "⑳", "_", "※", ":", "\\", "|", "\\", "＼", "ぉっふぉっふぉ", "ーーーー"]:
            text = text.replace(num, "")
        mecab = MeCab.Tagger("-Owakati -d /var/lib/mecab/dic/debian -u /home/haoki/dev/lie_detector/src/tokenizer/mecab_userdic/special.dic")
        parsed_text = mecab.parse(text)
        parsed_text = parsed_text.replace("＠ <person> ＠", "<person>")
        if parsed_text is None:
            return []
        if full_width_person_token:
            parsed_text = parsed_text.replace("<person>", "＜person＞").replace('<', '＜').replace('>', '＞')
        parsed = parsed_text.strip().split()
        return_list = []
        for token in parsed:
            if token in exclude_tokens:
                if excluded_tokens is not None:
                    excluded_tokens.append(token)
                continue
            if token_set is not None and token not in token_set:
                if excluded_tokens is not None:
                    excluded_tokens.append(token)
                continue
            return_list.append(token)
        return return_list
    
    elif tokenizer_name == "bigbird" or tokenizer_name == "hierbert":
        tokenizer: AutoTokenizer
        if tokenizer_name == "bigbird":
            tokenizer = bigbird_tokenizer
        elif tokenizer_name == "hierbert":
            tokenizer = bert_tokenizer
        tokens = tokenizer.tokenize(text)
        return_list = []
        for token in tokens:
            if token in exclude_tokens:
                if excluded_tokens is not None:
                    excluded_tokens.append(token)
                continue
            if token_set is not None and token not in token_set:
                if excluded_tokens is not None:
                    excluded_tokens.append(token)
                continue
            token = token[2:] if token.startswith("##") else token
            token = token[1:] if token.startswith("_") else token  
            return_list.append(token)
        return return_list


def tokenize_list_doc(doc: list, exclude_tokens=[], full_width_person_token=True, tokenizer_name="mecab"):
    parsed = [tokenize(utter, exclude_tokens, full_width_person_token, tokenizer_name=tokenizer_name) for utter in doc]
    return parsed


def tokenize_group(doc_list_per_user, exclude_tokens=[], full_width_person_token=True, tokenizer_name="mecab"):
    tokens_list_per_user = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(tokenize_list_doc)(doc, exclude_tokens, full_width_person_token, tokenizer_name) for doc in tqdm(doc_list_per_user, desc="tokenize group")
    )
    return tokens_list_per_user
    

def flatten_docs(docs):
    _flatten_docs = []
    for doc in docs["nested_utters"]:
        _flatten_docs.extend(doc["raw_nested_utters"].tolist())
    return _flatten_docs


def parallel_tokenize(doc: list):
    tokens = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(tokenize)(utter) for utter in tqdm(doc)
    )
    return tokens


def generate_ngrams_for_sentence(words, n_gram: int):
    ngrams = list(map(lambda x: " ".join(x), zip(*[words[i:] for i in range(n_gram)])))
    return ngrams


def generate_ngram_counts(tokens_list, n_gram):
    ngram_counts = Counter()
    for tokens in tqdm(tokens_list, desc=f"generate {n_gram}-gram"):
        ngrams = generate_ngrams_for_sentence(tokens, n_gram)
        ngram_counts.update(ngrams)
    return ngram_counts


def _generate_ngrams_set_per_user(docs):
    # ユーザー毎にn-gramのsetを取り、特定のユーザーに偏っていることを避ける。
    ngrams_dict = {"1": set(), "2": set(), "3": set(), "4": set(), "5": set()}
    for utter in docs:
        tokens = tokenize(utter)
        for n_gram in [1, 2, 3, 4, 5]:
            ngrams = generate_ngrams_for_sentence(tokens, n_gram)
            ngrams_dict[f"{n_gram}"].update(ngrams)
    return ngrams_dict


def count_ngrams_set_per_user(docs_per_user):
    ngrams_dicts_per_user = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(_generate_ngrams_set_per_user)(docs) for docs in tqdm(docs_per_user, desc="count ngrams_per_user")
    )
    ngrams_counter = {"1": Counter(), "2": Counter(), "3": Counter(), "4": Counter(), "5": Counter()}
    for ngrams_dict_per_user in tqdm(ngrams_dicts_per_user, desc="count ngrams"):
        for n_gram in [1, 2, 3, 4, 5]:
            ngrams_counter[f"{n_gram}"].update(ngrams_dict_per_user[f"{n_gram}"])
    return ngrams_counter


def generate_ngrams_counter(tokens_list, n_gram: int):
    counter = Counter()
    for tokens in tokens_list:
        ngrams = generate_ngrams_for_sentence(tokens, n_gram)
        counter.update(ngrams)
    return counter


def _generate_ngrams_counter_per_user(tokens_list, user_name, n_gram: int):
    ngram_counter = {f"{user_name}": Counter()}
    for tokens in tokens_list:
        ngrams = generate_ngrams_for_sentence(tokens, n_gram)
        ngram_counter[f"{user_name}"].update(ngrams)
    return ngram_counter


def generate_ngrams_counter_per_user(tokens_per_user, user_name_list, n_gram: int) -> list:
    ngrams_dicts_per_user = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(_generate_ngrams_counter_per_user)(tokens_list, user_name, n_gram) for tokens_list, user_name in tqdm(zip(tokens_per_user, user_name_list), desc=f"count {n_gram}_grams_per_user")
    )
    return ngrams_dicts_per_user


def _normalize_count(ngram, count):# , total):
    return ngram, count# / total


def normalize_ngram_counts(ngram_counts):
    #total = sum(ngram_counts.values()) # 全ユーザーの語彙数の合計
    results = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(_normalize_count)(ngram, count) for ngram, count in tqdm(ngram_counts.items(), desc="normalize ngram counts")
    )
    normalized_counts = dict(results)
    return normalized_counts


def compute_idf(doc_ngrams, total_docs):
    # 各N-gramが出現する文書数をカウント
    doc_freq = Counter()
    for ngrams in doc_ngrams:
        unique_ngrams = set(ngrams)
        doc_freq.update(unique_ngrams)

    # IDF計算
    idf = {ngram: np.log(total_docs / (1 + freq)) for ngram, freq in doc_freq.items()}
    return idf


def compute_tf_idf(ngram_counts, idf):
    tf_idf = {ngram: count * idf.get(ngram, 0) for ngram, count in ngram_counts.items()}
    return tf_idf


def get_ngrams_counts_with_per_doc_count(tokens_list, n_gram):
    # idfの計算に必要なため、文書毎のn-gramも返す
    ngram_counts = Counter()
    doc_ngrams = []

    for tokens in tokens_list:
        ngrams = generate_ngrams_for_sentence(tokens, n_gram)
        ngram_counts.update(ngrams)
        doc_ngrams.append(ngrams)

    return ngram_counts, doc_ngrams


def calc_diff_ngrams(target_label_ngram_counts, another_label_ngram_counts):
    all_ngrams = set(target_label_ngram_counts).union(set(another_label_ngram_counts))
    diff_ngrams = {ngram: target_label_ngram_counts.get(ngram, 0) - another_label_ngram_counts.get(ngram, 0) for ngram in all_ngrams}
    sorted_ngrams = sorted(diff_ngrams.items(), key=lambda x: x[1], reverse=True)
    return sorted_ngrams


def save_ngram_tfidf_diff(output_dir, label_0_tokens, label_1_tokens, n_gram, label_0_name="werewolf", label_1_name="twitter"):
    label_0_ngram_counts, label_0_doc_ngrams = get_ngrams_counts_with_per_doc_count(label_0_tokens, n_gram)
    label_1_ngram_counts, label_1_doc_ngrams = get_ngrams_counts_with_per_doc_count(label_1_tokens, n_gram)

    # カウントベースの正規化
    normalized_label_0_ngram_counts_count = normalize_ngram_counts(label_0_ngram_counts)
    normalized_label_1_ngram_counts_count = normalize_ngram_counts(label_1_ngram_counts)

    # カウントベースの結果を保存
    sorted_ngrams_0_count = calc_diff_ngrams(normalized_label_0_ngram_counts_count, normalized_label_1_ngram_counts_count)
    pd.DataFrame(sorted_ngrams_0_count, columns=['ngram', 'difference']).to_csv(os.path.join(output_dir, f"diff_{n_gram}_gram_{label_0_name}_count.csv"), index=False)
    sorted_ngrams_1_count = calc_diff_ngrams(normalized_label_1_ngram_counts_count, normalized_label_0_ngram_counts_count)
    pd.DataFrame(sorted_ngrams_1_count, columns=['ngram', 'difference']).to_csv(os.path.join(output_dir, f"diff_{n_gram}_gram_{label_1_name}_count.csv"), index=False)

    # TF-IDFベースの正規化
    idf = compute_idf(label_0_doc_ngrams + label_1_doc_ngrams, len(label_0_doc_ngrams) + len(label_1_doc_ngrams))
    normalized_label_0_ngram_counts_tfidf = compute_tf_idf(label_0_ngram_counts, idf)
    normalized_label_1_ngram_counts_tfidf = compute_tf_idf(label_1_ngram_counts, idf)

    # TF-IDFベースの結果を保存
    sorted_ngrams_0_tfidf = calc_diff_ngrams(normalized_label_0_ngram_counts_tfidf, normalized_label_1_ngram_counts_tfidf)
    pd.DataFrame(sorted_ngrams_0_tfidf, columns=['ngram', 'difference']).to_csv(os.path.join(output_dir, f"diff_{n_gram}_gram_{label_0_name}_tfidf.csv"), index=False)
    sorted_ngrams_1_tfidf = calc_diff_ngrams(normalized_label_1_ngram_counts_tfidf, normalized_label_0_ngram_counts_tfidf)
    pd.DataFrame(sorted_ngrams_1_tfidf, columns=['ngram', 'difference']).to_csv(os.path.join(output_dir, f"diff_{n_gram}_gram_{label_1_name}_tfidf.csv"), index=False)


def _calculate_vocab_count(ngrams_counter):
    return Counter(next(iter(ngrams_counter.values())).keys())


def calculate_vocab_count(ngrams_counters):
    # 各ユーザに付き、一度のみカウント。
    combined_counter = Counter()
    for ngrams_counter in tqdm(ngrams_counters, desc="Calculating vocab count"):
        partial_counter = _calculate_vocab_count(ngrams_counter)
        combined_counter.update(partial_counter)
    return combined_counter