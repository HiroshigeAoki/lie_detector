# coding=utf-8
# Copyright 2021 Masatoshi Suzuki (@singletongue)
# Copyright 2021 rinna Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import argparse
import os
import re
import unicodedata
import gzip
from urllib.request import urlretrieve


from tqdm import tqdm
import sys
sys.path.append('./src/')
from tokenizer.SPTokenizer import SentencePieceTokenizer


download_link = "https://dumps.wikimedia.org/other/cirrussearch/current/jawiki-20220328-cirrussearch-content.json.gz"
raw_data_dir = "data/jp_wiki/raw_data"
raw_data_path = f"{raw_data_dir}/wiki.json.gz"
extracted_data_path = f"{raw_data_dir}/wiki.extracted.txt"
doc_data_dir = "data/jp_wiki/doc_data"


class SPSentenceSplitter(object):
    def __init__(self):
        self.tokenizer = SentencePieceTokenizer(model_file='model/sentencepiece/werewolf.model', do_lower_case=True)

    def __call__(self, text):
        parsed_text = self.tokenizer.tokenize(text)
        return parsed_text


def download_data():
    if not os.path.exists(raw_data_path):
        print(f'Downloading {download_link} to {raw_data_path}')
        urlretrieve(download_link, raw_data_path)
        print(f'Successfully downloaded {raw_data_path}')


def preprocess_text(text, title=None):
    text = unicodedata.normalize("NFKC", text)

    # remove invisible characters
    text = "".join(c for c in text if c.isprintable())

    # remove templates
    text = re.sub(r"\[\d+?\]", "", text)
    text = re.sub(r"\[要.+?\]", "", text)
    text = re.sub(r"\{\{+[^{}]+?\}\}+", "", text)

    # remove navigation
    if title is not None:
        text = re.sub(r"^.+? \> " + re.escape(title), "", text)

    # remove footnotes
    text = re.sub(r" \^ .+", "", text)
    # remove annotations
    text = re.sub(r"\[(要出典|リンク切れ|.+?\?)\]", "", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def filter_text(text):
    # filter out text containing equations
    if "\displaystyle" in text:
        return False

    return True


def wiki_extract(num: int):
    if not os.path.exists(raw_data_dir):
        os.makedirs(raw_data_dir)

    download_data()

    sent_splitter = SPSentenceSplitter()

    with gzip.open(raw_data_path, "rt") as input_file:
        texts = []
        cnt = 0
        for line in tqdm(input_file):
            json_item = json.loads(line)
            text = json_item.get("text")
            if text is None:
                continue

            title = json_item.get("title")
            text = preprocess_text(text, title=title)

            parsed_text = sent_splitter(text)
            if len(parsed_text) == 0:
                continue
            assert len(parsed_text) != 0
            texts.append(parsed_text)
            cnt += 1
            if cnt >= num:
                return texts
        return texts
