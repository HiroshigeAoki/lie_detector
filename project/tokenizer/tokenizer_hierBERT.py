from typing import Tuple
from pandas.core.frame import DataFrame
import torch

from transformers import BertJapaneseTokenizer
from transformers.utils.dummy_pt_objects import MaxLengthCriteria

class hierBertTokenizer():
    def __init__(
        self,
        max_len_sent: int,
        max_len_doc: int,
        pretrained_model: str = 'cl-tohoku/bert-large-japanese',
        ):
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(pretrained_model, additional_special_tokens=['<person>'])
        self.max_len_sent = max_len_sent
        self.max_len_doc = max_len_doc

    def padding_word_level(self, input_ids: list[int], attention_mask: list[int]) -> Tuple[list[int], list[int]]:
        if len(input_ids) > self.max_len_sent:
            return input_ids[:self.max_len_sent], attention_mask[:self.max_len_sent]
        else:
            padding = [0 for _ in range(self.max_len_sent - len(input_ids))]
            input_ids = input_ids + padding
            attention_mask = attention_mask + padding
            return input_ids, attention_mask

    def padding_sent_level(self, input_ids: list[list[int]], attention_mask: list[list[int]]) -> Tuple[list[list[int]], list[list[int]]]:
        if len(input_ids) > self.max_len_doc:
            return input_ids[:self.max_len_doc]
        else:
            padding = [[0 for _ in range(self.max_len_sent)] for _ in range(self.max_len_doc - len(input_ids))]
            input_ids = input_ids + padding
            attention_mask = attention_mask + padding
            return input_ids, attention_mask

    def encode(self, doc: DataFrame) -> Tuple[torch.LongTensor, torch.LongTensor]:
        input_ids, attention_mask = [], []
        for sent in doc.values.tolist():
            _input_ids, _, _attention_mask = self.tokenizer.encode_plus(''.join(sent)).values()
            _input_ids, _attention_mask = self.padding_word_level(_input_ids, _attention_mask)
            input_ids.append(_input_ids)
            attention_mask.append(_attention_mask)
        input_ids, attention_mask = self.padding_sent_level(input_ids, attention_mask)
        return torch.LongTensor(input_ids), torch.LongTensor(attention_mask)