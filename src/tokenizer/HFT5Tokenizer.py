from typing import Tuple
from pandas.core.frame import DataFrame
import torch
from transformers import AutoTokenizer


class HFT5Tokenizer():
    def __init__(
        self,
        sent_length: int,
        doc_length: int,
        pretrained_model: str = '',
        additional_special_tokens: list(str) = None,
        pad_index: int = 1,
        ):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, additional_special_tokens=additional_special_tokens)
        self.sent_length = sent_length
        self.doc_length = doc_length
        self.pad_index = pad_index

    def padding_word_level(self, input_ids: list[int], attention_mask: list[int]) -> Tuple[list[int], list[int]]:
        if len(input_ids) >= self.sent_length:
            return input_ids[:self.sent_length], attention_mask[:self.sent_length]
        else:
            padding = [self.pad_index for _ in range(self.sent_length - len(input_ids))]
            padding_attentionmask = [0 for _ in range(self.sent_length - len(input_ids))]
            input_ids = input_ids + padding
            attention_mask = attention_mask + padding_attentionmask
            return input_ids, attention_mask

    def padding_sent_level(self, input_ids: list[list[int]], attention_mask: list[list[int]]) -> Tuple[list[list[int]], list[list[int]], int]:
        if len(input_ids) >= self.doc_length:
            return input_ids[:self.doc_length], attention_mask[:self.doc_length], 0
        else:
            pad_sent_num = self.doc_length - len(input_ids)
            padding = [[self.pad_index for _ in range(self.sent_length)] for _ in range(pad_sent_num)]
            input_ids = input_ids + padding
            padding = [[0 for _ in range(self.sent_length)] for _ in range(pad_sent_num)]
            attention_mask = attention_mask + padding
            return input_ids, attention_mask, pad_sent_num

    def encode(self, doc: DataFrame) -> Tuple[torch.LongTensor, torch.FloatTensor, int]:
        input_ids, attention_mask = [], []
        for sent in doc.values.tolist():
            _input_ids, _attention_mask = self.tokenizer.encode_plus(''.join(sent)).values()
            _input_ids, _attention_mask = self.padding_word_level(_input_ids, _attention_mask)
            input_ids.append(_input_ids)
            attention_mask.append(_attention_mask)
        input_ids, attention_mask, pad_sent_num = self.padding_sent_level(input_ids, attention_mask)
        return torch.LongTensor(input_ids), torch.FloatTensor(attention_mask), pad_sent_num