from typing import Tuple
from pandas.core.frame import DataFrame
import torch
from transformers import AutoTokenizer

class HFTokenizer():
    def __init__(
        self,
        doc_length: int,
        pretrained_model: str = '',
        additional_special_tokens = None,
        pad_index: int = 0,
        ):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, additional_special_tokens=additional_special_tokens)
        self.doc_length = doc_length
        self.pad_index = pad_index
        
    def padding_sent_level(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if len(input_ids) >= self.doc_length:
            return input_ids[:self.doc_length], attention_mask[:self.doc_length], 0
        else:
            pad_sent_num = self.doc_length - len(input_ids)
            padding = torch.zeros((pad_sent_num, len(input_ids[0])), dtype=torch.long)
            input_ids = torch.cat([input_ids, padding], dim=0)
            attention_mask = torch.cat([attention_mask, padding], dim=0)
            return input_ids, attention_mask, pad_sent_num
    
    #TODO: batch_encode_plusでpaddingを行い、その後にsent levelでpaddingを行うようにする。

    def batch_encode_plus(self, doc: list, **kwargs) -> Tuple[torch.LongTensor, torch.FloatTensor, int]:
        encodes = self.tokenizer.batch_encode_plus(doc, **kwargs)
        input_ids, attention_mask = encodes['input_ids'], encodes['attention_mask']
        input_ids, attention_mask, pad_sent_num = self.padding_sent_level(input_ids, attention_mask)
        return input_ids, attention_mask, pad_sent_num