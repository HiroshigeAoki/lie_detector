from typing import Tuple
import torch

class HFTokenizer():
    def __init__(
        self,
        tokenizer,
        doc_length: int = 200,
        max_length: int = 768,
        ):
        self.tokenizer = tokenizer
        self.doc_length = doc_length
        self.pad_index = self.tokenizer.pad_token_id
        self.sep_token = self.tokenizer.sep_token
        self.max_length = max_length
        
        if self.pad_index is None:
            raise ValueError("pad token for padding is None")
        if self.sep_token is None:
            raise ValueError("sep token for padding is None")
        
    # nested
    def padding_sent_level(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if len(input_ids) >= self.doc_length:
            return input_ids[:self.doc_length], attention_mask[:self.doc_length], 0
        else:
            pad_sent_num = self.doc_length - len(input_ids)
            padding = torch.zeros((pad_sent_num, len(input_ids[0])), dtype=torch.long)
            input_ids = torch.cat([input_ids, padding], dim=0)
            attention_mask = torch.cat([attention_mask, padding], dim=0)
            return input_ids, attention_mask, pad_sent_num
    
    def batch_encode_nested(self, doc: list, **kwargs) -> Tuple[torch.LongTensor, torch.FloatTensor, int]:
        encodes = self.tokenizer.batch_encode_plus(doc, **kwargs)
        input_ids, attention_mask = encodes['input_ids'], encodes['attention_mask']
        input_ids, attention_mask, pad_sent_num = self.padding_sent_level(input_ids, attention_mask)
        return input_ids, attention_mask, pad_sent_num
    
    # flat
    def concatenate_utters(self, doc: list) -> str:
        return "".join([utter + self.sep_token for utter in doc])

    def encode_flat(self, doc: list):
        flatten_doc = self.concatenate_utters(doc)
        encodes = self.tokenizer(flatten_doc, padding='max_length', truncation=True, return_tensors='pt', max_length=self.max_length)
        squeeze_encodes = {k: v.squeeze(0) for k, v in encodes.items()}
        return squeeze_encodes