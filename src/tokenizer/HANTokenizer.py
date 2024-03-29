from typing import List

import pandas as pd
import torch
from torchtext.vocab import Vectors
import os, sys
from transformers import BertJapaneseTokenizer
# from src.preprocess.custom_mecab_tagger import CustomMeCabTagger
from src.preprocess.custom_vocab_func import build_vocab_from_training_data
from src.tokenizer.SPTokenizer import SentencePieceTokenizer

class HANTokenizer():
    def __init__(
            self,
            cache_dir: str,
            embed_dim: int,
            sent_length: int,
            doc_length: int,
            tokenizer_type: str,
            data_dir,
            pad_index: int,
            model_file: str = None,
            do_lower_case: bool = True,
            **kwargs
        ):
        self.sent_length = sent_length
        self.doc_length = doc_length
        self.specials = ['<unk>', '<PAD>', '<BOS>', '<EOS>']
        self.pad_index = pad_index
        kwargs['specials'] = self.specials
        self.embed_dim = embed_dim
        self.vocab = build_vocab_from_training_data(data_dir, tokenizer_type, **kwargs)
        self.vectors = Vectors(name='model_fasttext.vec', cache=os.path.join(cache_dir, f"{tokenizer_type}_vectors/dim_{self.embed_dim}"))
        self.stoi = self.vocab.get_stoi()
        self.embedding_matrix = self._mk_embedding_matrix()
        self.tokenizer_type = tokenizer_type

        if self.tokenizer_type == 'mecab-wordpiece':
            self.tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-large-japanese', additional_special_tokens=['<person>'])
#        elif self.tokenizer_type == 'mecab':
#            self.tokenizer = CustomMeCabTagger("-O wakati -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd -r /home/haoki/Documents/vscode-workplaces/lie_detector/src/tokenizer/mecab_userdic/mecabrc")
        elif self.tokenizer_type == 'sp':
            self.tokenizer = SentencePieceTokenizer(model_file=model_file, do_lower_case=do_lower_case)
        else:
            raise ValueError(f'tokenizer_type: {self.tokenizer_type} is invalid.')

    def _mk_embedding_matrix(self) -> torch.tensor:
        sorted_stoi = dict(sorted(self.stoi.items(), key=lambda x: x[1]))
        special_tokens_matrix = torch.zeros(len(self.specials), self.embed_dim)
        other_tokens_matrix = self.vectors.get_vecs_by_tokens(list(sorted_stoi.keys())[len(self.specials):])
        embedding_matrix = torch.cat((special_tokens_matrix ,other_tokens_matrix), dim=0)
        return embedding_matrix

    def tokenize(self, utter: str) -> list[str]:
        parsed_utter = self.tokenizer.tokenize(utter)
        return [word if word in self.stoi else '<unk>' for word in parsed_utter]

    def numericalize(self, utter: list[str]) -> list[int]:
        return self.vocab.lookup_indices(list(utter))

    def padding_word_level(self, utter: list[int]) -> pd.Series:
        if len(utter) > self.sent_length:
            attention_mask = torch.ones(self.sent_length)
            return pd.Series([utter[:self.sent_length], attention_mask.tolist()], index=['padded', 'attention_mask'])
        else:
            padded = utter + [self.pad_index for _ in range(self.sent_length - len(utter))]
            attention_mask = torch.ones(self.sent_length)
            attention_mask[len(utter):] = 0
            return pd.Series([padded, attention_mask.tolist()], index=['padded', 'attention_mask'])

    def padding_sent_level(self, nested_utters: list[torch.tensor], is_attention_mask=False) -> list[torch.tensor]:
        if len(nested_utters) > self.doc_length:
            return nested_utters[:self.doc_length], 0
        else:
            pad_sent_num = self.doc_length - len(nested_utters)
            if not is_attention_mask:
                padding = [[self.pad_index for _ in range(self.sent_length)] for _ in range(pad_sent_num)]
            else:
                padding = [[0 for _ in range(self.sent_length)] for _ in range(pad_sent_num)]
            padded = nested_utters + padding
            return padded, pad_sent_num

    def encode(self, nested_utters_df: pd.DataFrame):
        nested_utters_df_tokenied = nested_utters_df.apply(self.tokenize)
        nested_utters_df_numericalized = nested_utters_df_tokenied.apply(self.numericalize)
        nested_utters_df_padded_word = nested_utters_df_numericalized.apply(self.padding_word_level)
        if 'padded' not in nested_utters_df_padded_word.columns:
            print(nested_utters_df_padded_word)
        else:
            nested_utter_padded_word = nested_utters_df_padded_word['padded'].to_list()
        padded_nested_utters, pad_sent_num = self.padding_sent_level(nested_utter_padded_word)
        attention_mask, _ = self.padding_sent_level(nested_utters_df_padded_word['attention_mask'].to_list(), is_attention_mask=True)
        return torch.tensor(padded_nested_utters), torch.tensor(attention_mask), pad_sent_num

    def batch_decode(self, indices: List[int]) -> List[str]:
        return self.vocab.lookup_tokens(indices)
