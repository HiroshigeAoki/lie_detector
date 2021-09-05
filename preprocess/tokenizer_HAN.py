from numpy import not_equal
import pandas as pd
import torch
from torchtext.vocab import Vectors
import os, sys
sys.path.append(os.pardir)
from preprocess.custom_mecab_tagger import CustomMeCabTagger
from preprocess.custom_vocab_func import build_vocab_from_training_data


class HANtokenizer():
    def __init__(self, cache: str, **kwargs) -> None:
        self.wakati = CustomMeCabTagger("-O wakati -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")
        self.specials = ['<unk>', '<PAD>', '<BOS>', '<EOS>']
        kwargs['specials'] = self.specials
        self.vocab = build_vocab_from_training_data(**kwargs)
        self.vectors = Vectors(name='model_fasttext.vec', cache=cache)
        self.stoi = self.vocab.get_stoi()
        self.embedding_matrix = self._mk_embedding_matrix()

    def _mk_embedding_matrix(self) -> torch.tensor:
        sorted_stoi = dict(sorted(self.stoi.items(), key=lambda x: x[1]))
        special_tokens_matrix = torch.zeros(len(self.specials), 200)
        other_tokens_matrix = self.vectors.get_vecs_by_tokens(list(sorted_stoi.keys())[len(self.specials):])
        embedding_matrix = torch.cat((special_tokens_matrix ,other_tokens_matrix), dim=0)
        return embedding_matrix

    def tokenize(self, utter: str) -> list[str]:
        parsed_utter = self.wakati(utter).split()
        return [word if word in self.stoi else '<unk>' for word in parsed_utter]

    def numericalize(self, utter: list[str]) -> list[int]:
        return self.vocab.lookup_indices(list(utter))

    def padding_word_level(self, utter: list[int]) -> list[int]:
        if len(utter) > self.max_word_len:
            return utter[:self.max_word_len]
        else:
            padded = utter + [1 for _ in range(self.max_word_len - len(utter))]
            return padded

    def padding_sent_level(self, nested_utters: list[torch.tensor]) -> list[torch.tensor]:
        if len(nested_utters) > self.max_sent_len:
            return nested_utters[:self.max_sent_len]
        else:
            padding = [[1 for _ in range(self.max_word_len)] for _ in range(self.max_sent_len - len(nested_utters))]
            padded = nested_utters + padding
            return padded

    def encode(self, nested_utters_df: pd.DataFrame, max_word_len: int = 180, max_sent_len: int = 180) -> list[torch.tensor]:
        self.max_word_len = max_word_len
        self.max_sent_len = max_sent_len
        nested_utters_df_tokenied = nested_utters_df.apply(self.tokenize)
        nested_utters_df_numericalized = nested_utters_df_tokenied.apply(self.numericalize)
        nested_utters_df_padded_word = nested_utters_df_numericalized.apply(self.padding_word_level)
        nested_utter_padded_word = nested_utters_df_padded_word.to_list()
        padded_nested_utters = self.padding_sent_level(nested_utter_padded_word)
        return torch.tensor(padded_nested_utters)