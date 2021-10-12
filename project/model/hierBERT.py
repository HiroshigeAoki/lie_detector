from typing import Tuple
import torch
from torch import nn
import pytorch_lightning as pl
from transformers import BertModel, BertJapaneseTokenizer
from transformers.utils.dummy_pt_objects import PreTrainedModel


class HierBERT(pl.LightningModule):
    def __init__(self,
        max_sent_len: int,
        max_doc_len: int,
        hidden_dim: int,
        batch_size: int,
        pretrained_model: str,
        ):
        super(HierBERT, self).__init__()
        self.save_hyperparameters()

        self.max_sent_len = max_sent_len
        self.max_doc_len = max_doc_len
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        self.wordAttennet = WordAttention(
            pretrained_model=pretrained_model,
        )

        self.sentAttention = SentAttention(
            hidden_dim=self.hidden_dim
        )

    def forward(self, input_ids, attention_mask):
        input_ids = input_ids.permute(1,0,2)
        attention_mask = attention_mask.permute(1,0,2)

        assert input_ids.shape == (self.max_sent_len, self.batch_size, self.max_doc_len), gen_assertion_message(target_name='input_ids', target=input_ids, expected=(self.max_sent_len, self.batch_size, self.max_doc_len))
        assert attention_mask.shape == (self.max_sent_len, self.batch_size, self.max_doc_len), gen_assertion_message(target_name='attention_mask', target=attention_mask, expected=(self.max_sent_len, self.batch_size, self.max_doc_len))

        last_hidden_state_word_level, pooler_output_word_level = [], []
        for _input_ids, _attention_mask in zip(input_ids, attention_mask):
            _last_hidden_state, _pooler_output = self.wordAttennet(input_ids=_input_ids, attention_mask=_attention_mask).values()
            last_hidden_state_word_level.append(_last_hidden_state)
            pooler_output_word_level.append(_pooler_output)

        # 今の所こんな感じ
        # TODO: poolerを使うか、last_hiddenstateの平均を使うか
        # TODO: sentenceAttenにBERTを使うか、GRUを使うか
        # TODO: hidden_dimやbatchなどを引数で、selfに登録して、assertでshapeを確認する。
        input_ids = torch.stack(pooler_output_word_level).permute(1, 0, 2)
        assert input_ids.shape == (self.batch_size, self.max_doc_len, self.hidden_dim), gen_assertion_message(target_name='input_ids', target=input_ids, expected=(self.batch_size, self.max_doc_len, self.hidden_dim))
        # TODO: paddingした文の数を記録しておいて、attention maskをかける。
        attention_mask = torch.ones_like(input_ids)

        last_hidden_state_sent_level, pooler_output_sent_level = [], []
        for _input_ids, _attention_mask in zip(input_ids, attention_mask):
            _last_hidden_state, _pooler_output = self.wordAttennet(input_ids=_input_ids, attention_mask=_attention_mask).values()
            last_hidden_state_word_level.append(_last_hidden_state)
            pooler_output_word_level.append(_pooler_output)
        # TODO: ld, fc, criterionを通して予測をする。


class WordAttention(nn.Module):
    def __init__(self,
        max_sent_len: int,
        hidden_dim: int,
        batch_size: int,
        pretrained_model: str = 'cl-tohoku/bert-large-japanese',
        ):
        super(WordAttention, self).__init__()

        self.max_sent_len = max_sent_len
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        self.bert = BertModel.from_pretrained(pretrained_model)
        tokenizer = BertJapaneseTokenizer(pretrained_model, additional_special_tokens=['<person>'])
        self.bert.resize_token_embeddings(len(tokenizer))
        # TODO: person tokenの初期化方法を決め手初期化する。
        # self.bert.embeddings.word_embeddings.weight[-1, :] = torch.zeros([self.model.config.hidden_size]) # ゼロで初期化する場合。

        # won't update word level bert layers
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        assert input_ids.shape == (self.batch_size, self.hidden_dim), gen_assertion_message(target_name=input_ids, target=input_ids.shape, expected=(self.batch_size, self.hidden_dim))
        assert attention_mask.shape == (self.batch_size, self.hidden_dim), gen_assertion_message(target_name='attention_mask', target=attention_mask, expected=(self.batch_size, self.hidden_dim))
        last_hidden_state, pooler_output = self.bert(input_ids, attention_mask)
        return last_hidden_state, pooler_output


class SentAttention(nn.Module):
    def __init__(self,
        max_doc_len: int,
        hidden_dim: int,
        batch_size: int,
        ):
        super(SentAttention, self).__init__()
        # TODO: bertモデルのinitialize　→　ランダムにinitializeして、学習する予定。

        self.max_doc_len = max_doc_len
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        self.bert = BertModel()

    def forward(self, input_ids, attention_mask):
        last_hidden_state, pooler_output = self.bert(input_ids, attention_mask)
        return last_hidden_state, pooler_output


def gen_assertion_message(target_name: str, target: torch.tensor, expected: Tuple) -> str:
    return f'The shape of {target_name} is abnormal. {target_name}.shape:{target.shape}, expected:{expected}'