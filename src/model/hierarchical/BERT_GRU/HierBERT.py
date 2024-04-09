import torch
from torch import nn

from src.model.AbstractHierModel import AbstractHierModel
from src.model.BERT_GRU.BERT import BERT
from src.model.BERT_GRU.GRU import GRU


class HierBERT(AbstractHierModel):
    def __init__(
        self,
        wordattennet_params: dict,
        sentattennet_params: dict,
        optim: dict,
        dropout: int = 0.1,
        num_class: int = 2,
        use_gmail_notification: bool = False,
        use_attention: bool = False,
    ):
        super().__init__(
            optim=optim,
            use_gmail_notification=use_gmail_notification,
        )
        self.save_hyperparameters()

        self.build_wordattennet(**wordattennet_params)
        self.build_sentattennet(**sentattennet_params, dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(768, 256)

        self.last_dropout = nn.Dropout(dropout)
        self.last_fc = nn.Linear(256, num_class)
        # self.fc = nn.Linear(sentattennet_params.hidden_dim*4, num_class)
        self.criterion = nn.CrossEntropyLoss()

    def build_wordattennet(self, *args, **kwargs):
        self.wordattennet = BERT(
            use_attention=self.hparams.use_attention, *args, **kwargs
        )

    def build_sentattennet(self, hidden_dim: int, dropout: float):
        # if not self.hparams.use_attention:
        #     hidden_dim = hidden_dim * 4

        self.sentattennet = GRU(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            batch_first=True,
            use_attention=self.hparams.use_attention,
        )

    def forward(
        self, input_ids, attention_mask, pad_sent_num, labels=None, indices=None
    ):
        input_ids = input_ids.permute(1, 0, 2)
        attention_mask = attention_mask.permute(1, 0, 2)

        word_pooled_outputs = []
        word_attentions = []

        for _input_ids, _attention_mask in zip(input_ids, attention_mask):
            outputs = self.wordattennet(_input_ids, _attention_mask)
            word_pooled_outputs.append(outputs["pooled_output"])
            word_attentions.append(outputs["attentions"])

        word_pooled_outputs = torch.cat(word_pooled_outputs, dim=1)
        if self.hparams.use_attention:
            word_attentions = torch.cat(word_attentions, dim=1)
        else:
            word_attentions = None

        lengths = word_pooled_outputs.shape[1] - torch.tensor(pad_sent_num)

        sentattennet_outputs = self.sentattennet(
            self.fc(self.dropout(word_pooled_outputs)), lengths
        )
        # sentattennet_outputs = self.sentattennet(word_pooled_outputs[:,:20])

        dropout_outputs = self.last_dropout(sentattennet_outputs["weighted_outputs"])

        if dropout_outputs.shape[0] != 1:
            dropout_outputs = dropout_outputs.squeeze()

        preds = self.last_fc(dropout_outputs)
        loss = self.criterion(preds, labels) if labels is not None else None

        return dict(
            loss=loss,
            preds=preds,
            word_attentions=word_attentions,
            sent_attentions=sentattennet_outputs["attentions"],
        )
