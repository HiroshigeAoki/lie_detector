import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from src.model.hierarchical.regularize import WeightDrop


class SentAttnNet(pl.LightningModule):
    def __init__(
            self,
            word_hidden_dim: int,
            sent_hidden_dim: int,
            weight_drop: float,
    ):
        super(SentAttnNet, self).__init__()

        self.rnn = nn.GRU(
            word_hidden_dim * 2, sent_hidden_dim, bidirectional=True, batch_first=True
        )
        if weight_drop:
            self.rnn = WeightDrop(
                self.rnn, ["weight_hh_l0", "weight_hh_l0_reverse"], dropout=weight_drop, device=self.device
            )

        self.sent_attn = AttentionWithContext(sent_hidden_dim * 2)

    def forward(self, X, attention_mask): #will receive a tensor of dim(bsz, doc_len, word_hidden_dim * 2)
        X = X.masked_fill(attention_mask, 0)
        h_t, h_n = self.rnn(X)
        a, v = self.sent_attn(h_t, attention_mask[:,:,0])
        return a, v #doc vector (bsz, sent_hidden_dim*2)


class AttentionWithContext(pl.LightningModule):
    def __init__(self, hidden_dim: int):

        super(AttentionWithContext, self).__init__()

        self.atten = nn.Linear(hidden_dim, hidden_dim)
        self.contx = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h_t, attention_mask):
        """caliculate the sentence vector s which is the weighted sum of word hidden states inp

        Args:
            h_t ([type]): word annotation

        Returns:
            [type]: [description]
        """
        u = self.contx(torch.tanh_(self.atten(h_t))) #inp: the output of the word-GRU as same as HAN's paper
        u = u.masked_fill(attention_mask.unsqueeze(2), -1e4)
        a = F.softmax(u, dim=1)
        s = (a * h_t).sum(1)
        return a.permute(0, 2, 1), s