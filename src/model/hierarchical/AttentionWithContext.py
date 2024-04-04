import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


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
        u = self.contx(torch.tanh(self.atten(h_t))) #inp: the output of the word-GRU as same as HAN's paper
        u = u.masked_fill(attention_mask.unsqueeze(2), -1e4)
        a = F.softmax(u, dim=1)
        s = (a * h_t).sum(1)
        return a.permute(0, 2, 1), s
