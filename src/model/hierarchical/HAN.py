import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from src.model.hierarchical.regularize import embedded_dropout, WeightDrop, LockedDropout
from src.model.hierarchical.AbstractHierModel import AbstractHierModel

"""hierarchical attention network"""
class HierAttnNet(AbstractHierModel):
    def __init__(self, wordattennet_params: dict, sentattennet_params: dict,
                 embedding_matrix, last_drop: float, word_hidden_dim: int, sent_hidden_dim: int, num_class: int, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        
        self.build_wordattennet(**wordattennet_params, embedding_matrix=embedding_matrix)
        self.build_sentattennet(**sentattennet_params)
    
        self.last_drop = nn.Dropout(p=last_drop)
        self.fc = nn.Linear(sent_hidden_dim * 2, num_class)
        self.criterion = nn.CrossEntropyLoss()

    def build_wordattennet(self, *args, **kwargs):
        self.wordattnnet = WordAttnNet(*args, **kwargs)
        
    def build_sentattennet(self, *args, **kwargs):
        self.sentattennet = SentAttnNet(*args, **kwargs)

    def forward(self, nested_utters, labels, attention_mask, pad_sent_num, **kwargs):
        x = nested_utters.permute(1, 0, 2) # X: (batch_size, doc_len, sent_len) -> x: (doc_len, bsz, sent_len)
        attention_mask = attention_mask.permute(1, 0, 2)
        attention_mask = attention_mask == 0
        word_h_n = torch.zeros(2, nested_utters.shape[0], self.hparams.word_hidden_dim, device=self.device)

        #alpha and s Tensor List
        word_a_list, word_s_list = [], []
        for sent, _attention_mask in zip(x, attention_mask): # sent: (bsz, sent_len)
            word_a, word_s, word_h_n = self.wordattnnet(sent, word_h_n, _attention_mask)
            word_a_list.append(word_a)
            word_s_list.append(word_s)
        #Importance attention weights per word in sentence
        self.sent_a = torch.cat(word_a_list, 1)
        #Sentence representation
        sent_s = torch.cat(word_s_list, 1)
        sent_attention_mask = torch.ones_like(sent_s)
        max_doc_len = sent_s.shape[1]
        for idx, _pad_sent_num in enumerate(pad_sent_num):
            sent_attention_mask[idx, max_doc_len - _pad_sent_num:,:] = 0
        sent_attention_mask = sent_attention_mask == 0
        #Importance attention weights per sentence in doc and document representation

        doc_a, doc_s = self.sentattennet(sent_s, sent_attention_mask)

        self.doc_a = doc_a.permute(0, 2, 1)
        doc_s = self.last_drop(doc_s)
        preds = self.fc(doc_s) # (bsz, class_num)
        loss = self.criterion(preds, labels)
        return dict(loss=loss, preds=preds, word_attentions=self.sent_a, sent_attentions=self.doc_a, **kwargs)


class WordAttnNet(pl.LightningModule):
    def __init__(
            self,
            vocab_size: int,
            hidden_dim: int = 32,
            padding_idx: int = 1,
            embedding_matrix=None,
            embed_drop: float = 0.0,
            weight_drop: float = 0.0,
            locked_drop: float = 0.0,
    ):
        super(WordAttnNet, self).__init__()
        self.save_hyperparameters()

        self.lockdrop = LockedDropout(p=locked_drop)

        #if isinstance(embedding_matrix, np.ndarray):
        embed_dim = embedding_matrix.shape[1]
        self.word_embed = nn.Embedding(
            vocab_size, embed_dim, padding_idx=padding_idx
        )
        self.word_embed.weight = nn.Parameter(torch.tensor(embedding_matrix))
        #else:
        #    self.word_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

        self.rnn = nn.GRU(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        if weight_drop:
            self.rnn = WeightDrop(
                self.rnn, ["weight_hh_l0", "weight_hh_l0_reverse"], dropout=weight_drop , device=self.device
            )
        self.word_atten = AttentionWithContext(hidden_dim * 2) # since GRU is bidirectional


    def forward(self, X, h_n, attention_mask):
        r"""
        :param X: each review in the batch. One sentence at a time. (bsz, seq_len)
        :param h_n(h_0): initial hidden state for each element in the batch. (num_layers*num_directions, batch, hidden_size)
        :var embed: input for GRU. (seq_len, batch, input_size) #according to official document.
        :var h_t: tensor containing the output features h_t from the last layer of the GRU, for each t(time). (bsz, seq_len, hidden_dim*2)
        :var h_n(return from GRU): tensor containing the hidden state for t=seq_len Like output
        :return:
        """
        if self.hparams.embed_drop:
            embed = embedded_dropout(
                self.word_embed, X.long(), dropout=self.hparams.embed_drop if self.training else 0,
            )
        else:
            embed = self.word_embed(X.long())
        if self.lockdrop:
            embed = self.lockdrop(embed)

        h_t, h_n = self.rnn(embed, h_n)
        a, s = self.word_atten(h_t, attention_mask)
        return a, s.unsqueeze(1), h_n


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