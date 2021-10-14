import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics import Accuracy, MetricCollection, Precision, Recall, F1, ConfusionMatrix
import pytorch_lightning as pl
import hydra

from project.model.regularize import embedded_dropout, WeightDrop, LockedDropout

"""hierarchical attention network"""
class HierAttnNet(pl.LightningModule):
    def __init__(
            self,
            optim,
            vocab_size: int,
            word_hidden_dim: int = 32,
            sent_hidden_dim: int = 32,
            padding_idx: int = 1,
            embedding_matrix=None,
            num_class: int = 2,
            weight_drop: float = 0.0,
            embed_drop: float = 0.0,
            locked_drop: float = 0.0,
            last_drop: float = 0.0,
    ):
        super(HierAttnNet, self).__init__()
        self.save_hyperparameters()

        self.wordattnnet = WordAttnNet(
            vocab_size=vocab_size,
            hidden_dim=word_hidden_dim,
            padding_idx=padding_idx,
            embedding_matrix=embedding_matrix,
            weight_drop=weight_drop,
            embed_drop=embed_drop,
            locked_drop=locked_drop,
        )

        self.sentattennet = SentAttnNet(
            word_hidden_dim=word_hidden_dim,
            sent_hidden_dim=sent_hidden_dim,
            padding_idx=padding_idx,
            weight_drop=weight_drop,
        )

        self.last_drop = nn.Dropout(p=last_drop)
        self.fc = nn.Linear(sent_hidden_dim * 2, num_class)
        self.criterion = nn.CrossEntropyLoss()

        metrics = MetricCollection([
            Accuracy(num_classes=2, average='macro'),
            Precision(num_classes=2, average='macro'),
            Recall(num_classes=2, average='macro'),
            F1(num_classes=2, average='macro')
        ])

        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='valid_')

        self.test_metrics = MetricCollection([
            Accuracy(num_classes=2, average='macro'),
            Precision(num_classes=2, average='macro'),
            Recall(num_classes=2, average='macro'),
            F1(num_classes=2, average='macro'),
            ConfusionMatrix(num_classes=2)
        ])

        # self.example_input_array = (torch.from_numpy(np.random.choice(vocab_size, (160, 150))), torch.tensor(0))


    def forward(self, X, y):
        x = X.permute(1, 0, 2) # X: (batch_size, doc_len, sent_len) -> x: (doc_len, bsz, sent_len)
        word_h_n = torch.zeros(2, X.shape[0], self.hparams.word_hidden_dim)
        # word_h_n = nn.init.zeros_(torch.Tensor(2, X.shape[0], self.word_hidden_dim).to)

        #alpha and s Tensor List
        word_a_list, word_s_list = [], []
        for sent in x: # sent: (bsz, sent_len)
            word_a, word_s, word_h_n = self.wordattnnet(sent, word_h_n)
            word_a_list.append(word_a)
            word_s_list.append(word_s)
        #Importance attention weights per word in sentence
        self.sent_a = torch.cat(word_a_list, 1)
        #Sentence representation
        sent_s = torch.cat(word_s_list, 1)
        #Importance attention weights per sentence in doc and document representation
        doc_a, doc_s = self.sentattennet(sent_s)
        self.doc_a = doc_a.permute(0, 2, 1)
        doc_s = self.last_drop(doc_s)
        preds = self.fc(doc_s) # (bsz, class_num)
        loss = self.criterion(preds, y)
        return loss, preds


    def training_step(self, batch, batch_idx):
        loss, preds = self(batch['nested_utters'], batch['labels'])
        return {'loss': loss, 'batch_preds': preds, 'batch_labels': batch['labels']}

    def training_step_end(self, outputs):
        output = self.train_metrics(outputs['preds'], outputs['batch_labels'])
        self.log_dict(output)

    def training_epoch_end(self, outputs):
        epoch_preds = torch.cat([x['batch_preds'] for x in outputs])
        epoch_labels = torch.cat([x['batch_labels'] for x in outputs])
        epoch_loss = self.criterion(epoch_preds, epoch_labels)
        self.log("train_loss", epoch_loss, logger=True)
        self.log_dict(self.train_metrics.compute(), logger=True)

    def validation_step(self, batch, batch_idx):
        loss, preds = self(batch['nested_utters'], batch['labels'])
        return {'loss': loss, 'batch_preds': preds, 'batch_labels': batch['labels']}

    def validation_step_end(self, outputs):
        output = self.valid_metrics(outputs['preds'], outputs['batch_labels'])
        self.log_dict(output)

    def validation_epoch_end(self, outputs):
        epoch_preds = torch.cat([x['batch_preds'] for x in outputs])
        epoch_labels = torch.cat([x['batch_labels'] for x in outputs])
        epoch_loss = self.criterion(epoch_preds, epoch_labels)
        self.log("val_loss", epoch_loss, logger=True)
        self.log_dict(self.valid_metrics.compute(), logger=True)

    def test_step(self, batch, batch_idx):
        loss, preds = self(batch['nested_utters'], batch['labels']) # call forward()
        return {'loss': loss.detach(), 'batch_preds': preds.detach(), 'batch_labels': batch['labels']}

    def test_step_end(self, outputs):
        output = self.test_metrics(outputs['preds'], outputs['batch_labels'])
        self.cm(outputs['preds'], outputs['batch_labels'])
        self.log_dict(output)

    def test_epoch_end(self, outputs):
        preds = torch.cat([x['batch_preds'] for x in outputs])
        labels = torch.cat([x['batch_labels'] for x in outputs])
        epoch_loss = self.criterion(preds, labels)
        self.log("test_loss", epoch_loss, logger=True)
        self.log_dict(self.test_metrics.compute(), logger=True)

    def configure_optimizers(self):
        return hydra.utils.instantiate(self.hparams.optim, params=self.parameters())




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
                self.rnn, ["weight_hh_l0", "weight_hh_l0_reverse"], dropout=weight_drop# , device=self.device #必要かも、、
            )
        self.word_atten = AttentionWithContext(hidden_dim * 2) # since GRU is bidirectional


    def forward(self, X, h_n):
        r"""
        :param X: each review in the batch. One sentence at a time. (bsz, seq_len)
        :param h_n(h_0): initial hidden state for each element in the batch. (num_layers*num_directions, batch, hidden_size)
        :var embed: input for GRU. (seq_len, batch, input_size) #according to official document.
        :var h_t: tensor containing the output features h_t from the last layer of the GRU, for each t(time). (bsz, seq_len, hidden_dim*2)
        :var h_n(return from GRU): tensor containing the hidden state for t=seq_len Like output
        :return:
        """
        if self.embed_drop:
            embed = embedded_dropout(
                self.word_embed, X.long(), dropout=self.hparams.embed_drop if self.training else 0,
            )
        else:
            embed = self.word_embed(X.long())
        if self.lockdrop:
            embed = self.lockdrop(embed)

        h_t, h_n = self.rnn(embed, h_n)
        a, s = self.word_atten(h_t)
        return a, s.unsqueeze(1), h_n


# TODO: インターフェースをhierBERT.pyのsentlevelBERTと同じにする。
class SentAttnNet(pl.LightningModule):
    def __init__(
            self,
            word_hidden_dim: int = 32,
            sent_hidden_dim: int = 32,
            weight_drop: float = 0.0,
            padding_idx: int = 1,
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


    def forward(self, X): #will receive a tensor of dim(bsz, doc_len, word_hidden_dim * 2)
        h_t, h_n = self.rnn(X)
        a, v = self.sent_attn(h_t)
        return a, v #doc vector (bsz, sent_hidden_dim*2)


class AttentionWithContext(pl.LightningModule):
    def __init__(self, hidden_dim: int):

        super(AttentionWithContext, self).__init__()

        self.atten = nn.Linear(hidden_dim, hidden_dim)
        self.contx = nn.Linear(hidden_dim, 1, bias=False)


    def forward(self, h_t):
        """caliculate the sentence vector s which is the weighted sum of word hidden states inp

        Args:
            h_t ([type]): word annotation

        Returns:
            [type]: [description]
        """
        u = torch.tanh_(self.atten(h_t)) #inp: the output of the word-GRU as same as HAN's paper
        a = F.softmax(self.contx(u), dim=1)
        s = (a * h_t).sum(1)
        return a.permute(0, 2, 1), s