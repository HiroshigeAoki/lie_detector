import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics import Accuracy, MetricCollection, Precision, Recall, F1, ConfusionMatrix
import pytorch_lightning as pl
from sklearn.metrics import precision_recall_fscore_support
import hydra

from src.model.regularize import embedded_dropout, WeightDrop, LockedDropout


"""hierarchical attention network"""
class HierAttnNet(pl.LightningModule):
    def __init__(
            self,
            optim,
            vocab_size: int,
            word_hidden_dim: int,
            sent_hidden_dim: int,
            padding_idx: int,
            embedding_matrix,
            num_class: int,
            weight_drop: float,
            embed_drop: float,
            locked_drop: float,
            last_drop: float,
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
        self.test_metrics = metrics.clone(prefix='test_')

        self.cm = ConfusionMatrix(num_classes=2, compute_on_step=False)


    def forward(self, X, y, pad_sent_num):
        x = X.permute(1, 0, 2) # X: (batch_size, doc_len, sent_len) -> x: (doc_len, bsz, sent_len)
        word_h_n = torch.zeros(2, X.shape[0], self.hparams.word_hidden_dim, device=self.device)

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
        attention_mask = torch.ones_like(sent_s)
        max_doc_len = sent_s.shape[1]
        for idx, _pad_sent_num in enumerate(pad_sent_num):
            attention_mask[idx, max_doc_len - _pad_sent_num:, :] = 0
        attention_mask = attention_mask == 0
        #Importance attention weights per sentence in doc and document representation
        doc_a, doc_s = self.sentattennet(sent_s, attention_mask)
        self.doc_a = doc_a.permute(0, 2, 1)
        doc_s = self.last_drop(doc_s)
        preds = self.fc(doc_s) # (bsz, class_num)
        loss = self.criterion(preds, y)
        return dict(loss=loss, preds=preds, word_attentions=self.sent_a, sent_attentions=self.doc_a)


    def training_step(self, batch, batch_idx):
        outputs = self(batch['nested_utters'], batch['labels'], batch['pad_sent_num'])
        return {'loss': outputs['loss'], 'batch_preds': outputs['preds'], 'batch_labels': batch['labels']}


    def training_step_end(self, outputs):
        output = self.train_metrics(outputs['batch_preds'], outputs['batch_labels'])
        self.log_dict(output)


    def training_epoch_end(self, outputs):
        epoch_preds = torch.cat([x['batch_preds'] for x in outputs])
        epoch_labels = torch.cat([x['batch_labels'] for x in outputs])
        epoch_loss = self.criterion(epoch_preds, epoch_labels)
        self.log("train_loss", epoch_loss, logger=True)
        self.log_dict(self.train_metrics.compute(), logger=True)


    def validation_step(self, batch, batch_idx):
        outputs = self(batch['nested_utters'], batch['labels'], batch['pad_sent_num'])
        return {'loss': outputs['loss'], 'batch_preds': outputs['preds'], 'batch_labels': batch['labels']}


    def validation_step_end(self, outputs):
        output = self.valid_metrics(outputs['batch_preds'], outputs['batch_labels'])
        self.log_dict(output)


    def validation_epoch_end(self, outputs):
        epoch_preds = torch.cat([x['batch_preds'] for x in outputs])
        epoch_labels = torch.cat([x['batch_labels'] for x in outputs])
        epoch_loss = self.criterion(epoch_preds, epoch_labels)
        self.log("val_loss", epoch_loss, logger=True)
        self.log_dict(self.valid_metrics.compute(), logger=True)


    def test_step(self, batch, batch_idx):
        outputs = self(batch['nested_utters'], batch['labels'], batch['pad_sent_num'])
        return {'loss': outputs['loss'], 'batch_preds': outputs['preds'], 'batch_labels': batch['labels']}


    def test_step_end(self, outputs):
        output = self.test_metrics(outputs['batch_preds'], outputs['batch_labels'])
        self.cm(outputs['batch_preds'], outputs['batch_labels'])
        self.log_dict(output)


    def test_epoch_end(self, outputs):
        preds = torch.cat([x['batch_preds'] for x in outputs])
        labels = torch.cat([x['batch_labels'] for x in outputs])
        epoch_loss = self.criterion(preds, labels)
        self.log("test_loss", epoch_loss, logger=True)
        cm = pd.DataFrame(self.cm.compute())
        test_metrix = self.test_metrics.compute()
        self.log_dict(test_metrix, logger=True)
        cm.to_csv(f'{self.logger.log_dir}/confusionmatrix.csv')
        pd.DataFrame([metrix.cpu().numpy() for metrix in test_metrix.values()], index=test_metrix.keys()).to_csv(f'{self.logger.log_dir}/scores.csv')
        # For debug
        num_correct = (preds.argmax(dim=1) == labels).sum().item()
        epoch_accuracy = num_correct / len(labels)
        scores_df = pd.DataFrame(np.array(precision_recall_fscore_support(labels.cpu(), preds.argmax(dim=1).cpu())).T,
                                    columns=["precision", "recall", "f1", "support"],
                                )
        scores_df.to_csv(f'{self.logger.log_dir}/precision_recall_fscore_support.csv')
        self.print(f"test_accuracy:{epoch_accuracy}")
        self.print(f"confusion_matrix\n{cm.to_string()}\n")
        self.print(f"f1_precision_accuracy\n{scores_df.to_string()}")


    def configure_optimizers(self):
        return hydra.utils.instantiate(self.hparams.optim.optimizer, params=self.parameters())


    def predict_step(self, batch, batch_idx):
        outputs = self(batch['nested_utters'], batch['labels'], batch['pad_sent_num'])
        return dict(input_ids=batch['nested_utters'], labels=batch['labels'], loss=outputs['loss'], logits=outputs['preds'], word_attentions=outputs['word_attentions'], sent_attentions=outputs['sent_attentions'])


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


    def forward(self, X, h_n):
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
        a, s = self.word_atten(h_t)
        return a, s.unsqueeze(1), h_n


class SentAttnNet(pl.LightningModule):
    def __init__(
            self,
            word_hidden_dim: int = 32,
            sent_hidden_dim: int = 32,
            weight_drop: float = 0.0,
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
        a, v = self.sent_attn(h_t, attention_mask)
        return a, v #doc vector (bsz, sent_hidden_dim*2)


class AttentionWithContext(pl.LightningModule):
    def __init__(self, hidden_dim: int):

        super(AttentionWithContext, self).__init__()

        self.atten = nn.Linear(hidden_dim, hidden_dim)
        self.contx = nn.Linear(hidden_dim, 1, bias=False)


    def forward(self, h_t, attention_mask=None):
        """caliculate the sentence vector s which is the weighted sum of word hidden states inp

        Args:
            h_t ([type]): word annotation

        Returns:
            [type]: [description]
        """
        u = torch.tanh_(self.atten(h_t)) #inp: the output of the word-GRU as same as HAN's paper
        if attention_mask is not None:
            u = u.masked_fill(attention_mask, -1e4)
        a = F.softmax(self.contx(u), dim=1)
        s = (a * h_t).sum(1)
        return a.permute(0, 2, 1), s