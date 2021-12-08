from typing import Tuple
from omegaconf import DictConfig
from omegaconf.omegaconf import OmegaConf
import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
import pytorch_lightning as pl
from torch.nn.modules.loss import CrossEntropyLoss
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1, ConfusionMatrix
from transformers import AutoModel
import hydra

from src.model.HierHFMoulde import HierarchicalHFModule
from src.model.regularize import WeightDrop


class HierchicalRoBERTaGRU(pl.LightningModule):
    def __init__(
        self,
        num_labels: int,
        pretrained_model: str,
        sent_level_config: DictConfig,
        classifier_config: DictConfig,
        optim,
        use_ave_pooled_output: bool,
        output_attentions: bool,
        ):
        super(HierchicalRoBERTaGRU, self).__init__()
        self.save_hyperparameters()

        self.word_level_roberta = RoBERTaWordLevel(
            output_attentions=output_attentions,
            pretrained_model=pretrained_model,
            use_ave_pooled_output=use_ave_pooled_output,
        )

        self.sent_level_bigru = SentAttnNet(
            **OmegaConf.to_container(sent_level_config),
        )

        self.classifier = Classifier(
            **OmegaConf.to_container(classifier_config),
            num_labels=num_labels,
            hidden_size=sent_level_config.sent_hidden_dim * 2,
        )

        self.loss_fct = CrossEntropyLoss()

        metrics = MetricCollection([
            Accuracy(num_classes=num_labels, average='macro'),
            Precision(num_classes=num_labels, average='macro'),
            Recall(num_classes=num_labels, average='macro'),
            F1(num_classes=num_labels, average='macro')
        ])

        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='valid_')
        self.test_metrics = metrics.clone(prefix='test_')

        self.cm = ConfusionMatrix(num_classes=num_labels, compute_on_step=False)

    def forward(self, input_ids: torch.FloatTensor, attention_mask: torch.LongTensor, pad_sent_num: torch.tensor, labels: torch.tensor):
        input_ids = input_ids.permute(1,0,2)
        attention_mask = attention_mask.permute(1,0,2)
        pooled_output_word_level, word_attentions = [], []
        for _input_ids, _attention_mask in zip(input_ids, attention_mask):
            outputs = self.word_level_roberta(input_ids=_input_ids, attention_mask=_attention_mask)
            pooled_output_word_level.append(outputs['pooled_output'])
            if self.hparams.output_attentions:
                word_attentions.append(outputs['attentions'])
        inputs_embeds = torch.stack(pooled_output_word_level).permute(1, 0, 2) # (sent_len, batch_size, hidden_dim) -> (batch_size, sent_len, hidden_dim)
        attention_mask = torch.ones_like(inputs_embeds)
        for idx, _pad_sent_num in enumerate(pad_sent_num):
            attention_mask[idx, _pad_sent_num:, :] = 0
        attention_mask = attention_mask == 0
        sent_attentions, doc_embedding = self.sent_level_bigru(inputs_embeds, attention_mask)
        loss, logits = self.classifier(doc_embedding, labels)
        return dict(loss=loss, logits=logits, word_attentions=word_attentions, sent_attentions=sent_attentions.flatten(1))

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        return {'loss': outputs['loss'], 'batch_preds': outputs['logits'], 'batch_labels': batch['labels']}

    def training_step_end(self, outputs):
        output = self.train_metrics(outputs['batch_preds'], outputs['batch_labels'])
        self.log_dict(output)

    def training_epoch_end(self, outputs):
        epoch_preds = torch.cat([x['batch_preds'] for x in outputs])
        epoch_labels = torch.cat([x['batch_labels'] for x in outputs])
        epoch_loss = self.loss_fct(epoch_preds, epoch_labels)
        self.log("train_loss", epoch_loss, logger=True)
        self.log_dict(self.train_metrics.compute(), logger=True)

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        return {'loss': outputs['loss'], 'batch_preds': outputs['logits'], 'batch_labels': batch['labels']}

    def validation_step_end(self, outputs):
        output = self.valid_metrics(outputs['batch_preds'], outputs['batch_labels'])
        self.log_dict(output)

    def validation_epoch_end(self, outputs):
        epoch_preds = torch.cat([x['batch_preds'] for x in outputs])
        epoch_labels = torch.cat([x['batch_labels'] for x in outputs])
        epoch_loss = self.loss_fct(epoch_preds, epoch_labels)
        self.log("val_loss", epoch_loss, logger=True)
        self.log_dict(self.valid_metrics.compute(), logger=True)

    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        return {'loss': outputs['loss'], 'batch_preds': outputs['logits'], 'batch_labels': batch['labels']}

    def test_step_end(self, outputs):
        output = self.test_metrics(outputs['batch_preds'], outputs['batch_labels'])
        self.cm(outputs['batch_preds'], outputs['batch_labels'])
        self.log_dict(output)

    def test_epoch_end(self, outputs):
        logits = torch.cat([x['batch_preds'] for x in outputs])
        labels = torch.cat([x['batch_labels'] for x in outputs])
        epoch_loss = self.loss_fct(logits, labels)
        from sklearn.metrics import precision_recall_fscore_support
        import numpy as np
        num_correct = (logits.argmax(dim=1) == labels).sum().item()
        epoch_accuracy = num_correct / len(labels)
        self.log("test_accuracy", epoch_accuracy, logger=True)
        cm = ConfusionMatrix(num_classes=2)
        df_cm = pd.DataFrame(cm(logits.argmax(dim=1).cpu(), labels.cpu()).numpy())
        self.print(f"confusion_matrix\n{df_cm.to_string()}\n")
        scores_df = pd.DataFrame(np.array(precision_recall_fscore_support(labels.cpu(), logits.argmax(dim=1).cpu())).T,
                                    columns=["precision", "recall", "f1", "support"],
                                )
        self.print(f"f1_precision_accuracy\n{scores_df.to_string()}")
        return {'loss': epoch_loss, 'epoch_preds': logits, 'labels': labels}

    def predict_step(self, batch, batch_idx: int):
        outputs = self(**batch)
        return dict(loss=outputs['loss'], logits=outputs['logits'], word_attentions=outputs['word_attentions'], sent_attentions=outputs['sent_attentions'], input_ids=batch['input_ids'], labels=batch['labels'])

    def configure_optimizers(self):
        return hydra.utils.instantiate(self.hparams.optim.optimizer, params=self.parameters())



class RoBERTaWordLevel(pl.LightningModule):
    def __init__(self,
        output_attentions: bool,
        pretrained_model: str,
        use_ave_pooled_output: bool, # CLS or average
        ):
        super(RoBERTaWordLevel, self).__init__()
        self.save_hyperparameters()

        self.robert = AutoModel.from_pretrained(pretrained_model)

        """# won't update word level bert layers
        for param in self.robert.parameters():
            param.requires_grad = False"""

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        last_hidden_state, pooled_output, attentions = self.robert(input_ids, attention_mask, output_attentions=self.hparams.output_attentions).values()
        if self.hparams.use_ave_pooled_output:
            pooled_output = last_hidden_state.mean(dim=1)
        return dict(pooled_output=pooled_output, attentions=average_attention(attentions))


class SentAttnNet(pl.LightningModule):
    def __init__(
            self,
            word_hidden_dim: int = 32,
            sent_hidden_dim: int = 32,
            weight_drop: float = 0.0,
    ):
        super(SentAttnNet, self).__init__()

        self.rnn = nn.GRU(
            word_hidden_dim, sent_hidden_dim, bidirectional=True, batch_first=True
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


    def forward(self, h_t, attention_mask):
        """caliculate the sentence vector s which is the weighted sum of word hidden states inp

        Args:
            h_t ([type]): word annotation

        Returns:
            [type]: [description]
        """
        u = torch.tanh_(self.atten(h_t)).float() #inp: the output of the word-GRU as same as HAN's paper
        u = u.masked_fill(attention_mask.repeat_interleave(2, dim=2), -1e4)
        a = F.softmax(self.contx(u), dim=1)
        s = (a * h_t).sum(1)
        return a.permute(0, 2, 1), s


class Classifier(pl.LightningModule):
    def __init__(self, num_labels, drop_out, hidden_size):
        super(Classifier, self).__init__()
        self.save_hyperparameters()

        self.num_labels = num_labels
        classifier_dropout = drop_out
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.loss_fct = CrossEntropyLoss()

    def forward(self, pooled_output, labels):
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss, logits


def average_attention(attentions):
    return torch.stack(attentions).mean(3).mean(2).mean(0)