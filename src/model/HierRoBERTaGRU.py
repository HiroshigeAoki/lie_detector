from logging import raiseExceptions
from typing import Tuple
from omegaconf import DictConfig
import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
import pytorch_lightning as pl
from torch.nn.modules.loss import CrossEntropyLoss
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1, ConfusionMatrix
from transformers import AutoModel, AutoTokenizer
import hydra

from src.model.regularize import WeightDrop


class HierchicalRoBERTaGRU(pl.LightningModule):
    def __init__(
        self,
        num_labels: int,
        pretrained_model: str,
        output_attentions: bool,
        sent_embed_dim: int,
        doc_embed_dim: int,
        weight_drop: float,
        classifier_drop_out: float,
        pooling_strategy: str,
        optim: DictConfig,
        update_last_layer: bool,
        additional_special_tokens: list(str) = None,
        ):
        super(HierchicalRoBERTaGRU, self).__init__()
        self.save_hyperparameters()

        self.word_level_roberta = RoBERTaWordLevel(
            output_attentions=output_attentions,
            pretrained_model=pretrained_model,
            sent_embed_dim=sent_embed_dim,
            pooling_strategy=pooling_strategy,
            update_last_layer=update_last_layer,
            additional_special_tokens=additional_special_tokens,
        )

        self.sent_level_bigru = SentAttnNet(
            sent_embed_dim=sent_embed_dim,
            doc_embed_dim=doc_embed_dim,
            weight_drop=weight_drop,
        )

        self.classifier = Classifier(
            drop_out=classifier_drop_out,
            num_labels=num_labels,
            hidden_size=doc_embed_dim * 2,
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
        max_doc_len = inputs_embeds.shape[1]
        for idx, _pad_sent_num in enumerate(pad_sent_num):
            attention_mask[idx, max_doc_len - _pad_sent_num:, :] = 0
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
        return hydra.utils.instantiate(self.hparams.optim.args, params=self.parameters())


class RoBERTaWordLevel(pl.LightningModule):
    def __init__(self,
        pretrained_model: str,
        sent_embed_dim: int,
        pooling_strategy: str,
        output_attentions: bool,
        update_last_layer: bool,
        additional_special_tokens: list(str) = None,
        ):
        super(RoBERTaWordLevel, self).__init__()
        self.save_hyperparameters()

        self.roberta = AutoModel.from_pretrained(pretrained_model)

        if additional_special_tokens is not None and '<person>' in additional_special_tokens and len(additional_special_tokens) == 1:
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model, additional_special_tokens=additional_special_tokens)
            self.roberta.resize_token_embeddings(len(tokenizer))
            # initialize  <person> token by the average of some personal_pronouns's weights.
            personal_pronouns = ['君', 'きみ', 'あなた' ,'彼', '彼女']
            personal_pronouns_weights = torch.stack([self.roberta.embeddings.word_embeddings.weight[i, :] for i in tokenizer.convert_tokens_to_ids(personal_pronouns)])
            self.roberta.embeddings.word_embeddings.weight.data[-1, :] = personal_pronouns_weights.mean(dim=0)
        else:
            raise ValueError(f"Additional tokens:{additional_special_tokens} except for the '<person>' token are currently not supported.")

        self.linear = nn.Linear(self.roberta.config.hidden_size, sent_embed_dim, bias=False)

        self.pooling_strategy = pooling_strategy

        """# only update the last word level bert layers"""
        for param in self.roberta.parameters():
            param.requires_grad = False
        if update_last_layer:
            for param in self.roberta.encoder.layer[-1].parameters():
                param.requires_grad = True

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        outputs = self.roberta(input_ids, attention_mask, output_attentions=self.hparams.output_attentions)
        if self.pooling_strategy=='mean':
            pooled_output=[]
            for batch, mask in zip(outputs['last_hidden_state'], attention_mask):
                accutual_sent_len = int(mask.sum())
                pooled_output.append(batch[:accutual_sent_len].mean(0))
            pooled_output = self.linear(torch.stack(pooled_output, dim=0))
        elif self.pooling_strategy=='max':
            pooled_output = self.linear(outputs['last_hidden_state'].max(1)[0])
        else:
            raiseExceptions(f'pooling_strategy "{self.pooling_strategy}" is invailed.')
        return dict(pooled_output=pooled_output, attentions=average_attention(outputs['attentions']) if self.hparams.output_attentions else None)


class SentAttnNet(pl.LightningModule):
    def __init__(
            self,
            sent_embed_dim: int,
            doc_embed_dim: int,
            weight_drop: float,
    ):
        super(SentAttnNet, self).__init__()

        self.rnn = nn.GRU(
            sent_embed_dim, doc_embed_dim, bidirectional=True, batch_first=True
        )
        if weight_drop:
            self.rnn = WeightDrop(
                self.rnn, ["weight_hh_l0", "weight_hh_l0_reverse"], dropout=weight_drop, device=self.device
            )

        self.sent_attn = AttentionWithContext(doc_embed_dim * 2)

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