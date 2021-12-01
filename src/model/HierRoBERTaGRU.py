from typing import Tuple
from omegaconf import DictConfig
from omegaconf.omegaconf import OmegaConf
import torch
from torch import nn
import pandas as pd
import pytorch_lightning as pl
from torch.nn.modules.loss import CrossEntropyLoss
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1, ConfusionMatrix
from transformers import AutoModel
import hydra

class HierchicalRoBERTaGRU(pl.LightningModule):
    def __init__(
        self,
        num_labels: int,
        pretrained_model: str,
        use_ave_pooled_output: bool,
        sent_level_config: DictConfig,
        classifier_config: DictConfig,
        optim,
        output_attentions: bool,
        ):
        super(HierchicalRoBERTaGRU, self).__init__()
        self.save_hyperparameters()

        self.word_level_roberta = RoBERTaWordLevel(
            output_attentions=output_attentions,
            pretrained_model=pretrained_model,
            use_ave_pooled_output=use_ave_pooled_output,
        )

        self.sent_level_bigru = hydra.utils.instantiate(
            sent_level_config,
            num_labels=num_labels,
        )

        self.classifier = Classifier(
            **OmegaConf.to_container(classifier_config),
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
        last_hidden_state_word_level, pooled_output_word_level, word_attentions = [], [], []
        for _input_ids, _attention_mask in zip(input_ids, attention_mask):
            outputs = self.word_level_roberta(input_ids=_input_ids, attention_mask=_attention_mask)
            last_hidden_state_word_level.append(outputs['last_hidden_state'])
            pooled_output_word_level.append(outputs['pooled_output'])
            if self.hparams.output_attentions:
                word_attentions.append(outputs['attentions'])
        # TODO: debug here later to check the shape of inputs_embeds
        inputs_embeds = torch.stack(pooled_output_word_level).permute(1, 0, 2)
        sent_attentions, doc_embedding = self.sent_level_bigru(inputs_embeds=inputs_embeds)
        loss, logits = self.classifier(doc_embedding, labels)
        return dict(loss=loss, logits=logits, word_attentions=word_attentions, sent_attentions=sent_attentions)

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

        # won't update word level bert layers
        for param in self.robert.parameters():
            param.requires_grad = False

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        last_hidden_state, pooled_output, attentions = self.robert(input_ids, attention_mask, output_attentions=self.hparams.output_attentions).values()
        if self.hparams.use_ave_pooled_output:
            pooled_output = last_hidden_state.mean(dim=1)
        return dict(pooled_output=pooled_output, attentions=average_attention(attentions))


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