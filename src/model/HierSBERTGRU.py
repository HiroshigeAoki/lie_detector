from typing import Tuple
from omegaconf import DictConfig
from omegaconf.omegaconf import OmegaConf
import torch
from torch import nn
import pandas as pd
import pytorch_lightning as pl
from torch.nn.modules.loss import CrossEntropyLoss
import torch.nn.functional as F
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1, ConfusionMatrix
from transformers import AutoModel
import hydra

class HierarchicalSBERTGRU(pl.LightningModule):
    def __init__(
        self,
        num_labels: int,
        pretrained_model: str,
        sent_level_config: DictConfig,
        classifer_config: DictConfig,
        optim: DictConfig,
        ):
        super(HierarchicalSBERTGRU, self).__init__()
        self.save_hyperparameters()

        self.word_level_sbert = SentenceEmbedder(
            pretrained_model=pretrained_model,
        )

        self.sent_level_bigru = hydra.utils.instantiate(
            sent_level_config,
            num_labels=num_labels,
        )

        self.classifier = Classifier(
            **OmegaConf.to_container(classifer_config),
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
        sent_embeddings = []
        for _input_ids, _attention_mask in zip(input_ids, attention_mask):
            sent_embeddings.append(self.word_level_sbert(input_ids=_input_ids, attention_mask=_attention_mask))
        # TODO: input_embedをdebugする。
        input_embeds = torch.stack(sent_embeddings).permute(1, 0, 2)
        sent_attentions, sent_embeddings = self.sent_level_bigru(inputs_embeds=input_embeds, pad_sent_num=pad_sent_num)
        loss, logits = self.classifier(sent_embeddings, labels)
        return dict(loss=loss, logits=logits, sent_attentions=sent_attentions)

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

class SentenceEmbedder(pl.LightningModule):
    def __init__(self,
        pretrained_model: str,
        ):
        super(SentenceEmbedder, self).__init__()
        self.save_hyperparameters()

        self.sbert = AutoModel(pretrained_model)

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.sbert(input_ids, attention_mask)
        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, attention_mask)
        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings


class Classifier(pl.LightningModule):
    def __init__(self, num_labels, drop_out, hidden_size):
        super(Classifier, self).__init__()
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


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)