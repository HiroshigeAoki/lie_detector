from typing import Tuple
from pytorch_lightning.utilities.parsing import save_hyperparameters
import torch
from torch import nn
import pytorch_lightning as pl
from torch.nn.modules.loss import CrossEntropyLoss
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1, ConfusionMatrix
from transformers import BertModel, BertJapaneseTokenizer, BertConfig
from transformers.models.bert.modeling_bert import BertPreTrainedModel
import hydra

from src.model.HAN import SentAttnNet
from src.model.HierBERT import BERTWordLevel, Classifier
from src.utils.gen_assertion import gen_torch_tensor_shape_assertion


class HierchicalBERTGRU(pl.LightningModule):
    def __init__(
        self,
        batch_size: int,
        hidden_size: int,
        sent_length: int,
        doc_length: int,
        num_labels: int,
        optim,
        pretrained_model: str,
        sent_level_BERT_config: BertConfig,
        use_ave_pooler_output: bool,
        use_sent_level_gru: bool,
        # TODO: dropout率については、後で考える。
        # TODO: optimもちゃんと入れる。
        ):
        super(HierchicalBERT, self).__init__()
        self.save_hyperparameters()

        sent_level_BERT_config = hydra.utils.instantiate(sent_level_BERT_config)

        self.word_level_bert = BERTWordLevel(
            sent_length=sent_length,
            hidden_size=hidden_size,
            batch_size=batch_size,
            pretrained_model=pretrained_model,
        )

        # TODO: sentレベルbigruをどうするか、、
        # TODO: configのなかに書く。
        self.sent_level_bigru = SentAttnNet(
            word_hidden_dim=hidden_size,
            sent_hidden_dim=hidden_size,
            # TODO: dropout率については後で直す。
            weight_drop=last_drop,
            # TODO: padding_idxはconfigから取り出した方が良さげ。→後で直す。
            padding_idx=0
        )

        self.classifier = Classifier(
            num_labels=num_labels,
            config=sent_level_BERT_config,
        )

        self.loss_fct = CrossEntropyLoss()

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

    def forward(self, input_ids: torch.FloatTensor, attention_mask: torch.LongTensor, pad_sent_num: torch.tensor, labels: torch.tensor):
        input_ids = input_ids.permute(1,0,2)
        attention_mask = attention_mask.permute(1,0,2)

        gen_torch_tensor_shape_assertion('input_ids', input_ids, (self.hparams.sent_length, self.hparams.batch_size, self.hparams.doc_length))
        gen_torch_tensor_shape_assertion('attention_mask', attention_mask, (self.hparams.sent_length, self.hparams.batch_size, self.hparams.doc_length))

        last_hidden_state_word_level, pooler_output_word_level = [], []
        for _input_ids, _attention_mask in zip(input_ids, attention_mask):
            _last_hidden_state, _pooler_output = self.word_level_bert(input_ids=_input_ids, attention_mask=_attention_mask).values()
            last_hidden_state_word_level.append(_last_hidden_state)
            pooler_output_word_level.append(_pooler_output)

        # TODO: sentenceAttenにBERTを使うか、GRUを使うか
        inputs_embeds = torch.stack(pooler_output_word_level).permute(1, 0, 2)
        pooled_output, attentions = self.sent_level_bert(inputs_embeds=inputs_embeds, pad_sent_num=pad_sent_num)
        loss, logits = self.classifier(pooled_output, labels)

        return loss, logits, attentions

    # TODO: ↓をちゃんと書く。
    def training_step(self, batch, batch_idx):
        # loss, logits, attentions = self(batch['input_ids'], batch['attention_mask'], batch['labels'])
        loss, logits, attentions = self(**batch)
        return {'loss': loss, 'batch_preds': logits, 'batch_labels': batch['labels']}

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
        loss, logits, attentions = self(**batch)
        return {'loss': loss, 'batch_preds': logits, 'batch_labels': batch['labels']}

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
        loss, logits, attentions = self(**batch)
        return {'loss': loss.detach(), 'batch_preds': logits.detach(), 'batch_labels': batch['labels']}

    def test_step_end(self, outputs):
        output = self.test_metrics(outputs['batch_preds'], outputs['batch_labels'])
        self.cm(outputs['batch_preds'], outputs['batch_labels'])
        self.log_dict(output)

    def test_epoch_end(self, outputs):
        logits = torch.cat([x['batch_preds'] for x in outputs])
        labels = torch.cat([x['batch_labels'] for x in outputs])
        epoch_loss = self.loss_fct(logits, labels)
        self.log("test_loss", epoch_loss, logger=True)
        self.log_dict(self.test_metrics.compute(), logger=True)

    # TODO: attention検証用のpredictionも書く。
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        return super().predict_step(batch, batch_idx, dataloader_idx=dataloader_idx)

    def configure_optimizers(self):
        return hydra.utils.instantiate(self.hparams.optim.args, params=self.parameters())