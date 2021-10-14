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

from project.model.HAN import SentAttnNet
from project.utils.gen_assertion import gen_torch_tensor_shape_assertion

class HierchicalBERT(pl.LightningModule):
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
        # TODO: sent levelをbigruにしたモデルは、別プログラムに書いたほうが良いかも
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

        self.sent_level_bert = BERTSentLevel(
            doc_length=doc_length,
            hidden_size=hidden_size,
            batch_size=batch_size,
            hidden_size=hidden_size,
            config=sent_level_BERT_config,
            use_ave_pooler_output=use_ave_pooler_output,
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
        loss, logits, attentions = self(batch['input_ids'], batch['attention_mask'], batch['labels'])
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
        loss, logits, attentions = self(batch['input_ids'], batch['attention_mask'], batch['labels'])
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
        loss, logits, attentions = self(batch['input_ids'], batch['attention_mask'], batch['labels'])
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
        return hydra.utils.instantiate(self.hparams.optim, params=self.parameters())


class BERTWordLevel(pl.LightningModule):
    def __init__(self,
        sent_length: int,
        hidden_size: int,
        batch_size: int,
        pretrained_model: str = 'cl-tohoku/bert-large-japanese',
        ):
        super(BERTWordLevel, self).__init__()
        self.save_hyperparameters()

        self.bert = BertModel.from_pretrained(pretrained_model)
        tokenizer = BertJapaneseTokenizer(pretrained_model, additional_special_tokens=['<person>'])
        self.bert.resize_token_embeddings(len(tokenizer))
        # initialize <person> token by the average of some personal_pronouns's weights.
        personal_pronouns = ['君', 'きみ', 'あなた' ,'彼', '彼女']
        personal_pronouns_ids = tokenizer.convert_tokens_to_ids(personal_pronouns)
        personal_pronouns_weights = torch.stack([self.bert.embeddings.word_embeddings.weight[i, :] for i in personal_pronouns_ids])
        self.bert.embeddings.word_embeddings.weight[-1, :] = personal_pronouns_weights.mean(dim=0)
        # won't update word level bert layers
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        gen_torch_tensor_shape_assertion(input_ids, input_ids.shape, (self.hparams.batch_size, self.hparams.hidden_size))
        gen_torch_tensor_shape_assertion('attention_mask', attention_mask, (self.hparams.batch_size, self.hparams.hidden_size))
        last_hidden_state, pooled_output = self.bert(input_ids, attention_mask)
        return last_hidden_state, pooled_output


class BERTSentLevel(pl.LightningModule):
    def __init__(self,
        doc_length: int,
        hidden_size: int,
        batch_size: int,
        config: BertConfig,
        use_ave_pooler_output: bool, # CLS or average
        ):
        super(BERTSentLevel, self).__init__()
        self.save_hyperparameters()

        self.bert = BertModel(config)

    def forward(self, inputs_embeds: torch.FloatTensor,  pad_sent_num: torch.tensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        # TODO: 諸々決まったら、返り値型・数宣言の変更をする。
        gen_torch_tensor_shape_assertion('inputs_embeds', inputs_embeds, (self.hparams.batch_size, self.hparams.doc_length, self.hparams.hidden_size))
        attention_mask = torch.ones(self.hparams.batch_size, self.hparams.doc_length)
        # apply masks to pad sentences.
        attention_mask[-pad_sent_num:] = torch.zeros(pad_sent_num)
        last_hidden_state, pooled_output, attentions = self.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        if self.use_ave_pooler_output:
            pooled_output = last_hidden_state.mean(dim=1)
        gen_torch_tensor_shape_assertion('pooled_output', pooled_output, (self.hparams.batch_size, self.hparams.hidden_size))
        # TODO: attentionsの平均を取りたいけど、ミスるの怖いから、notebookで実験してからやる。↓で平均を取る。
        gen_torch_tensor_shape_assertion('attention', attentions, (self.hparams.batch_size, self.hparams.config.num_head, self.hparams.doc_length, self.hparams.doc_length))
        return pooled_output, attentions


class Classifier(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fct = CrossEntropyLoss()

        self.init_weights()

    def forward(self, pooled_output, labels):
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss, logits