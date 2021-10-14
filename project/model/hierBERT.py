from typing import Tuple
import torch
from torch import nn
import pytorch_lightning as pl
from torch.nn.modules.loss import CrossEntropyLoss
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1, ConfusionMatrix
from transformers import BertModel, BertJapaneseTokenizer, BertConfig
from transformers.models.bert.modeling_bert import BertPreTrainedModel
import hydra


class HierBERT(pl.LightningModule):
    def __init__(
        self,
        max_sent_len: int,
        max_doc_len: int,
        hidden_size: int,
        batch_size: int,
        last_drop: int,
        num_labels: int,
        classifier_drop: float,
        pretrained_model: str,
        sent_level_BERT_config: BertConfig,
        ):
        super(HierBERT, self).__init__()
        self.save_hyperparameters()

        self.max_sent_len = max_sent_len
        self.max_doc_len = max_doc_len
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_labels = num_labels

        self.word_level_bert = BERTWordLevel(
            pretrained_model=pretrained_model,
        )

        self.sent_level_bert = BERTSentLevel(
            hidden_size=self.hidden_size,
            config=sent_level_BERT_config,
        )

        self.classifier = Classifier(
            config=sent_level_BERT_config
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

        gen_assertion('input_ids', input_ids, (self.max_sent_len, self.batch_size, self.max_doc_len))
        gen_assertion('attention_mask', attention_mask, (self.max_sent_len, self.batch_size, self.max_doc_len))

        last_hidden_state_word_level, pooler_output_word_level = [], []
        for _input_ids, _attention_mask in zip(input_ids, attention_mask):
            _last_hidden_state, _pooler_output = self.word_level_bert(input_ids=_input_ids, attention_mask=_attention_mask).values()
            last_hidden_state_word_level.append(_last_hidden_state)
            pooler_output_word_level.append(_pooler_output)

        # 今の所こんな感じ
        # TODO: poolerを使うか、last_hiddenstateの平均を使うか
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
        output = self.train_metrics(outputs['logits'], outputs['batch_labels'])
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
        output = self.valid_metrics(outputs['logits'], outputs['batch_labels'])
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
        output = self.test_metrics(outputs['logits'], outputs['batch_labels'])
        self.cm(outputs['logits'], outputs['batch_labels'])
        self.log_dict(output)

    def test_epoch_end(self, outputs):
        logits = torch.cat([x['batch_preds'] for x in outputs])
        labels = torch.cat([x['batch_labels'] for x in outputs])
        epoch_loss = self.loss_fct(logits, labels)
        self.log("test_loss", epoch_loss, logger=True)
        self.log_dict(self.test_metrics.compute(), logger=True)

    # TODO: attention検証用のpredictionも書く。

    def configure_optimizers(self):
        return hydra.utils.instantiate(self.hparams.optim, params=self.parameters())


class BERTWordLevel(nn.Module):
    def __init__(self,
        max_sent_len: int,
        hidden_size: int,
        batch_size: int,
        pretrained_model: str = 'cl-tohoku/bert-large-japanese',
        ):
        super(BERTWordLevel, self).__init__()

        self.max_sent_len = max_sent_len
        self.hidden_size = hidden_size
        self.batch_size = batch_size

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
        gen_assertion(input_ids, input_ids.shape, (self.batch_size, self.hidden_size))
        gen_assertion('attention_mask', attention_mask, (self.batch_size, self.hidden_size))
        last_hidden_state, pooled_output = self.bert(input_ids, attention_mask)
        return last_hidden_state, pooled_output


class BERTSentLevel(nn.Module):
    def __init__(self,
        max_doc_len: int,
        hidden_size: int,
        batch_size: int,
        config: BertConfig,
        use_ave_pooler_output: bool, # CLS or average
        ):
        super(BERTSentLevel, self).__init__()
        self.max_doc_len = max_doc_len
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.use_ave_pooler_output = use_ave_pooler_output

        self.bert = BertModel(config)

    def forward(self, inputs_embeds: torch.FloatTensor,  pad_sent_num: torch.tensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        # TODO: 諸々決まったら、返り値型・数宣言の変更をする。
        gen_assertion('inputs_embeds', inputs_embeds, (self.batch_size, self.max_doc_len, self.hidden_size))
        attention_mask = torch.ones(self.batch_size, self.max_doc_len)
        # apply masks to pad sentences.
        attention_mask[-pad_sent_num:] = torch.zeros(pad_sent_num)
        last_hidden_state, pooled_output, attentions = self.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        if self.use_ave_pooler_output:
            pooled_output = last_hidden_state.mean(dim=1)
        gen_assertion('pooled_output', pooled_output, (self.batch_size, self.hidden_size))
        # TODO: attentionsの平均を取りたいけど、ミスるの怖いから、notebookで実験してからやる。↓で平均を取る。
        gen_assertion('attention', attentions, (self.batch_size, self.config.num_head, self.max_doc_len, self.max_doc_len))
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


def gen_assertion(target_name: str, target: torch.tensor, expected: Tuple) -> str:
    assert target.shape == expected, f'The shape of {target_name} is abnormal. {target_name}.shape:{target.shape}, expected:{expected}'