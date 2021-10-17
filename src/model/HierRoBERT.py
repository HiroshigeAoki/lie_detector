from typing import Tuple
from omegaconf import DictConfig
import torch
from torch import nn
import pandas as pd
import pytorch_lightning as pl
from torch.nn.modules.loss import CrossEntropyLoss
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1, ConfusionMatrix
from transformers import RobertaModel, CamembertTokenizer, BertConfig
import hydra

from src.model.HierBERT import BERTSentLevel, Classifier


class HierchicalRoBERT(pl.LightningModule):
    def __init__(
        self,
        num_labels: int,
        optim,
        pretrained_model: str,
        sent_level_BERT_config: BertConfig,
        output_attentions: bool,
        use_ave_pooled_output: bool,
        is_japanese: bool=True,
        # TODO: dropout率については、後で考える。
        ):
        super(HierchicalRoBERT, self).__init__()
        self.save_hyperparameters()

        if isinstance(sent_level_BERT_config, DictConfig):
            sent_level_BERT_config = hydra.utils.instantiate(sent_level_BERT_config)

        self.word_level_roberta = RoBERTaWordLevel(
            output_attentions=output_attentions,
            pretrained_model=pretrained_model,
            is_japanese=is_japanese,
        )

        self.sent_level_bert = BERTSentLevel(
            config=sent_level_BERT_config,
            output_attentions=output_attentions,
            use_ave_pooled_output=use_ave_pooled_output,
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
        self.test_metrics = metrics.clone(prefix='test_')

        self.cm = ConfusionMatrix(num_classes=2, compute_on_step=False)

    def forward(self, input_ids: torch.FloatTensor, attention_mask: torch.LongTensor, pad_sent_num: torch.tensor, labels: torch.tensor):
        input_ids = input_ids.permute(1,0,2)
        attention_mask = attention_mask.permute(1,0,2)

        # gen_torch_tensor_shape_assertion('input_ids', input_ids, (self.hparams.sent_length, input_ids.shape[1], self.hparams.doc_length))
        # gen_torch_tensor_shape_assertion('attention_mask', attention_mask, (self.hparams.sent_length, input_ids.shape[1], self.hparams.doc_length))

        last_hidden_state_word_level, pooled_output_word_level, word_attentions = [], [], []
        for _input_ids, _attention_mask in zip(input_ids, attention_mask):
            outputs = self.word_level_roberta(input_ids=_input_ids, attention_mask=_attention_mask)
            last_hidden_state_word_level.append(outputs['last_hidden_state'])
            pooled_output_word_level.append(outputs['pooled_output'])
            if self.hparams.output_attentions:
                word_attentions.append(outputs['attentions'])
        inputs_embeds = torch.stack(pooled_output_word_level).permute(1, 0, 2)
        outputs = self.sent_level_bert(inputs_embeds=inputs_embeds, pad_sent_num=pad_sent_num)
        loss, logits = self.classifier(outputs['pooled_output'], labels)
        word_attentions = word_attentions if self.hparams.output_attentions else None
        sent_attentions = outputs['attentions'] if self.hparams.output_attentions else None
        return dict(loss=loss, logits=logits, word_attentions=word_attentions, sent_attentions=sent_attentions)

    def training_step(self, batch, batch_idx):
        # loss, logits, attentions = self(batch['input_ids'], batch['attention_mask'], batch['labels'])
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
        self.log("test_loss", epoch_loss, logger=True)
        cm = self.cm.compute()
        test_metrix = self.test_metrics.compute()
        self.log("ConfusionMatrix", cm, logger=True)
        self.log_dict(test_metrix, logger=True)
        pd.DataFrame(cm.cpu().numpy()).to_csv(f'{self.logger.log_dir}/confusionmatrix.csv')
        pd.DataFrame([metrix.cpu().numpy() for metrix in test_metrix.values()], index=test_metrix.keys()).to_csv(f'{self.logger.log_dir}/scores.csv')

    def predict_step(self, batch, batch_idx: int):
        return self(**batch)

    def configure_optimizers(self):
        return hydra.utils.instantiate(self.hparams.optim.optimizer, params=self.parameters())


class RoBERTaWordLevel(pl.LightningModule):
    def __init__(self,
        output_attentions: bool,
        pretrained_model: str = 'itsunoda/wolfbbsRoBERTa-large',
        is_japanese: bool=True,
        ):
        super(RoBERTaWordLevel, self).__init__()
        self.save_hyperparameters()

        self.roberta = RobertaModel.from_pretrained(pretrained_model)
        if is_japanese:
            tokenizer = CamembertTokenizer.from_pretrained(pretrained_model, additional_special_tokens=['<person>'])
            self.roberta.resize_token_embeddings(len(tokenizer))
            # initialize <person> token by the average of some personal_pronouns's weights.
            personal_pronouns = ['君', 'きみ', 'あなた' ,'彼', '彼女']
            personal_pronouns_weights = torch.stack([self.roberta.embeddings.word_embeddings.weight[i, :] for i in tokenizer.convert_tokens_to_ids(personal_pronouns)])
            self.roberta.embeddings.word_embeddings.weight.data[-1, :] = personal_pronouns_weights.mean(dim=0)

        # won't update word level roberta layers
        for param in self.roberta.parameters():
            param.requires_grad = False


    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        last_hidden_state, pooled_output, attentions = self.roberta(input_ids, attention_mask, output_attentions=self.hparams.output_attentions).values()
        return dict(last_hidden_state=last_hidden_state, pooled_output=pooled_output, attentions=average_attention(attentions))


def average_attention(attentions):
    return torch.stack(attentions).mean(3).mean(2).mean(0)