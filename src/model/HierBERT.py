from typing import Tuple
from omegaconf import DictConfig
import torch
from torch import nn
import pandas as pd
import pytorch_lightning as pl
from torch.nn.modules.loss import CrossEntropyLoss
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1, ConfusionMatrix
from transformers import BertModel, BertJapaneseTokenizer, BertConfig
from transformers.models.bert.modeling_bert import BertPreTrainedModel
import hydra

from src.utils.gen_assertion import gen_torch_tensor_shape_assertion

class HierchicalBERT(pl.LightningModule):
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
        super(HierchicalBERT, self).__init__()
        self.save_hyperparameters()

        if isinstance(sent_level_BERT_config, DictConfig):
            sent_level_BERT_config = hydra.utils.instantiate(sent_level_BERT_config)

        self.word_level_bert = BERTWordLevel(
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

        last_hidden_state_word_level, pooled_output_word_level, word_attentions = [], [], []
        for _input_ids, _attention_mask in zip(input_ids, attention_mask):
            outputs = self.word_level_bert(input_ids=_input_ids, attention_mask=_attention_mask)
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

    """
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
    """
    def predict_step(self, batch, batch_idx: int):
        outputs = self(**batch)
        return dict(loss=outputs['loss'], logits=outputs['logits'], word_attentions=outputs['word_attentions'], sent_attentions=outputs['sent_attentions'], input_ids=batch['input_ids'], labels=batch['labels'])

    def configure_optimizers(self):
        return hydra.utils.instantiate(self.hparams.optim.optimizer, params=self.parameters())


class BERTWordLevel(pl.LightningModule):
    def __init__(self,
        output_attentions: bool,
        pretrained_model: str,
        is_japanese: bool=True,
        ):
        super(BERTWordLevel, self).__init__()
        self.save_hyperparameters()

        self.bert = BertModel.from_pretrained(pretrained_model)
        if is_japanese:
            tokenizer = BertJapaneseTokenizer.from_pretrained(pretrained_model, additional_special_tokens=['<person>'])
            self.bert.resize_token_embeddings(len(tokenizer))
            # initialize  <person> token by the average of some personal_pronouns's weights.
            personal_pronouns = ['君', 'きみ', 'あなた' ,'彼', '彼女']
            personal_pronouns_weights = torch.stack([self.bert.embeddings.word_embeddings.weight[i, :] for i in tokenizer.convert_tokens_to_ids(personal_pronouns)])
            self.bert.embeddings.word_embeddings.weight.data[-1, :] = personal_pronouns_weights.mean(dim=0)

        # won't update word level bert layers
        for param in self.bert.parameters():
            param.requires_grad = False


    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        last_hidden_state, pooled_output, attentions = self.bert(input_ids, attention_mask, output_attentions=self.hparams.output_attentions).values()
        return dict(last_hidden_state=last_hidden_state, pooled_output=pooled_output, attentions=average_attention(attentions))


class BERTSentLevel(pl.LightningModule):
    def __init__(self,
        config: BertConfig,
        output_attentions: bool,
        use_ave_pooled_output: bool, # CLS or average
        ):
        super(BERTSentLevel, self).__init__()
        self.save_hyperparameters()

        self.bert = BertModel(config)

    def forward(self, inputs_embeds: torch.FloatTensor,  pad_sent_num: torch.tensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        # TODO: 諸々決まったら、返り値型・数宣言の変更をする。
        attention_mask = torch.ones(inputs_embeds.shape[0], inputs_embeds.shape[1])
        # apply masks to pad sentences.
        for i, num in enumerate(pad_sent_num):
            if num != 0:
                attention_mask[i, -num:] = torch.zeros(num)

        last_hidden_state, pooled_output, attentions = self.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask.to(self.device), output_attentions=self.hparams.output_attentions).values()
        if self.hparams.use_ave_pooled_output:
            pooled_output = last_hidden_state.mean(dim=1)
        return dict(pooled_output=pooled_output, attentions=average_attention(attentions))


class Classifier(BertPreTrainedModel):
    def __init__(self, num_labels, config):
        super().__init__(config)
        self.num_labels = num_labels
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

def average_attention(attentions):
    return torch.stack(attentions).mean(3).mean(2).mean(0)