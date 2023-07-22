import pytorch_lightning as pl
from torchmetrics import Accuracy, MetricCollection, Precision, Recall, F1, ConfusionMatrix
from sklearn.metrics import precision_recall_fscore_support
import hydra
import os
import torch
import pandas as pd
import numpy as np


class AbstractModel(pl.LightningModule):
    def __init__(self, optim: dict):
        super().__init__()

        metrics = MetricCollection([
            Accuracy(num_classes=2, average='macro'),
            Precision(num_classes=2, average='macro'),
            Recall(num_classes=2, average='macro'),
            F1(num_classes=2, average='macro')
        ])
        
        metrics_80 = MetricCollection([
            Accuracy(num_classes=2, average='macro'),
            Precision(num_classes=2, average='macro'),
            Recall(num_classes=2, average='macro'),
            F1(num_classes=2, average='macro')
        ])

        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='valid_')
        self.test_metrics = metrics.clone(prefix='test_50_')
        self.test_metrics_80 = metrics_80.clone(prefix='test_80_')

        self.cm = ConfusionMatrix(num_classes=2, compute_on_step=False)
        self.cm_80 = ConfusionMatrix(num_classes=2, compute_on_step=False, threshold=0.8)
    

    def training_step(self, batch, batch_idx):
        outputs = self(batch['nested_utters'], batch['labels'], batch['attention_mask'], batch['pad_sent_num'])
        return dict(loss=outputs['loss'], batch_preds=outputs['preds'], batch_labels=batch['labels'])


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
        outputs = self(batch['nested_utters'], batch['labels'], batch['attention_mask'], batch['pad_sent_num'])
        return dict(loss=outputs['loss'], batch_preds=outputs['preds'], batch_labels=batch['labels'])


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
        outputs = self(batch['nested_utters'], batch['labels'], batch['attention_mask'], batch['pad_sent_num'])
        return dict(loss=outputs['loss'], batch_preds=outputs['preds'], batch_labels=batch['labels'])


    def test_step_end(self, outputs):
        preds_softmax = torch.nn.functional.softmax(outputs['batch_preds'], dim=-1)
        preds_thresholded = (preds_softmax[:, 1] > 0.8).long()
        output = self.test_metrics(outputs['batch_preds'].argmax(dim=1), outputs['batch_labels'])
        output_80 = self.test_metrics_80(preds_thresholded, outputs['batch_labels'])
        self.cm(outputs['batch_preds'].argmax(dim=1), outputs['batch_labels'])
        self.cm_80(preds_thresholded, outputs['batch_labels'])
        self.log_dict(output)
        self.log_dict({f"{k}_80": v for k, v in output_80.items()})


    def test_epoch_end(self, outputs):
        preds = torch.cat([x['batch_preds'] for x in outputs])
        labels = torch.cat([x['batch_labels'] for x in outputs])
    
        # loss
        epoch_loss = self.criterion(preds, labels)
        self.log("test_loss", epoch_loss, logger=True)
        
        # metrics
        test_metrics = self.test_metrics.compute()
        test_metrics_80 = self.test_metrics_80.compute()
        self.log_dict(test_metrics, logger=True)
        self.log_dict({f"{k}_80": v for k, v in test_metrics_80.items()}, logger=True)
        pd.DataFrame([metrics.cpu().numpy() for metrics in test_metrics.values()], index=test_metrics.keys()).to_csv(f'{self.logger.log_dir}/scores.csv')
        pd.DataFrame([metrics.cpu().numpy() for metrics in test_metrics_80.values()], index=test_metrics_80.keys()).to_csv(f'{self.logger.log_dir}/scores_80.csv')

        # confusion matrix
        cm = pd.DataFrame(self.cm.compute().cpu().numpy())
        cm_80 = pd.DataFrame(self.cm_80.compute().cpu().numpy())
        cm.to_csv(os.path.join(self.logger.log_dir, 'confusionmatrix.csv'))
        cm_80.to_csv(os.path.join(self.logger.log_dir, 'confusionmatrix_80.csv'))
        self.print(f"confusion_matrix\n{cm.to_string()}\n")
        self.print(f"confusion_matrix_80\n{cm_80.to_string()}\n")

        # precision, recall, f1, support per class
        scores_df = pd.DataFrame(
            np.array(precision_recall_fscore_support(labels.cpu(), preds.argmax(dim=1).cpu())).T,
            columns=["precision", "recall", "f1", "support"])
        scores_df.to_csv(os.path.join(self.logger.log_dir, 'precision_recall_fscore_support.csv'))
        self.print(f"f1_precision_accuracy\n{scores_df.to_string()}")
        softmax = torch.nn.Softmax(dim=1)
        probs = softmax(preds).cpu().numpy()
        preds_80 = (probs[:, 1] > 0.8).astype(int)
        scores_df_80 = pd.DataFrame(
            np.array(precision_recall_fscore_support(labels.cpu(), preds_80)).T,
            columns=["precision", "recall", "f1", "support"]
        )
        scores_df_80.to_csv(os.path.join(self.logger.log_dir, 'precision_recall_fscore_support.csv'))
        self.print(f"f1_precision_accuracy\n{scores_df_80.to_string()}")


    def configure_optimizers(self):
        return hydra.utils.instantiate(self.hparams.optim.args, params=self.parameters())


    def predict_step(self, batch, batch_idx):
        outputs = self(batch['nested_utters'], batch['labels'], batch['attention_mask'], batch['pad_sent_num'])
        return dict(input_ids=batch['nested_utters'], labels=batch['labels'], pad_sent_num=batch['pad_sent_num'], loss=outputs['loss'], logits=outputs['preds'], word_attentions=outputs['word_attentions'], sent_attentions=outputs['sent_attentions'])
