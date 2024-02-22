import pytorch_lightning as pl
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score, ConfusionMatrix
import hydra
import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from src.utils.logger import OperationEndNotifier
from captum.attr import IntegratedGradients


class AbstractModel(pl.LightningModule):
    def __init__(self, optim: dict, use_gmail_notification: bool = False, is_scam_game: bool = False, is_murder_mystery: bool = False):
        super().__init__()
        self.save_hyperparameters()
        
        self.test_step_outputs = dict(logits=[], labels=[], indices=[])
        
        self.notifier = None
        if use_gmail_notification:
            self.notifier = OperationEndNotifier()

        metrics = MetricCollection([
            Accuracy(task="binary", num_classes=2, average='macro'),
            Precision(task="binary", num_classes=2, average='macro'),
            Recall(task="binary", num_classes=2, average='macro'),
            F1Score(task="binary", num_classes=2, average='macro')
        ])
        
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='valid_')
        self.test_metrics = metrics.clone(prefix='test_')

        self.cm = ConfusionMatrix(task="binary", num_classes=2)
        
        self.is_scam_game = is_scam_game
        self.is_murder_mystery = is_murder_mystery

    def training_step(self, batch, batch_idx):
        try:            
            outputs = self(**batch)
            self.log("train_loss", outputs["loss"], prog_bar=True)
            self.log_dict(self.train_metrics(outputs["preds"].argmax(dim=1), batch["labels"]), sync_dist=True)
            return outputs["loss"]
        except Exception as e:
            if self.notifier:
                self.notifier.notify(operation='training', status='failed', message=str(e))
            raise e

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute(), logger=True, sync_dist=True)
        if self.notifier:
            self.notifier.notify(operation='training', status='success', message='training finished')

    def validation_step(self, batch, batch_idx):
        try:
            outputs = self(**batch)
            self.log("val_loss", outputs['loss'],  on_epoch=True, prog_bar=True, sync_dist=True)
            self.log_dict(self.valid_metrics(outputs["preds"].argmax(dim=1), batch["labels"]), sync_dist=True)
            return outputs["loss"]
        except Exception as e:
            if self.notifier:
                self.notifier.notify(operation='validation', status='failed', message=str(e))
            raise e

    def on_validation_epoch_end(self):
        self.log_dict(self.valid_metrics.compute(), logger=True, sync_dist=True)
        if self.notifier:
            self.notifier.notify(operation='validation', status='success', message='validation finished')

    def test_step(self, batch, batch_idx):
        try:
            outputs = self(**batch)
            preds = outputs['preds'].argmax(dim=1)
            self.log("test_loss", outputs['loss'], prog_bar=True)
            self.log_dict(self.test_metrics(preds, batch["labels"]), sync_dist=True)
            self.cm(preds, batch['labels'])
            self.test_step_outputs["logits"].extend(outputs['preds'].cpu())
            self.test_step_outputs["labels"].extend(batch["labels"].cpu())
            self.test_step_outputs["indices"].append(batch["indices"])
            return outputs["loss"]
        except Exception as e:
            if self.notifier:
                self.notifier.notify(operation='testing', status='failed', message=str(e))
            raise e

    def label_weighted_attention_sents(self, sent_attentions, pad_sent_num: int):
        threshold = 1 / (len(sent_attentions) - pad_sent_num)
        return [1 if sent_att > threshold else 0 for sent_att in sent_attentions][:-pad_sent_num]

    def on_test_epoch_end(self):
        try:
            # metrics
            test_metrics = self.test_metrics.compute()
            pd.DataFrame([metrics.cpu().numpy() for metrics in test_metrics.values()], index=test_metrics.keys()).to_csv(f'{self.logger.log_dir}/scores.csv')
            self.log_dict(test_metrics, logger=True, sync_dist=True)
            
            # confusion matrix
            cm = pd.DataFrame(self.cm.compute().cpu().numpy())
            cm.to_csv(os.path.join(self.logger.log_dir, 'confusionmatrix.csv'))
            self.print(f"confusion_matrix\n{cm.to_string()}\n")
            
            if self.notifier:
                self.notifier.notify(operation='testing', status='success', message='testing finished')
        
        except Exception as e:
            if self.notifier:
                self.notifier.notify(operation='saving results', status='failed', message=str(e))
    
    def on_test_end(self):
        logits = torch.stack(self.test_step_outputs["logits"]).to(torch.float32)
        labels = torch.stack(self.test_step_outputs["labels"])
        indices = self.test_step_outputs["indices"]

        all_logits = self.all_gather(logits)
        all_labels = self.all_gather(labels)
        all_logits = torch.cat(tuple(all_logits)).to("cpu")
        all_labels = torch.cat(tuple(all_labels)).to("cpu")
        
        if self.trainer.is_global_zero:  # Only on process 0
            all_preds = all_logits.argmax(dim=1)
            logits_softmax = torch.nn.functional.softmax(all_logits, dim=-1)
            df = pd.DataFrame(logits_softmax.numpy(), columns=["confidence 0", "confidence 1"])
            df["true label"] = all_labels.numpy()
            df["pred label"] = all_preds.numpy()
            df["indices"] = indices
            df = df.sort_values(by="indices")
            df.to_csv("confidence.csv")
            
            # precision, recall, f1, support per class
            scores_df = pd.DataFrame(
                np.array(precision_recall_fscore_support(all_labels, all_preds)).T,
                columns=["precision", "recall", "f1", "support"])
            scores_df.to_csv(os.path.join(self.logger.log_dir, 'precision_recall_fscore_support.csv'))
            self.print(f"f1_precision_accuracy\n{scores_df.to_string()}")

    def configure_optimizers(self):
        return hydra.utils.instantiate(self.hparams.optim.args, params=self.parameters())

    def predict_step(self, batch, batch_idx):
        attributions = self.compute_attributions(**batch)

    def compute_attributions(self, input_ids, attention_mask, labels):
        integrated_gradients = IntegratedGradients(self.captum_forward)

        # 属性の計算
        attributions = integrated_gradients.attribute(
            inputs=input_ids,
            baselines=torch.zeros_like(input_ids),
            target=labels,
        )
        return attributions
