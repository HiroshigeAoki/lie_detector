import pytorch_lightning as pl
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1, ConfusionMatrix
# from sklearn.metrics import precision_recall_fscore_support
import hydra
import os
import torch
import pandas as pd
import numpy as np
import json
from src.utils.logger import OperationEndNotifier


class AbstractModel(pl.LightningModule):
    def __init__(self, optim: dict, use_gmail_notification: bool = False, is_scam_game: bool = False, is_murder_mystery: bool = False):
        super().__init__()
        self.save_hyperparameters()
        
        self.notifier = None
        if use_gmail_notification:
            self.notifier = OperationEndNotifier()

        metrics = MetricCollection([
            Accuracy(num_classes=2, average='macro'),
            Precision(num_classes=2, average='macro'),
            Recall(num_classes=2, average='macro'),
            F1(num_classes=2, average='macro')
        ])
        
        metrics_80 = MetricCollection([
            Accuracy(num_classes=2, average='macro', threshold=0.8),
            Precision(num_classes=2, average='macro', threshold=0.8),
            Recall(num_classes=2, average='macro', threshold=0.8),
            F1(num_classes=2, average='macro', threshold=0.8)
        ])
        
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='valid_')
        self.test_metrics = metrics.clone(prefix='test_50_')
        self.test_metrics_80 = metrics_80.clone(prefix='test_80_')

        self.cm = ConfusionMatrix(num_classes=2)
        self.cm_80 = ConfusionMatrix(num_classes=2, threshold=0.8)
        
        self.is_scam_game = is_scam_game
        self.is_murder_mystery = is_murder_mystery

    def training_step(self, batch, batch_idx):
        try:
            outputs = self(**batch)
            return dict(loss=outputs['loss'], batch_preds=outputs['preds'], batch_labels=batch['labels'])
        except Exception as e:
            if self.notifier:
                self.notifier.notify(operation='training', status='failed', message=str(e))

    def training_step_end(self, outputs):
        output = self.train_metrics(outputs['batch_preds'], outputs['batch_labels'])
        self.log_dict(output)

    def training_epoch_end(self, outputs):
        epoch_preds = torch.cat([x['batch_preds'] for x in outputs])
        epoch_labels = torch.cat([x['batch_labels'] for x in outputs])
        epoch_loss = self.criterion(epoch_preds, epoch_labels)
        self.log("train_loss", epoch_loss, logger=True)
        self.log_dict(self.train_metrics.compute(), logger=True)
        
        if self.notifier:
            self.notifier.notify(operation='training', status='success', message='training finished')

    def validation_step(self, batch, batch_idx):
        try:
            outputs = self(**batch)
            return dict(loss=outputs['loss'], batch_preds=outputs['preds'], batch_labels=batch['labels'])
        
        except Exception as e:
            if self.notifier:
                self.notifier.notify(operation='validation', status='failed', message=str(e))

    def validation_step_end(self, outputs):
        output = self.valid_metrics(outputs['batch_preds'], outputs['batch_labels'])
        self.log_dict(output)

    def validation_epoch_end(self, outputs):
        epoch_preds = torch.cat([x['batch_preds'] for x in outputs])
        epoch_labels = torch.cat([x['batch_labels'] for x in outputs])
        epoch_loss = self.criterion(epoch_preds, epoch_labels)
        self.log("val_loss", epoch_loss, logger=True)
        self.log_dict(self.valid_metrics.compute(), logger=True)
        
        if self.notifier:
           self.notifier.notify(operation='validation', status='success', message='validation finished')

    def test_step(self, batch, batch_idx):
        try:
            outputs = self(**batch)
            return_dict = dict(loss=outputs['loss'], batch_preds=outputs['preds'], batch_labels=batch['labels'], )
            #if self.is_murder_mystery:
            #    return_dict.update(
            #        dict(
            #            pad_sent_num=batch['pad_sent_num'], annotations=batch["annotations"], game_info=batch["game_info"],
            #            word_attentions=outputs['word_attentions'], sent_attentions=outputs['sent_attentions'],
            #        )
            #    )
            return return_dict
            
        except Exception as e:
            if self.notifier:
                self.notifier.notify(operation='testing', status='failed', message=str(e))

    def test_step_end(self, outputs):
        preds_softmax = torch.nn.functional.softmax(outputs['batch_preds'], dim=-1)
        preds_80, _ = torch.max(preds_softmax, dim=-1)
        preds_80 = (preds_80 > 0.8).long()
        metrics = self.test_metrics(outputs['batch_preds'].argmax(dim=1), outputs['batch_labels'])
        metrics_80 = self.test_metrics_80(preds_80, outputs['batch_labels'])
        self.cm(outputs['batch_preds'].argmax(dim=1), outputs['batch_labels'])
        self.cm_80(preds_80, outputs['batch_labels'])
        self.log_dict(metrics)
        self.log_dict({f"{k}_80": v for k, v in metrics_80.items()})
        
        # if self.is_murder_mystery:
        #     weighted_sent_labels = torch.tensor([self.label_weighted_attention_sents(sent_attentions, pad_sent_num) for sent_attentions, pad_sent_num in zip(outputs['sent_attentions'], outputs['pad_sent_num'])]).to(self.device)
        #     outputs['weighted_sent_labels'] = weighted_sent_labels
        #     return outputs

    def label_weighted_attention_sents(self, sent_attentions, pad_sent_num: int):
        threshold = 1 / (len(sent_attentions) - pad_sent_num)
        return [1 if sent_att > threshold else 0 for sent_att in sent_attentions][:-pad_sent_num]

    def test_epoch_end(self, outputs):
        try:
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
            
            preds_softmax = torch.nn.functional.softmax(preds, dim=-1)
            df = pd.DataFrame(preds_softmax.cpu().numpy(), columns=["confidence 0", "confidence 1"])
            df["true label"] = labels.cpu().numpy()
            df.to_csv("confidence.csv")
            
            # precision, recall, f1, support per class
            # scores_df = pd.DataFrame(
            #     np.array(precision_recall_fscore_support(labels.cpu(), preds.argmax(dim=1).cpu())).T,
            #     columns=["precision", "recall", "f1", "support"])
            # scores_df.to_csv(os.path.join(self.logger.log_dir, 'precision_recall_fscore_support.csv'))
            # self.print(f"f1_precision_accuracy\n{scores_df.to_string()}")
            # softmax = torch.nn.Softmax(dim=1)
            # probs = softmax(preds).cpu().numpy()
            # preds_80 = (probs[:, 1] > 0.8).astype(int)
            # scores_df_80 = pd.DataFrame(
            #     np.array(precision_recall_fscore_support(labels.cpu(), preds_80)).T,
            #     columns=["precision", "recall", "f1", "support"]
            # )
            # scores_df_80.to_csv(os.path.join(self.logger.log_dir, 'precision_recall_fscore_support.csv'))
            # self.print(f"f1_precision_accuracy\n{scores_df_80.to_string()}")

            # if self.is_murder_mystery:
            #     pred_labels = np.argmax(probs, axis=-1).tolist()
            #     confidences = list(map(lambda x: round(x, 2), np.max(probs, axis=-1).tolist()))
            #     channel_set =  set()
            #     with open(os.path.join(self.logger.log_dir, 'results.jsonl'), 'w', encoding='utf-8') as f:
            #         for i, output in enumerate(outputs):
            #             label = int(labels[i])
            #             channel_name = output['channel_name'][0]
            #             if label==1 and channel_name not in channel_set:
            #                 json.dump(dict(
            #                     channel_name=output['channel_name'][0],
            #                     label=int(labels[i]),
            #                     pred_label=pred_labels[i],
            #                     confidence=confidences[i],
            #                     judge=output['judge'][0].tolist(),
            #                     judge_reason=output['judge_reason'][0],
            #                     lie=output['lie'][0].tolist(),
            #                     suspicious=output['suspicious'][0].tolist(),
            #                     suspicious_reason=output['suspicious_reasons'],
            #                     weighted_sent_labels=output['weighted_sent_labels'][0].tolist(),
            #                 ), f, ensure_ascii=False)
            #                 f.write('\n')
            #                 channel_set.add(channel_name)

            if self.notifier:
                self.notifier.notify(operation='testing', status='success', message='testing finished')
        
        except Exception as e:
            if self.notifier:
                self.notifier.notify(operation='saving results', status='failed', message=str(e))

    def configure_optimizers(self):
        return hydra.utils.instantiate(self.hparams.optim.args, params=self.parameters())

    def predict_step(self, batch, batch_idx):
        outputs = self(batch['nested_utters'], batch['labels'], batch['attention_mask'], batch['pad_sent_num'])
        return dict(input_ids=batch['nested_utters'], labels=batch['labels'], pad_sent_num=batch['pad_sent_num'], loss=outputs['loss'], logits=outputs['preds'], word_attentions=outputs['word_attentions'], sent_attentions=outputs['sent_attentions'])
