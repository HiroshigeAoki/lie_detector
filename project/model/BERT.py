import pytorch_lightning as pl
from transformers import BertModel
import torch
from torch import nn, optim
import torchmetrics
import torchmetrics
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


class Classifier(pl.LightningModule):
    def __init__(self, n_classes: int, n_epochs=None,
                pretrained_model='cl-tohoku/bert-large-japanese'):
        super().__init__()

        self.bert = BertModel.from_pretrained(
            pretrained_model, return_dict=True
        )
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.n_epochs = n_epochs
        self.criterion = nn.CrossEntropyLoss()

        for param in self.bert.parameters():
            param.requires_grad = False # 結果次第で変えてみる
        for param in self.bert.encoder.layer[-1].parameters():
            param.requires_grad = True # BERTの最終層のみ学習


    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        preds = self.classifier(output.pooler_output) # output.pooler_output: CLS token
        loss = 0
        if labels is not None:
            loss = self.criterion(preds, labels)
        return loss, preds


    def training_step(self, batch, batch_idx):
        loss, preds = self.forward(input_ids=batch['input_ids'],
                                    attention_mask=batch['attention_mask'],
                                    labels=batch['labels'])
        return {'loss': loss, 'batch_preds': preds, 'batch_labels': batch['labels']}


    def validation_step(self, batch, batch_idx):
        loss, preds = self.forward(input_ids=batch['input_ids'],
                                    attention_mask=batch['attention_mask'],
                                    labels=batch['labels'])
        return {'loss': loss, 'batch_preds': preds, 'batch_labels': batch['labels']}


    def test_step(self, batch, batch_idx):
        loss, preds = self.forward(input_ids=batch['input_ids'],
                                    attention_mask=batch['attention_mask'],
                                    labels=batch['labels'])
        return {'loss': loss, 'batch_preds': preds, 'batch_labels': batch['labels']}


    def validation_epoch_end(self, outputs):
        # loss
        epoch_preds = torch.cat([x['batch_preds'] for x in outputs])
        epoch_labels = torch.cat([x['batch_labels'] for x in outputs])
        epoch_loss = self.criterion(epoch_preds, epoch_labels)
        self.log("val_loss", epoch_loss, logger=True)

        # accuracy
        num_correct = (epoch_preds.argmax(dim=1) == epoch_labels).sum().item() # tensorから値に変換
        epoch_accuracy = num_correct / len(epoch_labels)
        self.log("val_accuracy", epoch_accuracy, logger=True)


    def test_epoch_end(self, outputs):
        preds = torch.cat([x['batch_preds'] for x in outputs])
        labels = torch.cat([x['batch_labels'] for x in outputs])

        # loss
        loss = self.criterion(preds, labels)
        self.log("test_loss", loss, logger=True)

        # accuracy
        num_correct = (preds.argmax(dim=1) == labels).sum().item() # tensorから値に変換
        epoch_accuracy = num_correct / len(labels)
        self.log("test_accuracy", epoch_accuracy, logger=True)

        # confusion matrix
        cm = torchmetrics.ConfusionMatrix(num_classes=2)
        df_cm = pd.DataFrame(cm(preds.argmax(dim=1).cpu(), labels.cpu()).numpy())
        self.print(f"confusion_matrix\n{df_cm.to_string()}\n")
        """
        plt.figure(figsize=(2,2))
        fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
        plt.close(fig_)
        self.logger.experiment.add_figure("Confusion matrix", fig_)
        """

        #f1 precision recall
        scores_df = pd.DataFrame(np.array(precision_recall_fscore_support(labels.cpu(), preds.argmax(dim=1).cpu())).T,
                                    columns=["precision", "recall", "f1", "support"],
                                )
        self.print(f"f1_precision_accuracy\n{scores_df.to_string()}")


    # 全結合層の学習率は高く、BERTの最終層の学習率は低く
    def configure_optimizers(self):
        optimizer = optim.Adam([
            {'params': self.bert.encoder.layer[-1].parameters(), 'lr': 5e-5},
            {'params': self.classifier.parameters(), 'lr': 1e-4}
        ])
        return [optimizer]