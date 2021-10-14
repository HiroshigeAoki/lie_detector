from typing import Optional
import pytorch_lightning as pl
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import os
import hydra

class CreateDataset(Dataset):
    def __init__(self, df: pd.DataFrame, batch_size: int, tokenizer):
        self.df = df
        self.batch_size = batch_size
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        df_row = self.df.loc[:,'nested_utters'].iloc[index]
        nested_utters = df_row['raw_nested_utters']
        labels = self.df.loc[:,'labels'].iloc[index]

        input_ids, attention_mask, pad_sent_num = self.tokenizer.encode(nested_utters)

        return dict(input_ids=input_ids, attention_mask=attention_mask, labels=torch.tensor(labels), pad_sent_num=torch.tensor(pad_sent_num))


class CreateHierBertDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, tokenizer):
        super().__init__()
        self.train_df = pd.read_pickle(data_dir + "train.pkl")
        self.valid_df = pd.read_pickle(data_dir + "valid.pkl")
        self.test_df = pd.read_pickle(data_dir + "test.pkl")

        self.tokenizer = hydra.utils.instantiate(tokenizer)

        self.n_cpus = os.cpu_count()

        self.save_hyperparameters() # TODO: check Datamoduleでも使えるはず？、、

    def setup(self, stage: Optional[str] = None):
        # set train and valid dataset
        if stage == 'fit':
            self.train_ds = CreateDataset(self.train_df, self.hparams.batch_size, self.tokenizer)
            self.valid_ds = CreateDataset(self.valid_df, self.hparams.batch_size, self.tokenizer)
        # set test dataset
        if stage == 'test' or stage is None:
            self.test_ds = CreateDataset(self.test_df, self.hparams.batch_size, self.tokenizer)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.train_ds, batch_size=self.hparams.batch_size,
                    shuffle=True, num_workers=self.n_cpus)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.valid_ds, batch_size=self.hparams.batch_size,
                    shuffle=False, num_workers=self.n_cpus)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.test_ds, batch_size=self.hparams.batch_size,
                    shuffle=False, num_workers=self.n_cpus)