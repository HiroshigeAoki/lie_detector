from typing import Optional
import pytorch_lightning as pl
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import os, sys
sys.path.append(os.pardir)


class CreateHANDataset(Dataset):
    def __init__(self,
                    df: pd.DataFrame,
                    batch_size: int,
                    max_word_len: int,
                    max_sent_len: int,
                    tokenizer
                ):
        self.df = df
        self.batch_size = batch_size
        self.max_word_len = max_word_len
        self.max_sent_len = max_sent_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # encodingを行う。
        df_row = self.df.loc[:,'nested_utters'].iloc[index]
        nested_utters = df_row['raw_nested_utters']
        labels = self.df.loc[:,'label'].iloc[index]

        encoding = self.tokenizer.encode(
            nested_utters, self.max_word_len, self.max_sent_len
        )

        return dict(nested_utters=encoding, labels=torch.tensor(labels))


class CreateHANDataModule(pl.LightningDataModule):
    def __init__(self, train_df, valid_df, test_df, tokenizer, \
                    batch_size=16, max_word_len=180, max_sent_len=180):
        super().__init__()
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.max_word_len = max_word_len
        self.max_sent_len = max_sent_len
        self.tokenizer = tokenizer
        self.n_cpus = os.cpu_count()

    def setup(self, stage: Optional[str] = None):
        # set train and valid dataset
        if stage == 'fit':
            self.train_ds = CreateHANDataset(self.train_df, self.batch_size, self.max_word_len, self.max_sent_len, self.tokenizer)
            self.valid_ds = CreateHANDataset(self.valid_df, self.batch_size, self.max_word_len, self.max_sent_len, self.tokenizer)
        # set test dataset
        if stage == 'test' or stage is None:
            self.test_ds = CreateHANDataset(self.test_df, self.batch_size, self.max_word_len, self.max_sent_len, self.tokenizer)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.train_ds, batch_size=self.batch_size,
                    shuffle=True, num_workers=self.n_cpus, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.valid_ds, batch_size=self.batch_size,
                    shuffle=False, num_workers=self.n_cpus, pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.test_ds, batch_size=self.batch_size,
                    shuffle=False, num_workers=self.n_cpus, pin_memory=True)