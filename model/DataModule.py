from math import trunc
import os
from typing import Optional
import pytorch_lightning as pl
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import TensorDataset
from transformers import BertJapaneseTokenizer


class CreateDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer, max_token_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_row = self.data.iloc[index]
        text = data_row['text']
        labels = data_row['label']

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return dict(text=text, input_ids=encoding['input_ids'].flatten(),
                        attention_mask=encoding['attention_mask'].flatten(),
                        labels=torch.tensor(labels)
                    )


class CreateBERTDataModule(pl.LightningDataModule):
    def __init__(self, train_df, valid_df, test_df, batch_size=16, max_token_len=512
                , pretrained_model='cl-tohoku/bert-large-japanese'):
        super().__init__()
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.max_token_len = max_token_len
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(pretrained_model)

    def setup(self, stage: Optional[str]):
        self.train_ds = CreateDataset(self.train_df, self.tokenizer, self.max_token_len)
        self.valid_ds = CreateDataset(self.valid_df, self.tokenizer, self.max_token_len)
        self.test_ds = CreateDataset(self.test_df, self.tokenizer, self.max_token_len)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=os.cpu_count())

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.batch_size, shuffle=False, num_workers=os.cpu_count())

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=os.cpu_count())


class HANDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: Path, batch_size: int = 64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.n_cpus = os.cpu_count()

    def setup(self, stage: Optional[str] = None):
        # set train and valid dataset
        if stage == 'fit' or stage is None:
            train_mtx = np.load(self.data_dir / "train/train.npz")
            valid_mtx = np.load(self.data_dir / "valid/valid.npz")
            self.train_ds = TensorDataset(
                torch.from_numpy(train_mtx['X_train']),
                torch.from_numpy(train_mtx['y_train']).long()
                )
            self.valid_ds = TensorDataset(
                torch.from_numpy(valid_mtx['X_valid']),
                torch.from_numpy(valid_mtx['y_valid']),
            )
        # set test dataset
        if stage == 'test' or stage is None:
            test_mtx = np.load(self.data_dir / "test/test.npz")
            self.test_ds = TensorDataset(
                torch.from_numpy(test_mtx['X_test']),
                torch.from_numpy(test_mtx['y_test']),
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.train_ds, batch_size=self.batch_size,
                    shuffle=True, num_workers=self.n_cpus, pin_memory=True)

    def valid_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.valid_ds, batch_size=self.batch_size,
                    shuffle=False, num_workers=self.n_cpus, pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.test_ds, batch_size=self.batch_size,
                    shuffle=False, num_workers=self.n_cpus, pin_memory=True)

