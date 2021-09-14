import os
from typing import Optional
import pytorch_lightning as pl
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertJapaneseTokenizer


class CreateBERTDataset(Dataset):
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
        self.train_ds = CreateBERTDataset(self.train_df, self.tokenizer, self.max_token_len)
        self.valid_ds = CreateBERTDataset(self.valid_df, self.tokenizer, self.max_token_len)
        self.test_ds = CreateBERTDataset(self.test_df, self.tokenizer, self.max_token_len)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=os.cpu_count())

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.batch_size, shuffle=False, num_workers=os.cpu_count())

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=os.cpu_count())