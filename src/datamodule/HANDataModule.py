from typing import Optional
import pytorch_lightning as pl
import pandas as pd
import torch
import os
from torch.utils.data import DataLoader, Dataset
from src.tokenizer.HANTokenizer import HANTokenizer


class CreateHANDataset(Dataset):
    def __init__(self, df: pd.DataFrame, batch_size: int, tokenizer: HANTokenizer, 
                is_scam_game_data: bool = False, is_murder_mystery_data: bool = False):
        self.df = df
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.is_scam_game_data = is_scam_game_data
        self.is_murder_mystery_data = is_murder_mystery_data

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        df_row = self.df.iloc[index]
        nested_utters = df_row["nested_utters"]["raw_nested_utters"]
        labels = df_row['labels']

        encoding, attention_mask, pad_sent_num = self.tokenizer.encode(nested_utters)
        assert encoding.shape == (self.tokenizer.doc_length, self.tokenizer.sent_length), f"encoding shape: {encoding.shape} is wrong."

        item = dict(
            nested_utters=encoding, labels=torch.tensor(labels), attention_mask=attention_mask, pad_sent_num=pad_sent_num,
        )
        if self.is_scam_game_data:
            item.update(
                dict(
                    channel_name=df_row["channel_name"], 
                    judge=df_row['judge'],
                    judge_reason=df_row[:,'judge_reason'],
                    lie=torch.tensor(list(map(int, df_row['lie']))),
                    suspicious=torch.tensor(list(map(int, df_row['suspicious']))), 
                    suspicious_reasons=list(map(str, df_row['suspicious_reasons']))
                )
            )
        elif self.is_murder_mystery_data:
            item.update(
                dict(
                    annotations=df_row["annotations"],
                    game_info=df_row["game_info"],
                )
            )
        return item


class CreateHANDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, tokenizer: HANTokenizer, batch_size: int, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.n_cpus = os.cpu_count()
        self.kwargs = kwargs

    def setup(self, stage: Optional[str] = None):
        # set train and valid dataset
        if stage == 'fit':
            self.train_ds = CreateHANDataset(
                pd.read_pickle(os.path.join(self.data_dir, "train.pkl")), batch_size=self.batch_size, tokenizer=self.tokenizer, **self.kwargs)
            self.valid_ds = CreateHANDataset(
                pd.read_pickle(os.path.join(self.data_dir, "valid.pkl")), batch_size=self.batch_size, tokenizer=self.tokenizer, **self.kwargs)
        # set test dataset
        if stage == 'test' or stage == 'predict' or stage is None:
            self.test_ds = CreateHANDataset(
                pd.read_pickle(os.path.join(self.data_dir, "test.pkl")), batch_size=self.batch_size, tokenizer=self.tokenizer, **self.kwargs)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.train_ds, batch_size=self.batch_size,
                    shuffle=True, num_workers=self.n_cpus)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.valid_ds, batch_size=self.batch_size,
                    shuffle=False, num_workers=self.n_cpus)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.test_ds, batch_size=self.batch_size,
                    shuffle=False, num_workers=self.n_cpus)

    def predict_dataloader(self):
        return self.test_dataloader()
