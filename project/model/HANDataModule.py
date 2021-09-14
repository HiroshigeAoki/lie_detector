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
                    tokenizer
                ):
        self.df = df
        self.batch_size = batch_size
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # encodingを行う。
        df_row = self.df.loc[:,'nested_utters'].iloc[index]
        nested_utters = df_row['raw_nested_utters']
        labels = self.df.loc[:,'labels'].iloc[index]

        encoding = self.tokenizer.encode(nested_utters)

        assert encoding.shape == (self.tokenizer.max_utter_num, self.tokenizer.max_mor_num), f"encoding shape: {encoding.shape} is wrong."

        return dict(nested_utters=encoding, labels=torch.tensor(labels))


class CreateHANDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, tokenizer, batch_size=16):
        super().__init__()
        self.train_df = pd.read_pickle(data_dir + "train.pkl")
        self.valid_df = pd.read_pickle(data_dir + "valid.pkl")
        self.test_df = pd.read_pickle(data_dir + "test.pkl")
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.n_cpus = os.cpu_count()

    def setup(self, stage: Optional[str] = None):
        # set train and valid dataset
        if stage == 'fit':
            self.train_ds = CreateHANDataset(self.train_df, self.batch_size, self.tokenizer)
            self.valid_ds = CreateHANDataset(self.valid_df, self.batch_size, self.tokenizer)
        # set test dataset
        if stage == 'test' or stage is None:
            self.test_ds = CreateHANDataset(self.test_df, self.batch_size, self.tokenizer)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.train_ds, batch_size=self.batch_size,
                    shuffle=True, num_workers=self.n_cpus)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.valid_ds, batch_size=self.batch_size,
                    shuffle=False, num_workers=self.n_cpus)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.test_ds, batch_size=self.batch_size,
                    shuffle=False, num_workers=self.n_cpus)