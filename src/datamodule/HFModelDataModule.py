from typing import Optional
import pytorch_lightning as pl
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import os
import hydra
from omegaconf import DictConfig
from typing import Literal
from src.tokenizer.HFTokenizer import HFTokenizer


# Huggingfaceモデル共通のdatamodule
class CreateDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        batch_size: int,
        tokenizer: HFTokenizer,
        data_type: Literal["nested", "flat"],
    ):
        self.df = df
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.data_type = data_type

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        df_row = self.df.loc[:, "nested_utters"].iloc[index]
        nested_utters = df_row["raw_nested_utters"].tolist()
        labels = self.df.loc[:, "labels"].iloc[index]

        if self.data_type == "nested":
            input_ids, attention_mask, pad_sent_num = (
                self.tokenizer.batch_encode_nested(
                    nested_utters,
                    padding="max_length",
                    return_tensors="pt",
                    truncation=True,
                )
            )
            return dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=torch.tensor(labels),
                pad_sent_num=pad_sent_num,
                indices=index,
            )

        elif self.data_type == "flat":
            encodes = self.tokenizer.encode_flat(nested_utters)
            return dict(labels=torch.tensor(labels), indices=index, **encodes)

        else:
            raise ValueError(f"data_type:{self.data_type} is invalid.")


class CreateHFModelDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        tokenizer,
        data_type: str = "nested",
    ):
        super().__init__()
        self.n_cpus = os.cpu_count()

        if isinstance(tokenizer, DictConfig):
            self.tokenizer = hydra.utils.instantiate(tokenizer)
        else:
            self.tokenizer = tokenizer

        self.n_cpus = 8

        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        # set train and valid dataset
        if stage == "fit":
            self.train_ds = CreateDataset(
                pd.read_pickle(os.path.join(self.hparams.data_dir, "train.pkl")),
                self.hparams.batch_size,
                self.tokenizer,
                data_type=self.hparams.data_type,
            )
            self.valid_ds = CreateDataset(
                pd.read_pickle(os.path.join(self.hparams.data_dir, "valid.pkl")),
                self.hparams.batch_size,
                self.tokenizer,
                data_type=self.hparams.data_type,
            )
        # set test dataset
        if stage == "test" or stage == "predict" or stage is None:
            self.test_ds = CreateDataset(
                pd.read_pickle(os.path.join(self.hparams.data_dir, "test.pkl")),
                self.hparams.batch_size,
                self.tokenizer,
                data_type=self.hparams.data_type,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.n_cpus,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.valid_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.n_cpus,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.n_cpus,
            collate_fn=self.collate_fn,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    def collate_fn(self, batch) -> dict:
        batch.sort(key=lambda x: len(x["input_ids"]), reverse=True)
        input_ids = pad_sequence(
            [item["input_ids"] for item in batch],
            batch_first=True,
            padding_value=self.tokenizer.pad_index,
        )
        attention_mask = pad_sequence(
            [item["attention_mask"] for item in batch],
            batch_first=True,
            padding_value=self.tokenizer.pad_index,
        )
        labels = torch.stack([item["labels"] for item in batch])
        indices = batch[0]["indices"]

        if self.hparams.data_type == "nested":
            pad_sent_num = [item["pad_sent_num"] for item in batch]
            return dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                pad_sent_num=pad_sent_num,
                indices=indices,
            )
        elif self.hparams.data_type == "flat":
            return dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                indices=indices,
            )
