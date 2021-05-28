# -*- coding: utf-8 -*-
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
import pandas as pd
import argparse
from pathlib import Path

import sys, os
sys.path.append(os.pardir)
from utils.unix_command import mkdirs
from model.BERT import Classifier
from model.DataModule import CreateBERTDataModule


def read_data(balance: bool):
    if balance:
        dir = Path('./data/flat/balance/')
    else:
        dir = Path('./data/flat/unbalance/')
    train_df = pd.read_pickle(dir / 'train.pkl')
    valid_df = pd.read_pickle(dir / 'valid.pkl')
    test_df = pd.read_pickle(dir / 'test.pkl')
    return train_df, valid_df, test_df


def main(args):
    N_EPOCHS = 20

    checkpoints_dir = f"./checkpoints/{args.model}/{'balance' if args.balance else 'unbalance'}"
    log_dir = f"./lightning_logs/{args.model}/{'balance' if args.balance else 'unbalance'}"
    mkdirs(checkpoints_dir)
    mkdirs(log_dir)

    if args.model == 'tohoku_bert_large':
        pretrained_model = 'cl-tohoku/bert-large-japanese'
    elif args.model == 'tohoku_bert_base':
        pretrained_model = 'cl-tohoku/bert-base-japanese'

    pl.seed_everything(111)

    train_df, valid_df, test_df = read_data(args.balance)

    data_module = CreateBERTDataModule(train_df, valid_df, test_df, batch_size=350, pretrained_model=pretrained_model)

    model = Classifier(n_classes=2, n_epochs=N_EPOCHS, pretrained_model=pretrained_model)

    # 3 epochでval_lossが0.05減少しなければ学習をストップ
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.05,
        patience=3,
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename='{epoch}',
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_dir)

    trainer = pl.Trainer(max_epochs=N_EPOCHS,
                            gpus=args.gpus,
                            precision=16,
                            progress_bar_refresh_rate=10,
                            callbacks=[checkpoint_callback, early_stop_callback],
                            logger=tb_logger
    )

    trainer.fit(model=model, datamodule=data_module)

    trainer.test(ckpt_path=checkpoint_callback.best_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--balance', action='store_true')
    parser.add_argument('--gpus', type=str, default="6")
    parser.add_argument('--model', type=str, default='tohoku_bert_large')
    args = parser.parse_args()

    main(args)