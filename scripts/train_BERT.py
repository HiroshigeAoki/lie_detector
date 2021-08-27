# -*- coding: utf-8 -*-
import pytorch_lightning as pl
from pytorch_lightning.accelerators import accelerator
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.accelerators.gpu import GPUAccelerator
from pytorch_lightning.plugins import NativeMixedPrecisionPlugin
import pandas as pd
import glob
import argparse
from pathlib import Path

import sys, os
sys.path.append(os.pardir)

from model.BERT import Classifier
from model.BERTDataModule import CreateBERTDataModule


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
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    if args.model == 'tohoku_bert_large':
        pretrained_model = 'cl-tohoku/bert-large-japanese'
    elif args.model == 'tohoku_bert_base':
        pretrained_model = 'cl-tohoku/bert-base-japanese'

    pl.seed_everything(111)

    train_df, valid_df, test_df = read_data(args.balance)

    data_module = CreateBERTDataModule(train_df, valid_df, test_df, batch_size=350, pretrained_model=pretrained_model)

    if args.from_checkpoint:
        ckpt_file = glob.glob(os.path.join(checkpoints_dir, '*.ckpt'))
        model = Classifier.load_from_checkpoint(ckpt_file[0],
                                                n_classes=2,
                                                n_epochs=N_EPOCHS,
                                                pretrained_model=pretrained_model,
                                                accelerator='ddp'
                                                )
    else:
        model = Classifier(n_classes=2, n_epochs=N_EPOCHS, pretrained_model=pretrained_model)

    # 3 epochでval_lossが0.05減少しなければ学習をストップ
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.005,
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
    parser.add_argument('--from_checkpoint', action='store_true')
    args = parser.parse_args()

    main(args)