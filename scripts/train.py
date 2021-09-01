# -*- coding: utf-8 -*-
import pytorch_lightning as pl
from pytorch_lightning import plugins
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.plugins import DDPPlugin
import pandas as pd
import glob
import argparse
from pathlib import Path
import hydra
from hydra.utils import get_original_cwd
import os, sys
sys.path.append(os.pardir)
from model.HAN import HierAttnNet
from model.HANDataModule import CreateHANDataModule
from preprocess.tokenizer_HAN import HANtokenizer


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    model_cfg = cfg.model
    cfg = cfg.common

    print(f"reading train data...")
    train_df = pd.read_pickle(get_original_cwd().replace('scripts', 'model') + '/data/nested/train.pkl')
    print("done!")
    print(f"reading valid data...")
    valid_df = pd.read_pickle(get_original_cwd().replace('scripts', 'model') + '/data/nested/valid.pkl')
    print("done!")
    print(f"reading test data...")
    test_df = pd.read_pickle(get_original_cwd().replace('scripts', 'model') + '/data/nested/test.pkl')

    checkpoints_dir = get_original_cwd() + f"/checkpoints/{model_cfg.name}"
    log_dir = get_original_cwd() + f"/lightning_logs/{model_cfg.name}"
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    pl.seed_everything(111)

    if model_cfg.name == 'HAN':
        tokenizer = HANtokenizer(chache=get_original_cwd().replace('scripts', 'tokenizer') + f"dim_{cfg.embed_dim}/model_fasttext.vec",
                                    vocab_size=cfg.vocab_size,
                                    min_freq=cfg.min_freq,
                                    split_train_txt=get_original_cwd().replace('scripts', 'tokenizer') + 'split_train.txt'
                                )

        data_module = CreateHANDataModule(train_df, valid_df, test_df, batch_size=cfg.batch_size, tokenizer=tokenizer)

        model = HierAttnNet(vocab_size=cfg.vocab_size,
                                word_hidden_dim=cfg.word_hidden_dim,
                                sent_hidden_dim=cfg.sent_hidden_dim,
                                padding_idx=cfg.padding_idx,
                                embed_dim=cfg.embed_dim,
                                embedding_matrix=tokenizer.embedding_matrix,
                                num_class=cfg.num_class,
                                weight_drop=model_cfg.weight_drop,
                                locked_drop=model_cfg.locked_drop,
                                last_drop=model_cfg.last_drop,
                                lr=model_cfg.lr,
                                weight_decay=model_cfg.weight_decay
                            )

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

    trainer = pl.Trainer(max_epochs=cfg.N_EPOCHS,
                            gpus="6",
                            precision=16,
                            progress_bar_refresh_rate=10,
                            callbacks=[checkpoint_callback, early_stop_callback],
                            logger=tb_logger,
                            plugins=DDPPlugin(find_unused_parameters=False)
    )

    trainer.fit(model=model, datamodule=data_module)


if __name__ == "__main__":
    main()
