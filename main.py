# -*- coding: utf-8 -*-
import logging

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.plugins import DDPPlugin

import hydra
from omegaconf import OmegaConf, DictConfig
from project.utils.gmail_send import Gmailsender

logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore")


@hydra.main(config_path="config", config_name="defaults")
def main(cfg: DictConfig) -> None:
    gmail_sender = Gmailsender(subject="実行終了通知")

    pl.seed_everything(1234)
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    if cfg.model.name == 'HAN':
        tokenizer = hydra.utils.instantiate(
            cfg.model.tokenizer,
            data_dir=cfg.data.module.data_dir,
        )

        data_module = hydra.utils.instantiate(
            cfg.data.module,
            tokenizer=tokenizer,
        )

        model = hydra.utils.instantiate(
            cfg.model.general,
            embedding_matrix=tokenizer.embedding_matrix,
            _recursive_=False,
        )

    early_stop_callback = hydra.utils.instantiate(
        cfg.early_stopping,
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        #dirpath=checkpoints_dir,
        filename='{epoch}',
        verbose=True,
        monitor='val_loss',
        mode='min',
        save_top_k=1
    )

    tb_logger = pl_loggers.TensorBoardLogger(".", "", "", log_graph=True, default_hp_metric=False)

    trainer = pl.Trainer(
        **OmegaConf.to_container(cfg.trainer),
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=tb_logger,
        plugins=DDPPlugin(find_unused_parameters=True)
    )

    try:
        trainer.fit(model=model, datamodule=data_module)
        trainer.test(ckpt_path=checkpoint_callback.best_model_path)
    except:
        gmail_sender.send(body="error occurred in main.py")
    finally:
        gmail_sender.send(body="finished main.py")

if __name__ == "__main__":
    main()
