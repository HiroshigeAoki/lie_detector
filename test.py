# -*- coding: utf-8 -*-
import logging

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.core import datamodule
from pytorch_lightning.plugins import DDPPlugin

import hydra
from omegaconf import OmegaConf, DictConfig
from project.utils.gmail_send import Gmailsender

logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore")


@hydra.main(config_path="config", config_name="defaults")
def main(cfg: DictConfig) -> None:
    gmail_sender = Gmailsender(subject=f"Execution end notification (model:{cfg.model.name}, data:{cfg.data.name})")

    pl.seed_everything(1234)
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    if cfg.model.name == 'HAN':
        tokenizer = hydra.utils.instantiate(
            cfg.model.tokenizer,
            data_dir=cfg.data.dir,
        )

        data_module = hydra.utils.instantiate(
            cfg.model.data_module,
            data_dir=cfg.data.dir,
            tokenizer=tokenizer,
        )

        model = hydra.utils.instantiate(
            cfg.model.general,
            optim=cfg.optim,
            embedding_matrix=tokenizer.embedding_matrix,
            _recursive_=False,
        )

    elif cfg.model.name=='HierBERT':
        # TODO: 色々書く。
        data_module = hydra.utils.instantiate(
            cfg.model.data_module,
            data_dir=cfg.data.dir,
            _recursive_=False,
        )

        model = hydra.utils.instantiate(
            cfg.model.general,
            optim=cfg.optim,
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

    #TODO: sentence level がGRUのバージョンと分ける。
    # elif cfg.model.name=='HierBERT' and

    tb_logger = pl_loggers.TensorBoardLogger(".", "", "", log_graph=True, default_hp_metric=False)

    trainer = pl.Trainer(
        **OmegaConf.to_container(cfg.trainer),
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=tb_logger,
        plugins=DDPPlugin(find_unused_parameters=True)
    )
    # TODO: load from checkpoint 的なやつで、モデルをloadする。https://pytorch-lightning.readthedocs.io/en/latest/common/weights_loading.html
    try:
        trainer.predict(model, datamodule)
    except Exception as e:
        gmail_sender.send(body=f"error occurred in main.py.\n\n{type(e)}\n{e}")
    finally:
        gmail_sender.send(body=f"main.py finished.")

if __name__ == "__main__":
    main()