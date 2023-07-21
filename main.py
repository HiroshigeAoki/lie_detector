import logging
import traceback

import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.plugins import DDPPlugin

import hydra
from omegaconf import OmegaConf, DictConfig

from src.visualization.plot_attention import create_html
import os

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="defaults")
def main(cfg: DictConfig) -> None:
    try:
        # gmail_sender = Gmailsender(subject=f"Execution end notification (model:{cfg.model.name}, data:{cfg.data.name})")

        pl.seed_everything(1234)
        logger.info("\n" + OmegaConf.to_yaml(cfg))

        cfg.data.dir = os.path.join(cfg.workplace_dir, cfg.data.dir)

        """instantiate"""
        if cfg.model.name == 'HAN':
            cfg.tokenizer.args.cache_dir = os.path.join(
                cfg.workplace_dir, cfg.tokenizer.args.cache_dir)

            if cfg.data.name == 'nested_sample':
                cfg.tokenizer.args.cache_dir = os.path.join(
                    cfg.tokenizer.args.cache_dir, 'sample')

            if cfg.tokenizer.name == 'sentencepiece':
                cfg.tokenizer.args.model_file = os.path.join(
                    cfg.workplace_dir, cfg.tokenizer.args.model_file)

            tokenizer = hydra.utils.instantiate(
                cfg.tokenizer.args,
                data_dir=cfg.data.dir,
            )

            data_module = hydra.utils.instantiate(
                cfg.model.data_module,
                data_dir=cfg.data.dir,
                tokenizer=tokenizer,
                mode=cfg.mode,
            )

            model = hydra.utils.instantiate(
                cfg.model.args,
                optim=cfg.optim,
                embedding_matrix=tokenizer.embedding_matrix,
                _recursive_=False,
            )

        elif cfg.model.name in ['HierRoBERTaGRU', 'HierSBERTGRU']:
            if cfg.tokenizer.name == 'rinna_roberta_bbs':
                cfg.tokenizer.args.pretrained_model = os.path.join(
                    cfg.workplace_dir, cfg.tokenizer.pretrained_model)

            data_module = hydra.utils.instantiate(
                cfg.model.data_module,
                data_dir=cfg.data.dir,
                tokenizer=cfg.tokenizer.args,
                _recursive_=False,
            )

            model = hydra.utils.instantiate(
                cfg.model.args,
                pretrained_model=cfg.tokenizer.args.pretrained_model,
                optim=cfg.optim,
                _recursive_=False,
            )

        elif cfg.model.name in ['HierBERT', 'HierRoBERT']:
            data_module = hydra.utils.instantiate(
                cfg.model.data_module,
                pretrained_model=cfg.tokenizer.args.pretrained_model,
                data_dir=cfg.data.dir,
                tokenizer=cfg.tokenizer.args,
                _recursive_=False,
            )

            model = hydra.utils.instantiate(
                cfg.model.args,
                pretrained_model=cfg.tokenizer.args.pretrained_model,
                sent_level_BERT_config=cfg.model.sent_level_BERT_config,
                optim=cfg.optim,
                _recursive_=False,
            )

        else:
            raise Exception(f'Model:{cfg.model} is invalid.')

        early_stop_callback = hydra.utils.instantiate(
            cfg.early_stopping,
        )

        checkpoint_callback = hydra.utils.instantiate(
            cfg.checkpoint_callback,
        )

        tb_logger = pl_loggers.TensorBoardLogger(
            ".", "", "", log_graph=True, default_hp_metric=False)

        from pytorch_lightning.callbacks import LearningRateMonitor
        lr_monitor = LearningRateMonitor(logging_interval='step')

        trainer = pl.Trainer(
            **OmegaConf.to_container(cfg.trainer),
            callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
            logger=tb_logger,
            plugins=DDPPlugin(find_unused_parameters=True),
        )

        """train, test or plot_attention"""
        if cfg.mode == 'train':
            trainer.fit(model=model, datamodule=data_module)
            trainer.test(ckpt_path=checkpoint_callback.best_model_path)

        elif cfg.mode == 'test' or cfg.mode == 'plot_attention':
            if cfg.checkpoint_dir is not None:
                checkpoint_path = os.path.join(
                    cfg.workplace_dir, "outputs", cfg.checkpoint_dir, f'epoch={cfg.best_epoch}.ckpt')
            else:
                checkpoint_path = os.path.join(
                    cfg.workplace_dir, "outputs", cfg.model.name, cfg.data.name, f'checkpoints/epoch={cfg.best_epoch}.ckpt')

            if not os.path.exists(checkpoint_path):
                raise Exception(
                    f'checkpoint_path:{checkpoint_path} is not exist.')

            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['state_dict'])

        if cfg.mode == 'test':
            trainer.test(model, datamodule=data_module)

        elif cfg.mode == 'plot_attention':
            outputs = trainer.predict(model, datamodule=data_module)
            create_html(cfg, tokenizer, outputs)

        else:
            raise Exception(f'Mode:{cfg.mode} is invalid.')

    except Exception as e:
        print(traceback.format_exc())
        # gmail_sender.send(body=f"<p>Error occurred while training.<br>{e}</p>")

    finally:
        pass


if __name__ == "__main__":
    main()
