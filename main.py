import logging
import traceback

import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor
from transformers import AutoTokenizer
from pytorch_lightning.strategies import DDPStrategy

import hydra
from omegaconf import OmegaConf, DictConfig

import os
import re
import dotenv
from glob import glob
from src.tokenizer.HFTokenizer import HFTokenizer
from src.visualization.plot_attention import HtmlPlotter
from src.visualization.create_transfomer_attention_vis import (
    create_transfomer_attention_vis,
)

logger = logging.getLogger(__name__)
dotenv.load_dotenv()


def adjust_config_for_dataset(cfg):
    is_murder_mystery_data = False
    if cfg.data.name == "murder_mystery":
        cfg.model.data_module.batch_size = 1
        is_murder_mystery_data = True

    return is_murder_mystery_data


def setup_han(cfg):
    cfg.tokenizer.args.cache_dir = os.path.join(
        cfg.workplace_dir, cfg.tokenizer.args.cache_dir
    )

    if cfg.data.name == "nested_sample":
        cfg.tokenizer.args.cache_dir = os.path.join(
            cfg.tokenizer.args.cache_dir, "sample"
        )

    if cfg.tokenizer.name == "sentencepiece":
        cfg.tokenizer.args.model_file = os.path.join(
            cfg.workplace_dir, cfg.tokenizer.args.model_file
        )

    tokenizer = hydra.utils.instantiate(cfg.tokenizer.args, data_dir=cfg.data.dir)

    is_murder_mystery_data = adjust_config_for_dataset(cfg)

    data_module = hydra.utils.instantiate(
        cfg.model.data_module,
        data_dir=cfg.data.dir,
        tokenizer=tokenizer,
        is_murder_mystery_data=is_murder_mystery_data,
    )

    model = hydra.utils.instantiate(
        cfg.model.args,
        optim=cfg.optim,
        embedding_matrix=tokenizer.embedding_matrix,
        _recursive_=False,
    )
    return tokenizer, data_module, model


def setup_deberta(cfg):
    is_murder_mystery_data = adjust_config_for_dataset(cfg)

    if "deberta" not in cfg.tokenizer.name.lower():
        raise Exception(f"Tokenizer:{cfg.tokenizer.name} is invalid for Deberta model.")

    output_attentions = False
    if cfg.mode == "test" or "plot_attention":
        output_attentions = True

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=cfg.tokenizer.args.pretrained_model_name_or_path,
        additional_special_tokens=list(cfg.tokenizer.args.additional_special_tokens),
        output_attentions=output_attentions,
    )

    data_module = hydra.utils.instantiate(
        cfg.model.data_module,
        data_dir=cfg.data.dir,
        tokenizer=tokenizer,
        is_murder_mystery_data=is_murder_mystery_data,
    )

    model = hydra.utils.instantiate(
        cfg.model.args,
        optim=cfg.optim,
        _recursive_=False,
    )

    return tokenizer, data_module, model


def setup_hfmodel(cfg):
    is_murder_mystery_data = adjust_config_for_dataset(cfg)
    output_attentions = False

    if cfg.mode == "test" or "plot_attention":
        output_attentions = True

    tokenizer = HFTokenizer(
        AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=cfg.tokenizer.args.pretrained_model_name_or_path,
            additional_special_tokens=list(
                cfg.tokenizer.args.additional_special_tokens
            ),
            output_attentions=output_attentions,
        ),
        max_length=cfg.tokenizer.args.max_length,
    )

    data_module = hydra.utils.instantiate(
        cfg.model.data_module,
        data_dir=cfg.data.dir,
        tokenizer=tokenizer,
        is_murder_mystery_data=is_murder_mystery_data,
        data_type=cfg.data.data_type,
    )

    model = hydra.utils.instantiate(
        cfg.model.args,
        optim=cfg.optim,
        _recursive_=False,
    )

    return tokenizer, data_module, model


def setup_trainer(cfg):
    early_stop_callback = hydra.utils.instantiate(
        cfg.early_stopping,
    )

    checkpoint_callback = hydra.utils.instantiate(
        cfg.checkpoint_callback,
    )
    tb_logger = pl_loggers.TensorBoardLogger(
        ".", "", "", log_graph=True, default_hp_metric=False
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        **OmegaConf.to_container(cfg.trainer),
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=tb_logger,
        strategy=DDPStrategy(find_unused_parameters=True),
    )

    return checkpoint_callback, trainer


def check_if_exist_checkpoint(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise Exception(f"checkpoint_path:{checkpoint_path} is not exist.")
    return


@hydra.main(config_path="config", config_name="defaults")
def main(cfg: DictConfig) -> None:
    try:
        pl.seed_everything(1234)
        logger.info("\n" + OmegaConf.to_yaml(cfg))

        cfg.data.dir = os.path.join(cfg.workplace_dir, cfg.data.dir)

        """instantiate"""
        if cfg.model.name == "HAN":
            tokenizer, data_module, model = setup_han(cfg)
        elif cfg.model.name.lower().startswith("hf_"):
            tokenizer, data_module, model = setup_hfmodel(cfg)
        else:
            raise Exception(f"Model:{cfg.model} is invalid.")

        checkpoint_callback, trainer = setup_trainer(cfg)

        """train, test or plot_attention"""
        if cfg.mode == "train":
            trainer.fit(model=model, datamodule=data_module)
            # find_file_with_lowest_loss("./checkpoints")
            # model.load_state_dict(torch.load(os.path.join("./checkpoints", find_file_with_lowest_loss("./checkpoints")))['state_dict'])
            model.load_state_dict(
                torch.load(checkpoint_callback.best_model_path)["state_dict"]
            )
            trainer.test(model=model, datamodule=data_module)

        elif cfg.mode == "test" or cfg.mode == "plot_attention":
            if cfg.checkpoint_dir is not None:
                if cfg.checkpoint_dir.startswith("model/"):
                    files = glob(
                        os.path.join(cfg.workplace_dir, cfg.checkpoint_dir, "*.ckpt")
                    )
                    if len(files) == 0:
                        raise Exception(
                            f"Checkpoint directory:{cfg.checkpoint_dir} is not exist."
                        )
                    checkpoint_path = files[0]
                else:
                    checkpoint_path = os.path.join(
                        cfg.workplace_dir,
                        "outputs",
                        cfg.checkpoint_dir,
                        f"epoch={cfg.best_epoch}.ckpt",
                    )
            else:
                checkpoint_path = os.path.join(
                    cfg.workplace_dir,
                    "outputs",
                    cfg.model.name,
                    cfg.data.name,
                    f"checkpoints/epoch={cfg.best_epoch}.ckpt",
                )
            check_if_exist_checkpoint(checkpoint_path)
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["state_dict"])

            if cfg.mode == "test":
                trainer.test(model, datamodule=data_module)

            elif cfg.mode == "plot_attention":
                if cfg.model.name == "HAN":
                    outputs = trainer.predict(model, datamodule=data_module)
                    plotter = HtmlPlotter(cfg, tokenizer, outputs)
                    plotter.create_html()
                elif cfg.model.name == "hf_bigbird":
                    trainer.predict(model, datamodule=data_module)

        else:
            raise Exception(f"Mode:{cfg.mode} is invalid.")

    except Exception as e:
        print(e)
        print(traceback.format_exc())

    finally:
        pass


if __name__ == "__main__":
    main()
