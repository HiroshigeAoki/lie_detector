import logging
import traceback

import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import LearningRateMonitor

from transformers import AutoTokenizer

import hydra
from omegaconf import OmegaConf, DictConfig

import os
import dotenv
from src.tokenizer.HFTokenizer import HFTokenizer
from src.model.HFModel import HFModel
from src.visualization.plot_attention import HtmlPlotter

logger = logging.getLogger(__name__)

dotenv.load_dotenv()

def setup_han(cfg):
    cfg.tokenizer.args.cache_dir = os.path.join(
                cfg.workplace_dir, cfg.tokenizer.args.cache_dir)

    if cfg.data.name == 'nested_sample':
        cfg.tokenizer.args.cache_dir = os.path.join(
                    cfg.tokenizer.args.cache_dir, 'sample')

    if cfg.tokenizer.name == 'sentencepiece':
        cfg.tokenizer.args.model_file = os.path.join(
                    cfg.workplace_dir, cfg.tokenizer.args.model_file)

    tokenizer = hydra.utils.instantiate(cfg.tokenizer.args, data_dir=cfg.data.dir)
            
    is_scam_game_data, is_murder_mystery_data = adjust_config_for_dataset(cfg)

    data_module = hydra.utils.instantiate(
                cfg.model.data_module,
                data_dir=cfg.data.dir,
                tokenizer=tokenizer,
                is_scam_game_data=is_scam_game_data,
                is_murder_mystery_data=is_murder_mystery_data
            )

    model = hydra.utils.instantiate(
                cfg.model.args,
                optim=cfg.optim,
                embedding_matrix=tokenizer.embedding_matrix,
                is_scam_game=is_scam_game_data,
                is_murder_mystery=is_murder_mystery_data,
                _recursive_=False,
            )
    return tokenizer, data_module, model


def setup_deberta(cfg):
    is_scam_game_data, is_murder_mystery_data = adjust_config_for_dataset(cfg)

    if 'deberta' not in cfg.tokenizer.name.lower():
        raise Exception(f'Tokenizer:{cfg.tokenizer.name} is invalid for Deberta model.')
        
    output_attentions = False
    if cfg.mode == 'test' or 'plot_attention':
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
                is_scam_game_data=is_scam_game_data,
                is_murder_mystery_data=is_murder_mystery_data
            )

    model = hydra.utils.instantiate(
                cfg.model.args,
                optim=cfg.optim,
                is_scam_game=is_scam_game_data,
                _recursive_=False,
            )
    
    return tokenizer, data_module, model


def setup_hfmodel(cfg):
    is_scam_game_data, is_murder_mystery_data = adjust_config_for_dataset(cfg)
    output_attentions = False
    
    if cfg.mode == 'test' or 'plot_attention':
        output_attentions = True
        
    tokenizer = HFTokenizer(
        AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=cfg.tokenizer.args.pretrained_model_name_or_path,
                additional_special_tokens=list(cfg.tokenizer.args.additional_special_tokens),
                output_attentions=output_attentions,
        )
    )
    
    data_module = hydra.utils.instantiate(
                cfg.model.data_module,
                data_dir=cfg.data.dir,
                tokenizer=tokenizer,
                is_scam_game_data=is_scam_game_data,
                is_murder_mystery_data=is_murder_mystery_data,
                data_type=cfg.data.data_type,
            )
    
    model = hydra.utils.instantiate(
                cfg.model.args,
                optim=cfg.optim,
                is_scam_game=is_scam_game_data,
                is_murder_mystery=is_murder_mystery_data,                
                _recursive_=False,
            )

    return tokenizer, data_module, model


def adjust_config_for_dataset(cfg):
    is_scam_game_data = False
    if cfg.data.name == "scam_game":
        cfg.model.data_module.batch_size = 1
        is_scam_game_data = True
            
    is_murder_mystery_data = False
    if cfg.data.name == "murder_mystery":
        cfg.model.data_module.batch_size = 1
        is_murder_mystery_data = True
        
    return is_scam_game_data,is_murder_mystery_data


def setup_trainer(cfg):
    early_stop_callback = hydra.utils.instantiate(
            cfg.early_stopping,
        )

    checkpoint_callback = hydra.utils.instantiate(
            cfg.checkpoint_callback,
        )

    tb_logger = pl_loggers.TensorBoardLogger(
            ".", "", "", log_graph=True, default_hp_metric=False)

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
            **OmegaConf.to_container(cfg.trainer),
            callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
            logger=tb_logger,
            plugins=DDPPlugin(find_unused_parameters=True),
        )
    
    return checkpoint_callback,trainer


@hydra.main(config_path="config", config_name="defaults")
def main(cfg: DictConfig) -> None:
    try:
        pl.seed_everything(1234)
        logger.info("\n" + OmegaConf.to_yaml(cfg))

        cfg.data.dir = os.path.join(cfg.workplace_dir, cfg.data.dir)

        """instantiate"""
        if cfg.model.name == 'HAN':
            tokenizer, data_module, model = setup_han(cfg)
            
        elif 'deberta' in cfg.model.name.lower():
            tokenizer, data_module, model = setup_deberta(cfg)
            
        elif cfg.model.name.lower().startswith('hf_'):
            tokenizer, data_module, model = setup_hfmodel(cfg)
            
        else:
            raise Exception(f'Model:{cfg.model} is invalid.')

        checkpoint_callback, trainer = setup_trainer(cfg)

        """train, test or plot_attention"""
        if cfg.mode == 'train':
            trainer.fit(model=model, datamodule=data_module)
            model = HFModel.load_from_checkpoint(checkpoint_callback.best_model_path)
            trainer.test(model=model, datamodule=data_module)

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
                plotter = HtmlPlotter(cfg, tokenizer, outputs)
                plotter.create_html()

        else:
            raise Exception(f'Mode:{cfg.mode} is invalid.')

    except Exception as e:
        print(e)
        print(traceback.format_exc())

    finally:
        pass


if __name__ == "__main__":
    main()
