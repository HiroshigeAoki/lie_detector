import logging
import traceback

import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.plugins import DDPPlugin

import hydra
from omegaconf import OmegaConf, DictConfig

from src.visualization.plot_attention import plot_attentions
import os
import joblib
from tqdm import tqdm

from src.utils.gmail_send import Gmailsender

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="defaults")
def main(cfg: DictConfig) -> None:
    try:
        gmail_sender = Gmailsender(subject=f"Execution end notification (model:{cfg.model.name}, data:{cfg.data.name})")

        pl.seed_everything(1234)
        logger.info("\n" + OmegaConf.to_yaml(cfg))

        """instantiate"""
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
                cfg.model.model,
                optim=cfg.optim,
                embedding_matrix=tokenizer.embedding_matrix,
                _recursive_=False,
            )

        elif cfg.model.name=='HierBERT' or cfg.model.name=='HierRoBERT':
            data_module = hydra.utils.instantiate(
                cfg.model.data_module,
                data_dir=cfg.data.dir,
                tokenizer=cfg.model.tokenizer,
                _recursive_=False,
            )

            model = hydra.utils.instantiate(
                cfg.model.model,
                pretrained_model=cfg.model.tokenizer.pretrained_model,
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

        tb_logger = pl_loggers.TensorBoardLogger(".", "", "", log_graph=True, default_hp_metric=False)

        trainer = pl.Trainer(
            **OmegaConf.to_container(cfg.trainer),
            callbacks=[checkpoint_callback, early_stop_callback],
            logger=tb_logger,
            plugins=DDPPlugin()
        )

        """train, test, or plot_attention"""
        if cfg.mode == 'train':
            trainer.fit(model=model, datamodule=data_module)
            trainer.test(ckpt_path=checkpoint_callback.best_model_path)

        elif cfg.mode == 'test':
            ckpt_path = f'checkpoints/epoch={cfg.best_epoch}.ckpt'
            test_model = model.load_from_checkpoint(ckpt_path)
            trainer.test(model=test_model, datamodule=data_module)

        elif cfg.mode == 'plot_attention':
            #ckpt_path = f'/disk/ssd14tb/haoki/Documents/vscode-workplaces/lie_detector/outputs/wereWolf/HAN/baseline/{cfg.name}/checkpoints/epoch={cfg.best_epoch}.ckpt'
            ckpt_path = f'checkpoints/epoch={cfg.best_epoch}.ckpt'
            predict_model = model.load_from_checkpoint(ckpt_path)
            outputs = trainer.predict(model=predict_model, datamodule=data_module)
            if cfg.model.name == 'HAN':
                logits = torch.cat([p['logits'] for p in outputs], dim=0)
                word_attentions = torch.cat([p['word_attentions'] for p in outputs]).cpu()
                sent_attentions = torch.cat([p['sent_attentions'].squeeze(2) for p in outputs]).cpu()
                input_ids = torch.cat([p['input_ids'] for p in outputs]).cpu()
                labels = torch.cat([p['labels'] for p in outputs]).cpu()
                ignore_tokens = ['<PAD>', '<unk>']
                pad_token = '<PAD>'

            # label 1: deceptive role, label 0: deceived role
            os.makedirs('ploted_attention/TP', exist_ok=True) # preds: 1, label: 1
            os.makedirs('ploted_attention/TN', exist_ok=True) # preds: 0, label: 0
            os.makedirs('ploted_attention/FP', exist_ok=True) # preds: 1, label: 0
            os.makedirs('ploted_attention/FN', exist_ok=True) # preds: 0, label: 1

            softmax = torch.nn.Softmax(dim=1)
            probs = softmax(logits).cpu()
            preds = logits.argmax(dim=1).cpu()

            def make_ploted_doc(i, input_ids, word_weights,  sent_weights ,prob, pred, label, kwargs):
                doc = [list(map(lambda x: x.replace(' ', ''), tokenizer.batch_decode(ids.tolist()))) for ids in input_ids]
                ploted_doc = plot_attentions(doc=doc, word_weights=word_weights, sent_weights=sent_weights, **kwargs)
                table_of_contents = dict(TP=[], TN=[], FP=[], FN=[])
                if pred == label:
                    if label == 1:
                        save_path = f'ploted_attention/TP/DC:{prob[label] * 100:.2f}% No.{i}.html' # DV stands for Degree of Conviction
                        table_of_contents['TP'].append(save_path.replace('ploted_attention/', './'))
                    elif label == 0:
                        save_path = f'ploted_attention/TN/DC:{prob[label] * 100:.2f}% No.{i}.html'
                        table_of_contents['TN'].append(save_path.replace('ploted_attention/', './'))
                elif pred != label:
                    if label == 1:
                        save_path = f'ploted_attention/FP/DC:{prob[pred] * 100:.2f}% No.{i}.html'
                        table_of_contents['FP'].append(save_path.replace('ploted_attention/', './'))
                    elif label == 0:
                        save_path = f'ploted_attention/FN/DC:{prob[pred] * 100:.2f}% No.{i}.html'
                        table_of_contents['FN'].append(save_path.replace('ploted_attention/', './'))
                with open(save_path, 'w') as f:
                    f.write(ploted_doc)
                return table_of_contents

            list_args = [(i, *args) for i, args in enumerate(zip(input_ids, word_attentions, sent_attentions, probs, preds, labels))]

            kwargs = dict(
                threshold=0.01, word_cmap="Blues" , sent_cmap="Reds",
                word_color_level=4, sent_color_level=35, size=3,
                ignore_tokens=ignore_tokens,
                pad_token=pad_token,
            )

            # TODO: indexのページを作成する。
            table_of_contents = joblib.Parallel(n_jobs=-1)(
                joblib.delayed(make_ploted_doc)(
                    *args,
                    kwargs=kwargs,
                ) for args in tqdm(list_args, desc='making ploted doc')
            )

            file = '<td><a href="{}">{}</a></td>'
            with open('ploted_attention/index.html', 'w') as f:
                f.write()

        else:
            raise Exception(f'Mode:{cfg.mode} is invalid.')

    except Exception as e:
        print(traceback.format_exc())
        gmail_sender.send(body=f"<p>Error occurred while training.<br>{e}</p>")

    finally:
        gmail_sender.send(body=f"{cfg.mode} was finished.")


if __name__ == "__main__":
    main()