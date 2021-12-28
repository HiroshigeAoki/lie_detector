import logging
import traceback
from pytorch_lightning import plugins

import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.plugins import DDPPlugin, ParallelPlugin, DeepSpeedPlugin

import hydra
from omegaconf import OmegaConf, DictConfig

from src.visualization.plot_attention import plot_attentions
import os
import joblib
from tqdm import tqdm
from collections import Counter
import pandas as pd

from src.utils.gmail_send import Gmailsender

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="defaults")
def main(cfg: DictConfig) -> None:
    try:
        gmail_sender = Gmailsender(subject=f"Execution end notification (model:{cfg.model.name}, data:{cfg.data.name})")

        pl.seed_everything(1234)
        logger.info("\n" + OmegaConf.to_yaml(cfg))

        cfg.data.dir = os.path.join(cfg.workplace_dir, cfg.data.dir)

        """instantiate"""
        if cfg.model.name == 'HAN':
            cfg.tokenizer.args.cache_dir = os.path.join(cfg.workplace_dir, cfg.tokenizer.args.cache_dir)

            if cfg.data.name=='nested_sample':
                cfg.tokenizer.args.cache_dir = os.path.join(cfg.tokenizer.args.cache_dir, 'sample')

            if cfg.tokenizer.name=='sentencepiece':
                cfg.tokenizer.args.model_file = os.path.join(cfg.workplace_dir, cfg.tokenizer.args.model_file)

            tokenizer = hydra.utils.instantiate(
                cfg.tokenizer.args,
                data_dir=cfg.data.dir,
            )

            data_module = hydra.utils.instantiate(
                cfg.model.data_module,
                data_dir=cfg.data.dir,
                tokenizer=tokenizer,
            )

            model = hydra.utils.instantiate(
                cfg.model.args,
                optim=cfg.optim,
                embedding_matrix=tokenizer.embedding_matrix,
                _recursive_=False,
            )

        elif cfg.model.name in ['HierRoBERTaGRU', 'HierSBERTGRU']:
            if cfg.tokenizer.name == 'rinna_roberta_bbs':
                cfg.tokenizer.args.pretrained_model = os.path.join(cfg.workplace_dir, cfg.tokenizer.pretrained_model)

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

        tb_logger = pl_loggers.TensorBoardLogger(".", "", "", log_graph=True, default_hp_metric=False)

        #profiler = pl.profiler.PytorchProfiler(profile_memory=True)
        #from pytorch_lightning.profiler import PyTorchProfiler
        #profiler = PyTorchProfiler(filename='profile.txt')

        #from pytorch_lightning.accelerators import GPUAccelerator
        #from pytorch_lightning.plugins import NativeMixedPrecisionPlugin, DDPPlugin

        #accelerator = GPUAccelerator()
        #precision_plugin = NativeMixedPrecisionPlugin(precision=16, device="cuda")
        #training_type_plugin = DDPPlugin(accelerator=accelerator, precision_plugin=precision_plugin)

        trainer = pl.Trainer(
            **OmegaConf.to_container(cfg.trainer),
            callbacks=[checkpoint_callback, early_stop_callback],
            logger=tb_logger,
            #strategy=training_type_plugin,
            plugins=DDPPlugin(find_unused_parameters=True),
        )

        """train, test or plot_attention"""
        if cfg.mode == 'train':
            trainer.fit(model=model, datamodule=data_module)
            trainer.test(ckpt_path=checkpoint_callback.best_model_path)

        elif cfg.mode == 'test':
            ckpt_path = f'checkpoints/epoch={cfg.best_epoch}.ckpt'
            test_model = model.load_from_checkpoint(ckpt_path, strict=False)
            trainer.test(model=test_model, datamodule=data_module)

        elif cfg.mode == 'plot_attention':
            ckpt_path = f'checkpoints/epoch={cfg.best_epoch}.ckpt'
            if cfg.mode=='debug':
                ckpt_path = os.path.join(cfg.workplace_dir, 'outputs/nested/HAN/baseline/200_dim200_ignore_pad_sp_new/checkpoints/epoch=0.ckpt')
            predict_model = model.load_from_checkpoint(ckpt_path, strict=False)
            outputs = trainer.predict(model=predict_model, datamodule=data_module)

            if cfg.model.name == 'HAN':
                logits = torch.cat([p['logits'] for p in outputs], dim=0)
                word_attentions = torch.cat([p['word_attentions'] for p in outputs]).cpu()
                sent_attentions = torch.cat([p['sent_attentions'].squeeze(2) for p in outputs]).cpu()
                input_ids = torch.cat([p['input_ids'] for p in outputs]).cpu()
                pad_sent_num = torch.cat([p['pad_sent_num'] for p in outputs]).cpu()

                labels = torch.cat([p['labels'] for p in outputs]).cpu()

            save_dir = f'plot_attention_{cfg.tokenizer.plot_attention.n_gram}-gram'

            # label 1: deceptive role, label 0: deceived role
            os.makedirs(os.path.join(save_dir, 'TP'), exist_ok=True) # preds: 1, label: 1
            os.makedirs(os.path.join(save_dir, 'TN'), exist_ok=True) # preds: 0, label: 0
            os.makedirs(os.path.join(save_dir, 'FP'), exist_ok=True) # preds: 1, label: 0
            os.makedirs(os.path.join(save_dir, 'FN'), exist_ok=True) # preds: 0, label: 1

            softmax = torch.nn.Softmax(dim=1)
            probs = softmax(logits).cpu()
            preds = logits.argmax(dim=1).cpu()

            def make_ploted_doc(i, input_ids, word_weights, sent_weights, pad_sent_num ,prob, pred, label, kwargs):
                doc = [list(map(lambda x: x.replace(' ', ''), tokenizer.batch_decode(ids.tolist()))) for ids in input_ids]
                ploted_doc, vital_word_count = plot_attentions(doc=doc, word_weights=word_weights, sent_weights=sent_weights, pad_sent_num=pad_sent_num, **kwargs)
                table_of_contents_list = []
                if pred == label:
                    if label == 1:
                        pred_class = 'TP'
                        file_name = f'DC:{prob[label] * 100:.2f}% No.{i}.html' # DV stands for Degree of Conviction
                        table_of_contents_list.extend(('TP', file_name))
                    elif label == 0:
                        pred_class = 'TN'
                        file_name = f'DC:{prob[label] * 100:.2f}% No.{i}.html'
                        table_of_contents_list.extend(('TN', file_name))
                elif pred != label:
                    if label == 1:
                        pred_class = 'FP'
                        file_name = f'DC:{prob[pred] * 100:.2f}% No.{i}.html'
                        table_of_contents_list.extend(('FP', file_name))
                    elif label == 0:
                        pred_class = 'FN'
                        file_name = f'DC:{prob[pred] * 100:.2f}% No.{i}.html'
                        table_of_contents_list.extend(('FN', file_name))
                with open(os.path.join(save_dir, pred_class, file_name), 'w') as f:
                    f.write(ploted_doc)
                return table_of_contents_list, vital_word_count, pred_class, prob[pred]

            list_args = [(i, *args) for i, args in enumerate(zip(input_ids, word_attentions, sent_attentions, pad_sent_num, probs, preds, labels))]

            outputs = joblib.Parallel(n_jobs=-1)(
                joblib.delayed(make_ploted_doc)(
                    *args,
                    kwargs=OmegaConf.to_container(cfg.tokenizer.plot_attention),
                ) for args in tqdm(list_args, desc='making ploted doc')
            )

            template = '<td><a href="{}">{}</a></td>'

            table_of_contents = dict(TP=[], TN=[], FP=[], FN=[])
            vital_word_count_dict = dict(
                TP_90=[], TP_80=[], TP_70=[], TP_60_50=[],
                TN_90=[], TN_80=[], TN_70=[], TN_60_50=[],
                FP_90=[], FP_80=[], FP_70=[], FP_60_50=[],
                FN_90=[], FN_80=[], FN_70=[], FN_60_50=[],
            )
            for output in outputs:
                tc, vital_word_count, pred_class, prob = output[0], output[1], output[2], output[3]
                table_of_contents.get(tc[0]).append(tc[1])
                if prob >= 0.9:
                    confidence = 90
                elif 0.9 > prob >= 0.8:
                    confidence = 80
                elif 0.8 > prob >= 0.7:
                    confidence = 70
                else:
                    confidence = '60_50'
                vital_word_count_dict[f'{pred_class}_{confidence}'].extend(vital_word_count)

            par_link = [template.format(f'./{key}.html', key) for key in table_of_contents.keys()]
            with open(os.path.join(save_dir, 'index.html'), 'w') as f:
                f.write('<ui>')
                for link in par_link:
                    f.write('<li>' + link + '</li>')
                f.write('</ui>')

            for key, file_names in table_of_contents.items():
                file_names = sorted(file_names, reverse=True)
                chi_link = [template.format(f'./{key}/{file_name}', file_name) for file_name in file_names]
                with open(os.path.join(save_dir, f'{key}.html'), 'w') as f:
                    f.write('<ui>')
                    for link in chi_link:
                        f.write('<li>' + link + '</li>')
                    f.write('</ui>')

            def list_to_csv(pred_class_confidence, vital_word_list):
                df = pd.DataFrame([{'token': token, 'freq': freq} for token, freq in Counter(vital_word_list).most_common()])
                df.to_csv(os.path.join(save_dir, f'csv/{pred_class_confidence}_vital_word_freq.csv'))

            os.makedirs(os.path.join(save_dir, 'csv'), exist_ok=True)
            joblib.Parallel(n_jobs=4)(
                joblib.delayed(list_to_csv)(
                    *args
                ) for args in tqdm(vital_word_count_dict.items(), desc='making vital_word_count.csv')
            )

        else:
            raise Exception(f'Mode:{cfg.mode} is invalid.')

    except Exception as e:
        print(traceback.format_exc())
        gmail_sender.send(body=f"<p>Error occurred while training.<br>{e}</p>")

    finally:
        gmail_sender.send(body=f"{cfg.mode} was finished.")


if __name__ == "__main__":
    main()
