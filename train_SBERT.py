"""
This script is based on https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/other/training_batch_hard_trec.py

---
This script trains sentence transformers with a batch hard loss function.

Usual triplet loss takes 3 inputs: anchor, positive, negative and optimizes the network such that
the positive sentence is closer to the anchor than the negative sentence. However, a challenge here is
to select good triplets. If the negative sentence is selected randomly, the training objective is often
too easy and the network fails to learn good representations.

Batch hard triplet loss (https://arxiv.org/abs/1703.07737) creates triplets on the fly. It requires that the
data is labeled (e.g. labels 1, 2, 3) and we assume that samples with the same label are similar:

In a batch, it checks for sent1 with label 1 what is the other sentence with label 1 that is the furthest (hard positive)
which sentence with another label is the closest (hard negative example). It then tries to optimize this, i.e.
all sentences with the same label should be close and sentences for different labels should be clearly seperated.
"""

from sentence_transformers import SentenceTransformer, LoggingHandler, losses
from sentence_transformers.datasets import SentenceLabelDataset
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import TripletEvaluator
import pickle
import os

import logging
import traceback

import hydra
from omegaconf import DictConfig
from omegaconf.omegaconf import OmegaConf

from src.utils.gmail_send import Gmailsender

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)


def BBS_dataset(data_dir):
    with open(data_dir + 'train.pkl', 'rb') as f:
        train_set = pickle.load(f)
    with open(data_dir + 'valid.pkl', 'rb') as f:
        dev_set = pickle.load(f)
    with open(data_dir + 'test.pkl', 'rb') as f:
        test_set = pickle.load(f)
    return train_set, dev_set, test_set


@hydra.main(config_path="config", config_name="defaults")
def main(cfg: DictConfig):
    # You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
    ### Create a torch.DataLoader that passes training batch instances to our model
    try:
        gmail_sender = Gmailsender(subject=f"Execution end notification (model:{cfg.model.name}, data:{cfg.data.name})")

        if cfg.model.name != 'SBERT' or not cfg.data.name in ('triplet', 'triplet_sample'):
            raise ValueError(f'Model name{cfg.model.name} or data{cfg.data.name} are invalid. They must be "SBERT" and "triplet" or "triplet_sample" respectively.')

        logging.info("Loading BBS dataset")
        train_set, dev_set, test_set = BBS_dataset(cfg.data.dir)

        # We create a special dataset "SentenceLabelDataset" to wrap out train_set
        # It will yield batches that contain at least two samples with the same label
        train_data_sampler = SentenceLabelDataset(train_set)
        train_dataloader = DataLoader(train_data_sampler, batch_size=cfg.model.config.batch_size, drop_last=True)


        # Load pretrained model
        logging.info("Load model")
        model = SentenceTransformer(cfg.model.config.word_embedding_model) if not cfg.model.fit.resume_training else SentenceTransformer(cfg.model.fit.checkpoint_path_to_resume)


        ### Triplet losses ####################
        ### There are 4 triplet loss variants:
        ### - BatchHardTripletLoss
        ### - BatchHardSoftMarginTripletLoss
        ### - BatchSemiHardTripletLoss
        ### - BatchAllTripletLoss
        #######################################

        if cfg.model.config.loss_fnct == "all":
            train_loss = losses.BatchAllTripletLoss(model=model)
        elif cfg.model.config.loss_fnct == "hard":
            train_loss = losses.BatchHardTripletLoss(model=model)
        elif cfg.model.config.loss_fnct == "hard_soft_margin":
            train_loss = losses.BatchHardSoftMarginTripletLoss(model=model)
        elif cfg.model.config.loss_fnct == "semi_hard":
            train_loss = losses.BatchSemiHardTripletLoss(model=model)
        else:
            raise ValueError('loss_fnct(cfg.model.config.loss_fnct) must be either of "all", "hard", "hard_soft_margin" or "semi_hard".')


        logging.info("Read BBS val dataset")
        dev_evaluator = TripletEvaluator.from_input_examples(
            dev_set,
            name='BBS-dev',
            batch_size = cfg.model.config.batch_size,
            show_progress_bar = True
        )

        #logging.info("Performance before fine-tuning:")
        #dev_evaluator(model)

        warmup_steps = int(len(train_dataloader) * cfg.model.fit.epochs  * 0.1)  # 10% of train data

        os.makedirs(cfg.model.fit.output_path, exist_ok=True)
        os.makedirs(cfg.model.fit.checkpoint_path, exist_ok=True)

        # Train the model
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=dev_evaluator,
            warmup_steps=warmup_steps,
            **OmegaConf.to_container(cfg.model.fit),
        )

        ##############################################################################
        #
        # Load the stored model and evaluate its performance on BBS dataset
        #
        ##############################################################################

        logging.info("Evaluating model on test set")
        test_evaluator = TripletEvaluator.from_input_examples(
            test_set,
            name='BBS-test',
            batch_size = cfg.model.config.batch_size,
            show_progress_bar = True,
        )
        model.evaluate(test_evaluator)

    except Exception as e:
        print(traceback.format_exc())
        gmail_sender.send(body=f"<p>Error occurred while training.<br>{e}</p>")

    finally:
        gmail_sender.send(body=f"{cfg.mode} was finished.")


if __name__ == "__main__":
    main()
