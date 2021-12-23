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

from sentence_transformers.readers import InputExample

import pandas as pd
import pickle
import argparse

import os, sys
import random
from collections import defaultdict
import traceback

sys.path.append('./src/')
from utils.gmail_send import Gmailsender


# Inspired from torchnlp
def BBS_dataset(directory=''):
    print('Reading a nested dataset')
    train_df = pd.read_pickle(directory + "train.pkl")
    dev_df = pd.read_pickle(directory + "valid.pkl")
    test_df = pd.read_pickle(directory + "test.pkl")

    print('flattening a nested dataset')
    train_set = flatten_nested_dataset(train_df)
    dev_set = flatten_nested_dataset(dev_df)
    test_set = flatten_nested_dataset(test_df)

    # For dev & test set, we return triplets (anchor, positive, negative)
    random.seed(42) #Fix seed, so that we always get the same triplets
    print('making triplets in valid and test')
    dev_triplets = triplets_from_labeled_dataset(dev_set)
    test_triplets = triplets_from_labeled_dataset(test_set)

    return train_set, dev_triplets, test_triplets

def flatten_nested_dataset(df: pd.DataFrame):
    examples = []
    guid = 1
    for _, row in df.iterrows():
        for _, _row in row['nested_utters'].iterrows():
            guid += 1
            examples.append(InputExample(guid=guid, texts=[_row['raw_nested_utters']], label=row['labels']))
    return examples

def triplets_from_labeled_dataset(input_examples):
    # Create triplets for a [(label, sentence), (label, sentence)...] dataset
    # by using each example as an anchor and selecting randomly a
    # positive instance with the same label and a negative instance with a different label
    triplets = []
    label2sentence = defaultdict(list)
    for inp_example in input_examples:
        label2sentence[inp_example.label].append(inp_example)

    for inp_example in input_examples:
        anchor = inp_example

        if len(label2sentence[inp_example.label]) < 2: #We need at least 2 examples per label to create a triplet
            continue

        positive = None
        while positive is None or positive.guid == anchor.guid:
            positive = random.choice(label2sentence[inp_example.label])

        negative = None
        while negative is None or negative.label == anchor.label:
            negative = random.choice(input_examples)

        triplets.append(InputExample(texts=[anchor.texts[0], positive.texts[0], negative.texts[0]]))

    return triplets

def dump_as_pickle(save_dir, file_name, data):
    with open(save_dir + file_name, 'wb') as f:
        pickle.dump(data, f)


def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--sample", action="store_true")
        args = parser.parse_args()

        gmail_sender = Gmailsender(subject="Execution end notification: make triplet dataset")
        train, valid, test = BBS_dataset(directory='data/nested/') if not args.sample else BBS_dataset(directory='data/nested_sample/')
        save_dir = "data/triplet/" if not args.sample else 'data/triplet_sample/'
        os.makedirs(save_dir, exist_ok=True)
        print('dump a triplet dataset')
        dump_as_pickle(save_dir, 'train.pkl', train)
        dump_as_pickle(save_dir, 'valid.pkl', valid)
        dump_as_pickle(save_dir, 'test.pkl', test)

    except Exception as e:
        print(traceback.format_exc())
        gmail_sender.send(body=f"<p>Error occurred while training.<br>{e}</p>")

    finally:
        gmail_sender.send(body="make_triplet_dataset.py was finished.")


if __name__ == "__main__":
    main()