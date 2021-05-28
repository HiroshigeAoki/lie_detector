import fasttext
import argparse
from pathlib import Path
import os, sys
sys.path.append(os.pardir)
from utils.unix_command import mkdir
from gensim.models.wrappers.fasttext import FastText

def main(args):
    dim = args.dim
    load_dir = Path('./flat') if args.flat else Path('./nest')

    model = fasttext.train_unsupervised(load_dir / 'split_train.txt',
                                        model='skipgram',
                                        dim=dim,
                                        minCount=5)

    save_dir = load_dir / f'dim_{dim}'
    mkdir(save_dir)
    model.save_model(save_dir / 'model_fasttext.bin')

    model = FastText.load_fasttext_format(save_dir / 'model_fasttext.bin')
    model.wv.save_word2vec_format(save_dir / 'model_fasttext.vec')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=300)
    parser.add_argument("--flat", action='store_true')
    args = parser.parse_args()
    main(args)