import fasttext
import argparse
import os, sys
sys.path.append(os.pardir)

parser = argparse.ArgumentParser()
parser.add_argument("--dim", type=int, default=200)
args = parser.parse_args()

dim = args.dim

model = fasttext.train_unsupervised('split_train.txt',
                                        model='skipgram',
                                        dim=dim,
                                        minCount=5)

save_dir = f'dim_{dim}'
os.makedirs(save_dir, exist_ok=True)
model.save_model(f"{save_dir}/model_fasttext.bin")

from gensim.models.fasttext import FastText
model = FastText.load_fasttext_format(f"{save_dir}/model_fasttext.bin")
model.wv.save_word2vec_format(f"{save_dir}/model_fasttext.vec")