import fasttext
import argparse
import os, sys
import shutil
sys.path.append(os.pardir)
from utils.gmail_send import Gmailsender

parser = argparse.ArgumentParser()
parser.add_argument("--dim", type=int, default=200)
parser.add_argument("--data", type=str, default='nested')
parser.add_argument("--tokenizer", type=str, default='mecab-wordpiece')
args = parser.parse_args()

model = fasttext.train_unsupervised(f'../../data/{args.data}/split-train-{args.tokenizer}.txt',
                                        model='skipgram',
                                        dim=args.dim,
                                        minCount=1)

save_dir = f'{args.tokenizer}_vectors/dim_{args.dim}'
shutil.rmtree(save_dir)
os.makedirs(save_dir, exist_ok=True)
model.save_model(f"{save_dir}/model_fasttext.bin")

from gensim.models import fasttext
model = fasttext.load_facebook_vectors(f"{save_dir}/model_fasttext.bin")
model.save_word2vec_format(f"{save_dir}/model_fasttext.vec")

sender = Gmailsender()
sender.send(f"fasttext(dim={args.dim})訓練終わり。")
