import fasttext
import argparse
import os, sys
import shutil
from gensim.models import fasttext as gensim_fasttext
print(sys.path)
sys.path.append('./src/')
print(sys.path)
from utils.gmail_send import Gmailsender


def main(args):
    model = fasttext.train_unsupervised(os.path.join(args.input_dir, f'split-train-{args.tokenizer}.txt'),
                                        model='skipgram',
                                        dim=args.dim,
                                        minCount=1)

    save_dir = os.path.join(args.save_dir, f'{args.tokenizer}_vectors/dim_{args.dim}')
    shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    model.save_model(f"{save_dir}/model_fasttext.bin")

    model = gensim_fasttext.load_facebook_vectors(f"{save_dir}/model_fasttext.bin")
    model.save_word2vec_format(f"{save_dir}/model_fasttext.vec")

    sender = Gmailsender()
    sender.send(f"fasttext(dim={args.dim})訓練終わり。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=200)
    parser.add_argument("--input_dir", type=str, default='data/nested')
    parser.add_argument("--save_dir", type=str, default='model')
    parser.add_argument("--tokenizer", type=str, default='mecab-wordpiece')
    args = parser.parse_args()
    main(args)