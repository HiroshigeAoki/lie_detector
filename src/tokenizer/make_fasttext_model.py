import fasttext
import argparse
import os, sys
import shutil
from gensim.models import fasttext as gensim_fasttext
sys.path.append('./src/')
print(sys.path)
from utils.gmail_send import Gmailsender


def main(args):
    model = fasttext.train_unsupervised(
        os.path.join(args.input_dir, f'split-train-{args.tokenizer}.txt'),
        model='skipgram',
        dim=args.dim,
        minCount=1
    )

    save_dir = os.path.join(args.save_dir, f'{args.tokenizer}_vectors/dim_{args.dim}')
    shutil.rmtree(save_dir, ignore_errors=True)
    os.makedirs(save_dir, exist_ok=True)
    model.save_model(os.path.join(save_dir, "model_fasttext.bin"))

    model = gensim_fasttext.load_facebook_vectors(os.path.join(save_dir, "model_fasttext.bin"))
    model.save_word2vec_format(os.path.join(save_dir, "model_fasttext.vec"))

    sample_save_dir = os.path.join(save_dir, 'sample', f'{args.tokenizer}_vectors/dim_{args.dim}')
    shutil.rmtree(sample_save_dir, ignore_errors=True)
    os.makedirs(sample_save_dir, exist_ok=True)
    print(f"Copy '.vec' file to {shutil.copy2(os.path.join(save_dir, 'model_fasttext.vec'), sample_save_dir)} for debug.")

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