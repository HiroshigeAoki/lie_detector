import sentencepiece as spm
import argparse

import os


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)

    sp = spm.SentencePieceProcessor()
    spm.SentencePieceTrainer.Train(f"--input={os.path.join(args.input_dir, 'bbs.txt')} "
                                "--user_defined_symbols=<person> "
                                f"--model_prefix={os.path.join(args.save_dir, 'werewolf')} "
                                "--vocab_size=32000 "
                                "--model_type=bpe "
                                "--control_symbols=[PAD],[CLS],[SEP],[MASK] "
                                )

    sp = spm.SentencePieceProcessor()
    sp.load(os.path.join(args.save_dir, 'werewolf.model'))
    with open(os.path.join(args.input_dir, 'bbs.txt')) as f:
        with open(os.path.join(args.input_dir, 'split-train-sp.txt'), "w") as outfile:
            for line in f:
                outfile.writelines(" ".join(sp.Encode(input=line, out_type=str))+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default='data/nested')
    parser.add_argument("--save_dir", type=str, default='model/sp')

    args = parser.parse_args()
    main(args)