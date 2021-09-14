import sentencepiece as spm
from pathlib import Path
import argparse

import os, sys
sys.path.append(os.pardir)
from utils.unix_command import mkdir


def main(args):
    save_dir = Path('./flat') if args.flat else Path('./nest')
    mkdir(save_dir)

    sp = spm.SentencePieceProcessor()
    spm.SentencePieceTrainer.Train(f"--input={save_dir}train.txt"
                                "--user_defined_symbols=<br> "
                                "--user_defined_symbols=<EOS> "
                                "--pad_id=3 "
                                "--model_prefix=wereWolf "
                                f"--model_dir={save_dir}"
                                "--vocab_size=32000 "
                                "--model_type=bpe")

    sp = spm.SentencePieceProcessor()
    sp.load(save_dir / 'wereWolf.model')
    with open(save_dir / "train.txt") as f:
        with open(save_dir / 'split_train.txt', "w") as outfile:
            for line in f:
                outfile.writelines(" ".join(sp.Encode(input=line, out_type=str))+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flat", action='store_true')
    args = parser.parse_args()
    main(args)