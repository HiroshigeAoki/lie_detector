import sentencepiece as sp
import sys

# https://github.com/yoheikikuta/bert-japanese/blob/8d197e23b0e54da785ca9d16b7998c708767d649/src/tokenization_sentencepiece.py#L169
class SentencePieceTokenizer(object):
    """Runs SentencePiece tokenization (from raw text to tokens list)"""

    def __init__(self, model_file=None, do_lower_case=True):
        """Constructs a SentencePieceTokenizer."""
        self.tokenizer = sp.SentencePieceProcessor()
        if self.tokenizer.Load(model_file):
            print("Loaded a trained SentencePiece model.")
        else:
            print("You have to give a path of trained SentencePiece model.")
            sys.exit(1)
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        if self.do_lower_case:
            text = text.lower()
        output_tokens = self.tokenizer.EncodeAsPieces(text)
        return output_tokens
