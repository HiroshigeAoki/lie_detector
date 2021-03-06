from torchtext.vocab import Vocab, Vectors
from collections import Counter, OrderedDict
from typing import Dict, List, Optional, Iterable
import os
from torchtext._torchtext import (
    Vocab as VocabPybind,
)

def build_vocab_from_training_data(data_dir: str, tokenizer_type: str, **kwargs) -> Vocab:
    def yield_tokens():
            with open(os.path.join(data_dir, f'split-train-{tokenizer_type}.txt'), 'r') as f:
                for line in f:
                    yield line.split()
    return build_vocab_from_iterator_with_vocab_size(iterator=yield_tokens(), **kwargs)


def vocab_with_vocab_size(ordered_dict: Dict, min_freq: int = 1, vocab_size: int = 32000) -> Vocab:
    r"""Factory method for creating a vocab object which maps tokens to indices.

    Note that the ordering in which key value pairs were inserted in the `ordered_dict` will be respected when building the vocab.
    Therefore if sorting by token frequency is important to the user, the `ordered_dict` should be created in a way to reflect this.

    Args:
        ordered_dict: Ordered Dictionary mapping tokens to their corresponding occurance frequencies.
        min_freq: The minimum frequency needed to include a token in the vocabulary.
        vocab_size: The number of words in vocabulary.

    Returns:
        torchtext.vocab.Vocab: A `Vocab` object

    Examples:
        >>> from torchtext.vocab import vocab
        >>> from collections import Counter, OrderedDict
        >>> counter = Counter(["a", "a", "b", "b", "b"])
        >>> sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        >>> ordered_dict = OrderedDict(sorted_by_freq_tuples)
        >>> v1 = vocab(ordered_dict)
        >>> print(v1['a']) #prints 1
        >>> print(v1['out of vocab']) #raise RuntimeError since default index is not set
        >>> tokens = ['e', 'd', 'c', 'b', 'a']
        >>> v2 = vocab(OrderedDict([(token, 1) for token in tokens]))
        >>> #adding <unk> token and default index
        >>> unk_token = '<unk>'
        >>> default_index = -1
        >>> if unk_token not in v2: v2.insert_token(unk_token, 0)
        >>> v2.set_default_index(default_index)
        >>> print(v2['<unk>']) #prints 0
        >>> print(v2['out of vocab']) #prints -1
        >>> #make default index same as index of unk_token
        >>> v2.set_default_index(v2[unk_token])
        >>> v2['out of vocab'] is v2[unk_token] #prints True
    """

    tokens = []
    for token, freq in ordered_dict.items():
        if freq >= min_freq:
            tokens.append(token)
        if len(tokens) > vocab_size:
            break

    return Vocab(VocabPybind(tokens, None))


def build_vocab_from_iterator_with_vocab_size(iterator: Iterable, min_freq: int = 1, vocab_size: int = 32000, specials: Optional[List[str]] = None, special_first: bool = True) -> Vocab:
    """
    Build a Vocab from an iterator.

    Args:
        iterator: Iterator used to build Vocab. Must yield list or iterator of tokens.
        min_freq: The minimum frequency needed to include a token in the vocabulary.
        specials: Special symbols to add. The order of supplied tokens will be preserved.
        special_first: Indicates whether to insert symbols at the beginning or at the end.
        vocab_size: The number of words in vocabulary.


    Returns:
        torchtext.vocab.Vocab: A `Vocab` object

    Examples:
        >>> #generating vocab from text file
        >>> import io
        >>> from torchtext.vocab import build_vocab_from_iterator
        >>> def yield_tokens(file_path):
        >>>     with io.open(file_path, encoding = 'utf-8') as f:
        >>>         for line in f:
        >>>             yield line.strip().split()
        >>> vocab = build_vocab_from_iterator(yield_tokens_batch(file_path), specials=["<unk>"])
    """

    counter = Counter()
    for tokens in iterator:
        counter.update(tokens)

    if specials is not None:
        for tok in specials:
            del counter[tok]

    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[0])
    sorted_by_freq_tuples.sort(key=lambda x: x[1], reverse=True)
    ordered_dict = OrderedDict(sorted_by_freq_tuples)

    if specials is not None:
        if special_first:
            specials = specials[::-1]
        for symbol in specials:
            ordered_dict.update({symbol: min_freq})
            ordered_dict.move_to_end(symbol, last=not special_first)

    word_vocab = vocab_with_vocab_size(ordered_dict, min_freq=min_freq, vocab_size=vocab_size + len(specials))
    return word_vocab