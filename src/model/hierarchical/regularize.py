import torch
from torch import nn
import warnings
import sys

def embedded_dropout(embed, words, dropout=0.1, scale=None):
    if dropout:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(
            1 - dropout
        ).expand_as(embed.weight) / (1 - dropout)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight
    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1

    X = torch.nn.functional.embedding(
        words,
        masked_embed_weight,
        padding_idx,
        embed.max_norm,
        embed.norm_type,
        embed.scale_grad_by_freq,
        embed.sparse,
    )
    return X

class LockedDropout(nn.Module):
    """ LockedDropout applies the same dropout mask to every time step.

    **Thank you** to Sales Force for their initial implementation of :class:`WeightDrop`. Here is
    their `License
    <https://github.com/salesforce/awd-lstm-lm/blob/master/LICENSE>`__.

    Args:
        p (float): Probability of an element in the dropout mask to be zeroed.
    """

    def __init__(self, p=0.5):
        self.p = p
        super().__init__()

    def forward(self, x):
        """
        Args:
            x (:class:`torch.FloatTensor` [sequence length, batch size, rnn hidden size]): Input to
                apply dropout too.
        """
        if not self.training or not self.p:
            return x
        x = x.clone()
        mask = x.new_empty(1, x.size(1), x.size(2), requires_grad=False).bernoulli_(1 - self.p)
        mask = mask.div_(1 - self.p)
        mask = mask.expand_as(x)
        return x * mask


    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'p=' + str(self.p) + ')'


"""
Code here is taken DIRECTLY from the awd-lstm-lm Salesforce repo:
https://github.com/salesforce/awd-lstm-lm/tree/32fcb42562aeb5c7e6c9dec3f2a3baaaf68a5cb5
Credit to the authors: Stephen Merity, Nitish Shirish Keskar and Richard Socher
"""
class WeightDrop(nn.Module):
    def __init__(self, module, weights, dropout=0, variational=False, verbose=False, device="cpu"):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self.verbose = verbose
        self._setup()
        self.device = device

    def widget_demagnetizer_y2k_edition(*args, **kwargs):
        # We need to replace flatten_parameters with a nothing function
        # It must be a function rather than a lambda as otherwise pickling explodes
        # We can't write boring code though, so ... WIDGET DEMAGNETIZER Y2K EDITION!
        # (╯°□°）╯︵ ┻━┻
        return

    def _setup(self):
        # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = self.widget_demagnetizer_y2k_edition

        for name_w in self.weights:
            if self.verbose:
                print("Applying weight drop of {} to {}".format(self.dropout, name_w), file=sys.stderr)
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + "_raw", nn.Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + "_raw")
            w = None
            if self.variational:
                mask = torch.ones(raw_w.size(0), 1)
                if raw_w.is_cuda:
                    mask = mask.to(self.device)
                mask = torch.nn.functional.dropout(mask, p=self.dropout, training=self.training)
                w = nn.Parameter(mask.expand_as(raw_w) * raw_w)
            else:
                w = nn.Parameter(
                    torch.nn.functional.dropout(raw_w, p=self.dropout, training=self.training)
                )
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        with warnings.catch_warnings():
            # To avoid the warning that comes because the weights aren't flattened.
            warnings.simplefilter("ignore")
            return self.module.forward(*args)
