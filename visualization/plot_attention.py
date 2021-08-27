import matplotlib
import matplotlib.cm
import matplotlib.colors
import torch
from tqdm import tqdm


def plot_attention(sent: list[str], weights: list[str], threshold: float, cmap="Reds", color_level=5) -> str:
    r"""
    apply color to a sentence according to weights

    :param sent: a sentence
    :param weights: list of weights
    :param threshold: words under this threshold will be white background
    :param cmap: color of cmap
    :param color_level: adjust the intensity of color
    :return: colored sentence
    """
    cmap = matplotlib.cm.get_cmap(cmap)
    template = '<font face="monospace" \nsize="6"; span class="barcode"; style="color: black; background-color: {}">{}</span></font>'
    colored_sent = ""
    special_tokens = ['[PAD]', '[SEP]', '[CLS]', '[UNK]', '.', ' '] # ignore special tokens and punctuations
    for token, weight in zip(sent, weights):
        if token in special_tokens:
            continue
        color = matplotlib.colors.rgb2hex(cmap(weight.numpy() * color_level)[:3]) if not weight.numpy() < threshold else "#FFFFFF"
        colored_sent += template.format(color, "&nbsp" + token + "&nbsp")
    return colored_sent


def plot_attentions(sents: list[str], weights_list: list[torch.Tensor],
                    threshold: float, cmap="Reds", color_level: float = 5) -> list[str]:
    r"""
    apply color to sentences according to weights

    :param sents: list of sentences
    :param weights_list: list of weights
    :param threshold: words under this threshold will be white background
    :param cmap: color of cmap
    :param color_level: adjust the intensity of color
    :return: colored sentences
    """
    colored_sents = []
    for sent, weights in tqdm(zip(sents, weights_list), desc="ヒートマップを作成中..."):
        weights = weights
        colored_sent = plot_attention(sent, weights, cmap=cmap, color_level=color_level, threshold=threshold)
        colored_sents.append(colored_sent)
    return colored_sents
