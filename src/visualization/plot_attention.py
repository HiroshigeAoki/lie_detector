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
    ignore_tokens = ['[PAD]', '[SEP]', '[CLS]', '[UNK]', '.', ' '] # ignore special tokens and punctuations
    for token, weight in zip(sent, weights):
        if token in ignore_tokens:
            continue
        color = matplotlib.colors.rgb2hex(cmap(weight.numpy() * color_level)[:3]) if not weight.numpy() < threshold else "#FFFFFF"
        colored_sent += template.format(color, "&nbsp" + token + "&nbsp")
    return colored_sent


def plot_word_attentions(doc: list[str], weights_list: list[torch.Tensor],
                    threshold: float, cmap="Reds", color_level: float = 5, size: int = 4) -> list[str]:
    r"""
    apply color to sentences according to weights

    :param sents: list of sentences
    :param weights_list: list of weights
    :param threshold: words under this threshold will be white background
    :param cmap: color of cmap
    :param color_level: adjust the intensity of color
    :return: colored sentences
    """

    cmap = matplotlib.cm.get_cmap(cmap)
    template = '<font face="monospace" \nsize="{}"; span class="barcode"; style="color: black; background-color: {}">{}</span></font>'
    colored_doc = ""
    ignore_tokens = ['[PAD]', '[SEP]', '[CLS]', '[UNK]', '.', ' ']
    for sent, weights in tqdm(zip(doc, weights_list), desc="ヒートマップを作成中..."):
        weights = weights.cpu()
        for token, weight in zip(sent, weights):
            if token in ignore_tokens:
                continue
            color = matplotlib.colors.rgb2hex(cmap(weight.numpy() * color_level)[:3]) if not weight.numpy() < threshold else "#FFFFFF"
            colored_doc += template.format(size, color, "&nbsp" + token + "&nbsp")
        colored_doc += '</br>'
    return colored_doc


def plot_sent_attention(doc: list[str], weights_list: list[torch.Tensor],
                    threshold: float, cmap="Reds", color_level: float = 5, size: int = 4) -> list[str]:

    cmap = matplotlib.cm.get_cmap(cmap)
    template = '<font face="monospace" \nsize={}; span class="barcode"; style="color: black; background-color: {}">{}</span>'
    colored_doc = ""

    #TODO: Apply a some function to doc_w to highlight subtle difference between each weighs.
    #TODO: ここを直す。
    for sent, weight in zip(doc, weights_list):
        sent = ' '.join([token for token in sent if token != '<pad>'])
        if len(sent) > 0:
            color = matplotlib.colors.rgb2hex(cmap(weight.numpy()*color_level)[0][:3])  if not weight.numpy() < threshold else "#FFFFFF"
            colored_doc += template.format(size, color, "&nbsp" + sent + "&nbsp") + "</br>"
    return colored_doc

