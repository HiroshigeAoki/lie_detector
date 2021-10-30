import matplotlib
import matplotlib.cm
import matplotlib.colors
import torch

# colormaps: https://matplotlib.org/stable/tutorials/colors/colormaps.html
def plot_attentions(doc: list[str], word_weights: list[torch.tensor], sent_weights: list[torch.tensor],
                        threshold: float, word_cmap="Blues" , sent_cmap="Reds", word_color_level=5, sent_color_level=5, size: int = 4,
                        ignore_tokens = ['[PAD]', '[SEP]', '[CLS]', '[UNK]', '.', ' '], pad_token='[PAD]') -> str:

    colored_doc = ""
    word_cmap = matplotlib.cm.get_cmap(word_cmap)
    sent_cmap = matplotlib.cm.get_cmap(sent_cmap)
    template_word = '<font face="monospace" \nsize="{}"; span class="barcode"; style="color: black; background-color: {}">{}</span></font>'
    template_sent = '<font face="monospace" \nsize="{}"; span class="barcode"; style="color: black; background-color: {}">{}</span></font>'

    for sent, _word_weights, sent_weight in zip(doc, word_weights, sent_weights):
        if sent[0] == pad_token:
            break
        sent_color = matplotlib.colors.rgb2hex(sent_cmap(sent_weight.numpy() * sent_color_level)[:3]) if not sent_weight.numpy() < threshold else "#FFFFFF"
        colored_doc += template_sent.format(size, sent_color, "&nbsp" + ' ' + "&nbsp")
        for token, word_weight in zip(sent, _word_weights):
            if token in ignore_tokens:
                continue
            if '<' in token and '>' in token:
                token = token.replace('<', '＜')
                token = token.replace('>', '＞')
            token = token.replace('#', '')
            word_color = matplotlib.colors.rgb2hex(word_cmap(word_weight.numpy() * word_color_level)[:3]) if not word_weight.numpy() < threshold else "#FFFFFF"
            colored_doc += template_word.format(size, word_color, token)
        colored_doc += '</br>'
    return colored_doc