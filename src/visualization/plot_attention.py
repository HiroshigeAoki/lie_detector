import matplotlib
import matplotlib.cm
import matplotlib.colors
import torch
from collections import Counter

# colormaps: https://matplotlib.org/stable/tutorials/colors/colormaps.html
def plot_attentions(doc: list[str], word_weights: list[torch.tensor], sent_weights: list[torch.tensor], pad_sent_num: torch.tensor,
                        word_cmap="Blues" , sent_cmap="Reds", word_color_level=5, sent_color_level=5, size: int = 4,
                        ignore_tokens = ['[PAD]', '[SEP]', '[CLS]', '[UNK]', '.', ' '], pad_token='[PAD]') -> str:

    colored_doc = ""
    word_cmap = matplotlib.cm.get_cmap(word_cmap)
    sent_cmap = matplotlib.cm.get_cmap(sent_cmap)
    template_word = '<font face="monospace" \nsize="{}"; span class="barcode"; style="color: black; background-color: {}">{}</span></font>'
    template_sent = '<font face="monospace" \nsize="{}"; span class="barcode"; style="color: black; background-color: {}">{}</span></font>'

    vital_word_list = []
    template_vital_word = '<li><font face="monospace" \nsize="{}"; span class="barcode"; style="color: black">{}・・・{}回</span></font></li>'

    sent_threshold = adjust_threshold(len(doc) - pad_sent_num)

    for sent, _word_weights, sent_weight in zip(doc, word_weights, sent_weights):
        tokens, weights = [], []
        if sent[0] == pad_token:
            break

        sent_color = matplotlib.colors.rgb2hex(sent_cmap(sent_weight.numpy() * sent_color_level)[:3]) if sent_weight.numpy() > sent_threshold else "#FFFFFF"
        colored_doc += template_sent.format(size, sent_color, "&nbsp" + ' ' + "&nbsp")

        for token, word_weight in zip(sent, _word_weights):
            if token in ignore_tokens:
                continue
            if '<' in token and '>' in token:
                token = token.replace('<', '＜')
                token = token.replace('>', '＞')
            token = token.replace('#', '')
            tokens.append(token)
            weights.append(word_weight.numpy())

        word_threshold = adjust_threshold(len(tokens))
        for token, weight in zip(tokens, weights):
            if word_weight.numpy() > word_threshold:
                word_color = matplotlib.colors.rgb2hex(word_cmap(word_weight.numpy() * word_color_level)[:3])
                vital_word_list.append(token)
            else:
                word_color =  "#FFFFFF"
            colored_doc += template_word.format(size, word_color, token)

        colored_doc += '</br>'

    vital_word_count = Counter(vital_word_list)
    sorted_vital_words = vital_word_count.most_common()
    vital_word_table = ''
    for token, freq in sorted_vital_words:
        vital_word_table += template_vital_word.format(size, token, freq)

    colored_doc = (
        f'<h1>重要単語リスト(出現頻度順)</h1><ol>{vital_word_table}</ol>'
        +
        '<hr style="border:0;border-top:medium solid black;">'
        +
        '<h1>本文</h1>'
        +
        colored_doc
    )
    return colored_doc, vital_word_list

def adjust_threshold(length):
    """adjust threshold according to lengths

    Shape:
        ```
        import matplotlib.pyplot as plt
        import numpy as np
        import math
        a = 10
        b = 10
        x = np.linspace( 1, 256, 256)
        y = a / (x + b)
        plt.plot(x, y)
        plt.title(f'y = {a} / (x + {b})')
        plt.grid(color='grey', linestyle='-', linewidth=1, alpha=0.5)
        plt.xticks(np.arange(0,256,20))
        plt.yticks(np.arange(0,1,0.1))
        plt.show()
        ```

    Args:
        length (int): the length of corresponding sent / doc

    Returns:
        float: threshold
    """
    y = 10 / (length + 10)
    return y