import matplotlib
import matplotlib.cm
import matplotlib.colors
import torch
from collections import Counter

# colormaps: https://matplotlib.org/stable/tutorials/colors/colormaps.html
def plot_attentions(doc: list[str], word_weights: list[torch.tensor], sent_weights: list[torch.tensor], pad_sent_num: torch.tensor,
                        word_cmap="Blues" , sent_cmap="Reds", word_color_level=1000, sent_color_level=1000, size: int = 4,
                        ignore_tokens = ['[PAD]', '[SEP]', '[CLS]', '[UNK]', '.', ' '], pad_token='[PAD]', n_gram='uni') -> str:

    colored_doc = ""
    word_cmap = matplotlib.cm.get_cmap(word_cmap)
    sent_cmap = matplotlib.cm.get_cmap(sent_cmap)
    template_word = '<font face="monospace" \nsize="{}"; span class="barcode"; style="color: black; background-color: {}">{}</span></font>'
    template_sent = '<font face="monospace" \nsize="{}"; span class="barcode"; style="color: black; background-color: {}">{}</span></font>'

    vital_word_list_freq = []
    vital_word_dict_weight = {}
    template_vital_word_freq = '<li><font face="monospace" \nsize="{}"; span class="barcode"; style="color: black">{}・・・{}回</span></font></li>'
    template_vital_word_weight = '<li><font face="monospace" \nsize="{}"; span class="barcode"; style="color: black">{}・・・{:.4f}</span></font></li>'

    sent_threshold = adjust_threshold(len(doc) - int(pad_sent_num))

    for sent, _word_weights, sent_weight in zip(doc, word_weights, sent_weights):
        tokens, weights = [], []
        if sent[0] == pad_token:
            break

        sent_color = matplotlib.colors.rgb2hex(sent_cmap((sent_weight.numpy() - sent_threshold) * sent_color_level)[:3]) if sent_weight.numpy() > sent_threshold else "#FFFFFF"
        colored_doc += template_sent.format(size, sent_color, "&nbsp" + '  ' + "&nbsp")

        for token, word_weight in zip(sent, _word_weights):
            if token in ignore_tokens:
                continue
            if '<' in token and '>' in token:
                token = token.replace('<', '＜')
                token = token.replace('>', '＞')
            token = token.replace('#', '')
            tokens.append(token)
            weights.append(word_weight.numpy())

        if n_gram=='uni':
            word_threshold = adjust_threshold(length=len(tokens))
        elif n_gram=='bi':
            word_threshold = adjust_threshold(length=(len(tokens)-1) / 2)
        elif n_gram=='tri':
            word_threshold = adjust_threshold(length=(len(tokens)-2) / 3)
        else:
            raise ValueError(f"'{n_gram}-gram' is not supported.")

        #word_threshold = 0.05
        for i, (token, weight) in enumerate(zip(tokens, weights)):
            if n_gram=='bi':
                if i+1 == len(tokens):
                    break
                token += tokens[i+1]
                weight += weights[i+1]
            elif n_gram=='tri':
                if i+2 >= len(tokens):
                    break
                token += tokens[i+1] + tokens[i+2]
                weight += weights[i+1] + weights[i+2]
            if weight > word_threshold:
                word_color = matplotlib.colors.rgb2hex(word_cmap((weight - word_threshold) * word_color_level)[:3])
                vital_word_list_freq.append(token)
                vital_word_dict_weight[token] = vital_word_dict_weight.get(token, 0) + (weight - word_threshold)
            else:
                word_color =  "#FFFFFF"
            colored_doc += template_word.format(size, word_color, token)

        colored_doc += '</br>'

    vital_word_count_freq = Counter(vital_word_list_freq)
    sorted_vital_words_freq = vital_word_count_freq.most_common()
    vital_word_table_freq = ''
    for token, freq in sorted_vital_words_freq:
        vital_word_table_freq += template_vital_word_freq.format(size, token, freq)

    sorted_vital_words_weight = sorted(vital_word_dict_weight.items(), key= lambda x: x[1], reverse=True)
    vital_word_table_weight = ''
    for token, weight in sorted_vital_words_weight:
        vital_word_table_weight += template_vital_word_weight.format(size, token, weight)

    colored_doc = (
        f'<h1>重要単語リスト(出現頻度順)</h1><ol>{vital_word_table_freq}</ol>'
        +
        '<hr style="border:0;border-top:medium solid black;">'
        +
        f'<h1>重要単語リスト(重み順)</h1><ol>{vital_word_table_weight}</ol>'
        +
        '<hr style="border:0;border-top:medium solid black;">'
        +
        '<h1>本文</h1>'
        +
        colored_doc
    )
    return colored_doc, vital_word_list_freq, vital_word_dict_weight

def adjust_threshold(length: int) -> float:
    """adjust threshold according to lengths

    Shape:
        ```
        import matplotlib.pyplot as plt
        import numpy as np
        import math
        a = 0.5
        x = np.linspace( 1, 256, 256)
        y = a / x
        plt.plot(x, y)
        plt.title(f'y = {a} / x')
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
    y = 1 / (length + 1e-100)
    return y