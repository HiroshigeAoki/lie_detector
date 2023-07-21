import matplotlib
import matplotlib.cm
import matplotlib.colors
import torch
from collections import Counter
import os
import joblib
from tqdm import tqdm
import pandas as pd
from collections import Counter
from omegaconf import OmegaConf


# colormaps: https://matplotlib.org/stable/tutorials/colors/colormaps.html
def plot_attentions(doc: list[str], word_weights: list[torch.tensor], sent_weights: list[torch.tensor], pad_sent_num: torch.tensor,
                    word_cmap="Blues", sent_cmap="Reds", word_color_level=1000, sent_color_level=1000, size: int = 4,
                    ignore_tokens=['[PAD]', '[SEP]', '[CLS]', '[UNK]', '.', ' '], pad_token='[PAD]', n_gram='uni') -> str:

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

        sent_color = matplotlib.colors.rgb2hex(sent_cmap((sent_weight.numpy(
        ) - sent_threshold) * sent_color_level)[:3]) if sent_weight.numpy() > sent_threshold else "#FFFFFF"
        colored_doc += template_sent.format(size,
                                            sent_color, "&nbsp" + '  ' + "&nbsp")

        for token, word_weight in zip(sent, _word_weights):
            if token in ignore_tokens:
                continue
            if '<' in token and '>' in token:
                token = token.replace('<', '＜')
                token = token.replace('>', '＞')
            token = token.replace('#', '')
            tokens.append(token)
            weights.append(word_weight.numpy())

        if n_gram == 'uni':
            word_threshold = adjust_threshold(length=len(tokens))
        elif n_gram == 'bi':
            word_threshold = adjust_threshold(length=(len(tokens)-1) / 2)
        elif n_gram == 'tri':
            word_threshold = adjust_threshold(length=(len(tokens)-2) / 3)
        else:
            raise ValueError(f"'{n_gram}-gram' is not supported.")

        # word_threshold = 0.05
        for i, (token, weight) in enumerate(zip(tokens, weights)):
            if n_gram == 'bi':
                if i+1 == len(tokens):
                    break
                token += tokens[i+1]
                weight += weights[i+1]
            elif n_gram == 'tri':
                if i+2 >= len(tokens):
                    break
                token += tokens[i+1] + tokens[i+2]
                weight += weights[i+1] + weights[i+2]
            #if weight > word_threshold:
            word_color = matplotlib.colors.rgb2hex(
                word_cmap((weight - word_threshold) * word_color_level)[:3])
            vital_word_list_freq.append(token)
            vital_word_dict_weight[token] = vital_word_dict_weight.get(
                token, 0) + (weight - word_threshold)
            #else:
            #    word_color = "#FFFFFF"
            colored_doc += template_word.format(size, word_color, token)

        colored_doc += '</br>'

    vital_word_count_freq = Counter(vital_word_list_freq)
    sorted_vital_words_freq = vital_word_count_freq.most_common()
    vital_word_table_freq = ''
    for token, freq in sorted_vital_words_freq:
        vital_word_table_freq += template_vital_word_freq.format(size, token, freq)

    sorted_vital_words_weight = sorted(
        vital_word_dict_weight.items(), key=lambda x: x[1], reverse=True)
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


def make_ploted_doc(i, input_ids, word_weights, sent_weights, pad_sent_num, prob, pred, label, tokenizer, save_dir, kwargs):
    doc = [list(map(lambda x: x.replace(' ', ''), tokenizer.batch_decode(
                ids.tolist()))) for ids in input_ids]
    ploted_doc, vital_word_count_freq, vital_word_count_weight = plot_attentions(
        doc=doc, word_weights=word_weights, sent_weights=sent_weights, pad_sent_num=pad_sent_num, **kwargs)
    table_of_contents_list = []
    if pred == label:
        if label == 1:
            pred_class = 'TP'
            # DV stands for Degree of Conviction
            file_name = f'DC:{prob[label] * 100:.2f}% No.{i}.html'
            table_of_contents_list.extend(('TP', file_name))
        elif label == 0:
            pred_class = 'TN'
            file_name = f'DC:{prob[label] * 100:.2f}% No.{i}.html'
            table_of_contents_list.extend(('TN', file_name))
    elif pred != label:
        if label == 1:
            pred_class = 'FP'
            file_name = f'DC:{prob[pred] * 100:.2f}% No.{i}.html'
            table_of_contents_list.extend(('FP', file_name))
        elif label == 0:
            pred_class = 'FN'
            file_name = f'DC:{prob[pred] * 100:.2f}% No.{i}.html'
            table_of_contents_list.extend(('FN', file_name))
    with open(os.path.join(save_dir, pred_class, file_name), 'w') as f:
        f.write(ploted_doc)
    return table_of_contents_list, vital_word_count_freq, vital_word_count_weight, pred_class, prob[pred]


def list_to_csv(pred_class_confidence, vital_words, mode, save_dir):
    if len(vital_words) == 0:
        return

    if mode == 'freq':
        df = pd.DataFrame([{'token': token, 'freq': freq}
                           for token, freq in Counter(vital_words).most_common()])
    elif mode == 'weight':
        df = pd.DataFrame.from_dict(vital_words, orient='index').reset_index().rename(
            columns={'index': 'token', 0: 'weight'}).sort_values(by='weight', ascending=False)
    df.to_csv(os.path.join(
        save_dir, f'csv/{mode}/{pred_class_confidence}_vital_word_{mode}.csv'), index=False)


def create_html(cfg, tokenizer, outputs):

    if cfg.model.name == 'HAN':
        logits = torch.cat([p['logits'] for p in outputs], dim=0)
        word_attentions = torch.cat(
            [p['word_attentions'] for p in outputs]).cpu()
        sent_attentions = torch.cat(
            [p['sent_attentions'].squeeze(2) for p in outputs]).cpu()
        input_ids = torch.cat([p['input_ids'] for p in outputs]).cpu()
        pad_sent_num = torch.cat([p['pad_sent_num']
                                  for p in outputs]).cpu()

        labels = torch.cat([p['labels'] for p in outputs]).cpu()

    save_dir = f'plot_attention_{cfg.tokenizer.plot_attention.n_gram}-gram'

    # label 1: deceptive role, label 0: deceived role
    os.makedirs(os.path.join(save_dir, 'TP'),
                exist_ok=True)  # preds: 1, label: 1
    os.makedirs(os.path.join(save_dir, 'TN'),
                exist_ok=True)  # preds: 0, label: 0
    os.makedirs(os.path.join(save_dir, 'FP'),
                exist_ok=True)  # preds: 1, label: 0
    os.makedirs(os.path.join(save_dir, 'FN'),
                exist_ok=True)  # preds: 0, label: 1

    softmax = torch.nn.Softmax(dim=1)
    probs = softmax(logits).cpu()
    # preds = logits.argmax(dim=1).cpu()
    preds = (probs[:, 1] > 0.8).long()
    

    list_args = [(i, *args) for i, args in enumerate(zip(input_ids,
                                                         word_attentions, sent_attentions, pad_sent_num, probs, preds, labels))]

    outputs = joblib.Parallel(n_jobs=4, backend="threading")(
        joblib.delayed(make_ploted_doc)(
            *args,
            tokenizer=tokenizer,
            save_dir=save_dir,
            kwargs=OmegaConf.to_container(
                cfg.tokenizer.plot_attention),
        ) for args in tqdm(list_args, desc='making ploted doc')
    )

    template = '<td><a href="{}">{}</a></td>'

    table_of_contents = dict(TP=[], TN=[], FP=[], FN=[])

    vital_word_count_dict_freq = {}
    vital_word_count_dict_weight = {}

    for output in outputs:
        tc, vital_word_count_freq, vital_word_count_weight, pred_class, prob = output
        table_of_contents.get(tc[0]).append(tc[1])
        if prob >= 0.9:
            confidence = 90
        elif 0.9 > prob >= 0.8:
            confidence = 80
        elif 0.8 > prob >= 0.7:
            confidence = 70
        else:
            confidence = '60_50'
            # vital_word_count_dict_freq[f'{pred_class}_{confidence}'].extend(vital_word_count_freq)
        vital_word_count_dict_freq[f'{pred_class}_{confidence}'] = vital_word_count_dict_freq.get(
            f'{pred_class}_{confidence}', []) + vital_word_count_freq
        vital_word_count_dict_weight[f'{pred_class}_{confidence}'] = vital_word_count_dict_weight.get(
            f'{pred_class}_{confidence}', Counter()) + Counter(vital_word_count_weight)

    par_link = [template.format(f'./{key}.html', key)
                for key in table_of_contents.keys()]
    with open(os.path.join(save_dir, 'index.html'), 'w') as f:
        f.write('<ui>')
        for link in par_link:
            f.write('<li>' + link + '</li>')
        f.write('</ui>')

    for key, file_names in table_of_contents.items():
        file_names = sorted(file_names, reverse=True)
        chi_link = [template.format(
                    f'./{key}/{file_name}', file_name) for file_name in file_names]
        with open(os.path.join(save_dir, f'{key}.html'), 'w') as f:
            f.write('<ui>')
            for link in chi_link:
                f.write('<li>' + link + '</li>')
            f.write('</ui>')

    os.makedirs(os.path.join(save_dir, 'csv/freq'), exist_ok=True)

    joblib.Parallel(n_jobs=1)(
        joblib.delayed(list_to_csv)(
            *args,
            save_dir=save_dir,
            mode='freq'
        ) for args in tqdm(vital_word_count_dict_freq.items(), desc='making vital_word_count_freq.csv')
    )

    os.makedirs(os.path.join(save_dir, 'csv/weight'), exist_ok=True)
    joblib.Parallel(n_jobs=1)(
        joblib.delayed(list_to_csv)(
            *args,
            save_dir=save_dir,
            mode='weight'
        ) for args in tqdm(vital_word_count_dict_weight.items(), desc='making vital_word_count_weight.csv')
    )
    # gmail_sender.send(body=f"{cfg.mode} was finished.")
