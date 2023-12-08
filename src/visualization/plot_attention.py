import os
from collections import Counter
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.cm
import matplotlib.colors
from typing import Tuple


def n_gram_to_int(n_gram: str) -> int:
    n_gram_dict = {'uni': 1, 'bi': 2, 'tri': 3}
    return n_gram_dict.get(n_gram, 0)

def adjust_threshold(length: int) -> float:
    return 1 / (length + 1e-100)

def color_mapping(weight: float, cmap: str, threshold: float, color_level: int) -> str:
    if weight > threshold:
        cmap_obj = matplotlib.cm.get_cmap(cmap)
        rgb_values = cmap_obj((weight - threshold) * color_level)[:3]
        return matplotlib.colors.rgb2hex(rgb_values)
    else:
        return "#FFFFFF"

def prepare_word_and_weight(sent: list[str], _word_weights: list[torch.tensor], ignore_tokens: set[str]) -> Tuple:
    tokens, weights = zip(*((token.replace('<', '＜').replace('>', '＞').replace('#', ''), weight.numpy())
                            for token, weight in zip(sent, _word_weights) if token not in ignore_tokens))
    return tokens, weights

def update_word_and_weight(n_gram: int, tokens: list[str], weights: list[float], word_threshold: float) -> Tuple:
    vital_word_list_freq = []
    vital_word_dict_weight = {}
    for i, (token, weight) in enumerate(zip(tokens, weights)):
        if n_gram in [2, 3] and i+n_gram >= len(tokens):
            break
        if n_gram != 1:
            token += ''.join(tokens[i+1:i+(n_gram-1)])
            weight += sum(weights[i+1:i+n_gram-1])
        if weight > word_threshold:
            vital_word_list_freq.append(token)
            vital_word_dict_weight[token] = vital_word_dict_weight.get(token, 0) + weight - word_threshold
    return vital_word_list_freq, vital_word_dict_weight

def plot_attentions(doc: list[str], word_weights: list[torch.tensor], sent_weights: list[torch.tensor], pad_sent_num: torch.tensor,
                    word_cmap="Blues", sent_cmap="Reds", word_color_level=1000, sent_color_level=1000, size: int = 4,
                    ignore_tokens={'[PAD]', '[SEP]', '[CLS]', '[UNK]', '.', ' '}, pad_token='[PAD]', n_gram='uni') -> str:

    template = '<font face="monospace" size="{}"; span class="barcode"; style="color: black; background-color: {}">{}</span></font>'
    template_vital_word_freq = '<li><font face="monospace" size="{}"; span class="barcode"; style="color: black">{}・・・{}回</span></font></li>'
    template_vital_word_weight = '<li><font face="monospace" size="{}"; span class="barcode"; style="color: black">{}・・・{:.4f}</span></font></li>'

    sent_threshold = adjust_threshold(len(doc) - int(pad_sent_num))
    colored_doc, vital_word_list_freq, vital_word_dict_weight = "", [], {}

    for sent, _word_weights, sent_weight in zip(doc, word_weights, sent_weights):
        if sent[0] == pad_token:
            break

        colored_doc += template.format(size, color_mapping(sent_weight.numpy(), sent_cmap, sent_threshold, sent_color_level), "&nbsp" + '  ' + "&nbsp")

        tokens, weights = prepare_word_and_weight(sent, _word_weights, ignore_tokens)

        if n_gram in ['uni', 'bi', 'tri']:
            word_threshold = adjust_threshold(length=(len(tokens)-(n_gram_to_int(n_gram) - 1)) / n_gram_to_int(n_gram))
        else:
            raise ValueError(f"'{n_gram}-gram' is not supported.")

        list_freq, dict_weight = update_word_and_weight(n_gram_to_int(n_gram), tokens, weights, word_threshold)
        vital_word_list_freq.extend(list_freq)
        vital_word_dict_weight.update(dict_weight)

        for token, weight in zip(tokens, weights):
            colored_doc += template.format(size, color_mapping(weight, word_cmap, word_threshold, word_color_level), token)

        colored_doc += '</br>'

    vital_word_table_freq = ''.join(template_vital_word_freq.format(size, token, freq) for token, freq in Counter(vital_word_list_freq).most_common())
    vital_word_table_weight = ''.join(template_vital_word_weight.format(size, token, weight) for token, weight in sorted(vital_word_dict_weight.items(), key=lambda x: x[1], reverse=True))

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


class HtmlPlotter:
    def __init__(self, cfg, tokenizer, outputs):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.outputs = outputs

        self.save_dir = f'plot_attention_{cfg.tokenizer.plot_attention.n_gram}-gram'

        for folder in ['TP', 'TN', 'FP', 'FN', 'csv/freq', 'csv/weight']:
            os.makedirs(os.path.join(self.save_dir, folder), exist_ok=True)

    def create_html(self):
        if self.cfg.model.name == 'HAN':
            logits, word_attentions, sent_attentions, input_ids, pad_sent_num, labels = self._gather_outputs()

        logits = logits.float()
        probs = F.softmax(logits, dim=-1).cpu()

        preds = logits.argmax(dim=-1).cpu()
        # preds = (probs[:, 1] > 0.8).long()

        list_args = [(i, *args) for i, args in enumerate(zip(input_ids,
                                                             word_attentions, sent_attentions, pad_sent_num, probs, preds, labels))]

        kwargs = OmegaConf.to_container(self.cfg.tokenizer.plot_attention)
        outputs = Parallel(n_jobs=4, backend="threading")(
            delayed(self.make_ploted_doc)(
                *args,
                tokenizer=self.tokenizer,
                save_dir=self.save_dir,
                kwargs=kwargs
            ) for args in tqdm(list_args, desc='making ploted doc')
        )

        self._generate_html_files(outputs)
        self._generate_csv_files(outputs)

    def _gather_outputs(self):
        logits = torch.stack([p['logits'] for p in self.outputs]).cpu().squeeze()
        word_attentions = torch.stack([p['word_attentions'] for p in self.outputs]).cpu().squeeze()
        sent_attentions = torch.stack([p['sent_attentions'].squeeze(2) for p in self.outputs]).cpu().squeeze()
        input_ids = torch.stack([p['input_ids'] for p in self.outputs]).cpu().squeeze()
        pad_sent_num = torch.stack([p['pad_sent_num'] for p in self.outputs]).cpu()
        labels = torch.stack([p['labels'] for p in self.outputs]).cpu()
        return logits, word_attentions, sent_attentions, input_ids, pad_sent_num, labels

    def make_ploted_doc(self, i, input_ids, word_weights, sent_weights, pad_sent_num, prob, pred, label, tokenizer, save_dir, kwargs):
        doc = [list(map(lambda x: x.replace(' ', ''), tokenizer.batch_decode(
                    ids.tolist()))) for ids in input_ids]
        ploted_doc, vital_word_count_freq, vital_word_count_weight = plot_attentions(
            doc=doc, word_weights=word_weights, sent_weights=sent_weights, pad_sent_num=pad_sent_num, **kwargs)
        table_of_contents_list = []
        if pred == label:
            if label == 1:
                pred_class = 'TP'
                # DV stands for Degree of Conviction
                file_name = f'DC:{prob[label].item() * 100:.2f}% No.{i}.html'
                table_of_contents_list.extend(('TP', file_name))
            elif label == 0:
                pred_class = 'TN'
                file_name = f'DC:{prob[label].item() * 100:.2f}% No.{i}.html'
                table_of_contents_list.extend(('TN', file_name))
        elif pred != label:
            if label == 1:
                pred_class = 'FP'
                file_name = f'DC:{prob[pred].item() * 100:.2f}% No.{i}.html'
                table_of_contents_list.extend(('FP', file_name))
            elif label == 0:
                pred_class = 'FN'
                file_name = f'DC:{prob[pred].item() * 100:.2f}% No.{i}.html'
                table_of_contents_list.extend(('FN', file_name))
        with open(os.path.join(save_dir, pred_class, file_name), 'w') as f:
            f.write(ploted_doc)
        return table_of_contents_list, vital_word_count_freq, vital_word_count_weight, pred_class, prob[pred]

    def list_to_csv(self, pred_class_confidence, vital_words, mode, save_dir):
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

    def _generate_html_files(self, outputs):
        template = '<td><a href="{}">{}</a></td>'
        table_of_contents = dict(TP=[], TN=[], FP=[], FN=[])

        for output in outputs:
            tc, vital_word_count_freq, vital_word_count_weight, pred_class, prob = output
            table_of_contents.get(tc[0]).append(tc[1])

        self._write_index_html(template, table_of_contents)
        self._write_sub_html(template, table_of_contents)

    def _write_index_html(self, template, table_of_contents):
        par_link = [template.format(f'./{key}.html', key) for key in table_of_contents.keys()]
        with open(os.path.join(self.save_dir, 'index.html'), 'w') as f:
            f.write('<ui>')
            for link in par_link:
                f.write('<li>' + link + '</li>')
            f.write('</ui>')

    def _write_sub_html(self, template, table_of_contents):
        for key, file_names in table_of_contents.items():
            file_names = sorted(file_names, reverse=True)
            chi_link = [template.format(f'./{key}/{file_name}', file_name) for file_name in file_names]
            with open(os.path.join(self.save_dir, f'{key}.html'), 'w') as f:
                f.write('<ui>')
                for link in chi_link:
                    f.write('<li>' + link + '</li>')
                f.write('</ui>')

    def _generate_csv_files(self, outputs):
        vital_word_count_dict_freq = {}
        vital_word_count_dict_weight = {}

        for output in outputs:
            tc, vital_word_count_freq, vital_word_count_weight, pred_class, prob = output
            confidence = self._get_confidence(prob)

            vital_word_count_dict_freq[f'{pred_class}_{confidence}'] = vital_word_count_dict_freq.get(
                f'{pred_class}_{confidence}', []) + vital_word_count_freq
            vital_word_count_dict_weight[f'{pred_class}_{confidence}'] = vital_word_count_dict_weight.get(
                f'{pred_class}_{confidence}', Counter()) + Counter(vital_word_count_weight)

        Parallel(n_jobs=1)(
            delayed(self.list_to_csv)(
                *args,
                save_dir=self.save_dir,
                mode='freq'
            ) for args in tqdm(vital_word_count_dict_freq.items(), desc='making vital_word_count_freq.csv')
        )

        Parallel(n_jobs=1)(
            delayed(self.list_to_csv)(
                *args,
                save_dir=self.save_dir,
                mode='weight'
            ) for args in tqdm(vital_word_count_dict_weight.items(), desc='making vital_word_count_weight.csv')
        )

    def _get_confidence(self, prob):
        if prob >= 0.9:
            return 90
        elif 0.9 > prob >= 0.8:
            return 80
        elif 0.8 > prob >= 0.7:
            return 70
        else:
            return '60_50'
