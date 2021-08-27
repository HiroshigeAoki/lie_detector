# -*- coding: utf-8 -*-
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from tqdm import tqdm
from pathlib import Path
import copy
from IPython.display import display, HTML
from contextlib import redirect_stdout
import sys, os
sys.path.append(os.pardir)
from visualization.load_model import load_model
from utils.data_preparation import create_dl
from utils.unix_command import mkdir, mkdirs
from visualization.plot_attention import plot_attentions
from visualization.unify_attention import average_attentions, average_last_layer_attentions



# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
n_cpu = os.cpu_count()


def get_texts_attens_line_volumepage(dl, model, tokenizer, device, attention_type, temp):
    preds, texts, attentions = [], [], []

    with torch.no_grad():
        for batch in tqdm(dl, desc="predicting..."):
            vecs, a_masks = batch[0].to(device), batch[1].to(device)

            output = model(vecs, attention_mask=a_masks)

            preds.extend(output.logits)
            if attention_type == "average":
                attentions.extend(average_attentions(output.attentions, temp=temp))
            elif attention_type == "last":
                attentions.extend(average_last_layer_attentions(output.attentions, temp=temp))
            else:
                raise ValueError(f"{attention_type} is invalid attention_type.")

            for vec in vecs:
                texts.append([tokenizer.decode(id_).replace("#", "").replace(" ", "").replace(".", "") for id_ in vec])
        return torch.stack(preds), texts, torch.stack(attentions)


def line_concat(ploted_text, lines, volume_pages):
    concated = []
    tmp = list()
    for i, (text, line, volume_page) in enumerate(zip(ploted_text, lines, volume_pages)):
        if i != 0 and (line != lines[i - 1] or volume_page != volume_pages[i - 1]):
            concated.append([copy.copy(tmp), lines[i - 1], volume_pages[i - 1]])
            tmp.clear()
        tmp.append(text)
    return concated


def mk_html(save_dir: Path, ploted_lines_with_inf: list[list, int, str], temp: float, sample: bool, threshold: float, color_level: float):
    for line_inf in tqdm(ploted_lines_with_inf, desc="HTMLを作成中..."):
        line, n_line, vol_page = line_inf[0], line_inf[1], line_inf[2]
        output_file = f"{vol_page}_heatmap.html"
        save_path = save_dir / output_file
        inf_temp = '<font face="monospace" \nsize="4"; span class="inf"; style="color: black">{} {}行目</span></font>'
        if not sample:
            output_HTML(inf_temp.format(n_line, vol_page), "</br>", "".join(line), "</br>", path=save_path)
        else:
            output_HTML(f"temperature: {temp}</br>threshold: {threshold}</br> color_level:{color_level}</br>",
                        inf_temp.format(n_line, vol_page), "</br>", "".join(line), "</br>", path=save_path)

        with redirect_stdout(open(os.devnull, 'w')):
            display(HTML(f"{save_path}"))


def mk_html_inf(save_dir, preds, ploted_sentences, n_lines, vol_pages):
    category_dict = {0: "ごまかし表現でない", 1: "ごまかし表現"}
    for pred, sentence, n_line, vol_page in tqdm(zip(preds, ploted_sentences, n_lines, vol_pages), desc="HTMLを作成中..."):
        output_file = f"{vol_page}_heatmap.html"
        save_path = save_dir / output_file
        inf_temp = '<font face="monospace" \nsize="4"; span class="inf"; style="color: black">{}行目 {}' \
                   '</br>  {}</span></font>'
        prediction = np.argmax(pred.numpy())
        pred_inf = f"確率{round(float(pred[prediction]) * 100, 1)}%で{category_dict.get(prediction)}{'である' if prediction == 1 else ''}と予測。"
        output_HTML(inf_temp.format(n_line, vol_page, pred_inf), "</br>", sentence, "</br>", path=save_path)


def output_HTML(*contents, path):
    with open(path, 'a', encoding="iso-2022-jp", errors='replace') as f:
        for content in contents:
            f.write(str(content))
        f.write("</br>")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="aozora_wiki_bert")
    parser.add_argument("--yoshinobu", action='store_true')
    parser.add_argument("--attention_type", type=str, default="average", help="average or last")
    parser.add_argument("--sample", action='store_true')
    parser.add_argument("--inf", action='store_true')
    parser.add_argument("--color_level", type=float, default=4, help="adjust the intensity of color")
    parser.add_argument("--temp", type=float, default=0.02, help="adjust contrast of attention")
    parser.add_argument("--threshold", type=float, default=1e-4, help="words under this threshold will be white background")
    batch_size = 1024

    args = parser.parse_args()

    device = torch.device('cuda')

    model_name = args.model_name

    log.info(f"model name -> {args.model_name}")
    log.info(f"mode -> {'yoshinobu' if args.yoshinobu else 'othres'}")
    log.info(f"attention type -> {args.attention_type}")
    log.info(f"color_level -> {args.color_level}")
    log.info(f"temp -> {args.temp}")
    log.info(f"threshold -> {args.threshold}")
    log.info(f"device -> {device}")
    log.info(f"num_cpu -> {n_cpu}")
    log.info(f"batch_size -> {batch_size}")

    data_dir = Path("../datasets/")
    if args.inf:
        save_dir = Path("./HTML", f"{model_name}", "heatmap_with_inf", f"{args.attention_type}")
    else:
        save_dir = Path("./HTML", f"{model_name}", "heatmap", f"{args.attention_type}")
    mkdirs(save_dir)

    if args.yoshinobu:
        data_path = data_dir / "aimai_aozora_wiki_y.pickle"
        save_dir = save_dir / "yoshinobu"
        mkdir(save_dir)
    else:
        data_path = data_dir / "aimai_aozora_wiki_others.pickle"
        save_dir = save_dir / "others"
        mkdir(save_dir)

    log.info("データをロードしています...")
    ds, lines, volume_pages = pickle.load(open(data_path, 'rb'))

    # for test
    if args.sample:
        data_size = 100
        ds = Subset(ds, list(range(data_size)))

    dl = create_dl(ds=ds, batch_size=batch_size, is_train=False)

    log.info("モデルをロードしています...")
    model, tokenizer = load_model(model_name=model_name)
    model = torch.nn.DataParallel(model)
    model.eval()
    model.to(device)

    log.info(f"{'慶喜' if args.yoshinobu else '慶喜以外'}の文、{len(ds)}文をモデルに入力")

    preds, text, attentions = get_texts_attens_line_volumepage(dl=dl, model=model, tokenizer=tokenizer, device=device,
                                                        attention_type=args.attention_type, temp=args.temp)
    preds = F.softmax(preds, dim=1).cpu()

    plotted_text = plot_attentions(text, attentions, color_level=args.color_level, threshold=args.threshold)

    if args.inf:
        mk_html_inf(save_dir, preds, plotted_text, lines, volume_pages)
    else:
        plotted_text = line_concat(plotted_text, lines, volume_pages) if not args.sample \
            else line_concat(plotted_text, lines[:data_size], volume_pages[:data_size])
        mk_html(save_dir, plotted_text, temp=args.temp, sample=args.sample, threshold=args.threshold, color_level=args.color_level)

    log.info("完了！")


if __name__ == '__main__':
    from utils import logger
    log = logger.Logger("mk_heatmap.py")
    main()
