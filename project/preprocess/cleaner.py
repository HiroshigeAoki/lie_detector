import re
from typing import Tuple
import torch
import numpy as np
import os, sys
import neologdn
import demoji

sys.path.append(os.pardir)
from model.load_model import load_model

# TODO: load_modelをこのファイルに書いちゃう

def clean_sent(sent: str) -> str:
    sent = neologdn.normalize(sent) # 表記揺れを無くす
    sent = demoji.replace(string=sent, repl="") # 絵文字削除
    sent = re.sub((r'[\(@][^あ-ん\u30A1-\u30F4\u2E80-\u2FDF\u3005-\u3007\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\U00020000-\U0002EBEF]+?\)'), '', sent) # 顔文字削除
    sent = sent.replace('^^', '')
    sent = re.sub(r'[\t\n\r]', '', sent) # remove tab, newline
    sent = re.sub(r'[>]+[0-9]+', '', sent) # remove res anker
    sent = re.sub(r'[@][0-9]+', '', sent) # remove res anker
    #null_bite = re.compile(r'\x00')
    URL = re.sub(r'https?:?//[-_.!~*\'()a-zA-Z0-9;/?:@&=+$,%#]+', '', sent) # remove URL
    sent = re.sub(r'[\u0900-\u0FFF\u2500-\u2E52]', '', sent) #記号の削除
    sent = re.sub(r'[0-9,]+[0-9]+', '0', sent) # numbers to 0
    sent = re.sub(r'[0-9]+\.?[0-9]*', '0', sent) # decimal to 0
    #sent = re.sub(r'[\n\r\n]', '<br>', str(sent))
    return sent


def replace_term(sent: str) -> str:
    names = "行商人 アルビン|アル(ビン)*|行商人|商人|村長 ヴァルター|そんちょ|村長|ヴァル(ター)*|仕立て屋 エルナ|エルナ*|エレナ|エ*ルナ|パン屋 オットー|オットー*|オト|パン屋|羊飼い カタリナ|カタ[(リナ)(りん)]*|(カタ)*リナ|羊飼い|司書 クララ|クララ*|ク*ララ|司書|楽天家 ゲルト|ゲルト*|楽天家|神父 ジムゾン|ジム(ゾン)*|神父|負傷兵 シモン|シモン*|兵隊|負傷兵|ならず者 ディーター|Ｄ太|ディー*タ*ー*|ならず者|木こり トーマス|トー*マス*|トム|機関車トー○ス|肉妖精|木こり|旅人 ニコラス|スナフキン|薄緑|旅人|ニコ[(ラス)(ルン)]*|村娘 パメラ|村娘|娘|パメラ*|シスター フリーデル|シスター|フリ(ーデル)*|リデル|尼|少年 ペーター|ピーター|ベーター|ペー太|ペタ|少年|ペーター|老人 モーリッツ|おじいちゃん|お爺さん|(モー)*爺|(モリ)*爺|翁|長老|老人|モーリッツ|農夫 ヤコブ|ヤコ[ブ|ビン|ぷー]*|やこびー|農夫|青年 ヨアヒム|ヨア(ヒム)*|よあひー|青年|少女 リーザ|リ[ーズ]*ザ*|少女|宿屋の女主人 レジーナ|おばさん|マダム|レジ(ーナ)*|姐さん|姐御|宿屋の女主人|女将|小母様|レジーナ"
    pattern = re.compile(names)
    sent = re.sub(pattern, '<person>', sent)

    return sent


class auto_exclude_sent():
    def __init__(self):
        #load model
        self.model, self.tokenizer = load_model('sentence_classifier_RoBERTa')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.utter_type = {0: "CO", 1: "guard", 2: "inspect", 3: "other"}

    def __call__(self, sent: str) -> Tuple[bool, str]:
        encoded = self.tokenizer(sent, padding='max_length', max_length=128, truncation=True, return_tensors='pt')
        encoded = encoded.to(self.device)
        preds = self.model(encoded['input_ids'], encoded['attention_mask'])
        result = np.argmax(preds.logits.cpu().detach().numpy())
        return (result!=3, self.utter_type.get(result))