import re
import neologdn
import demoji
import regex


def clean_sent(sent: str) -> str:
    sent = neologdn.normalize(sent, repeat=3) # 表記揺れを無くす
    sent = sent.lower()
    sent = demoji.replace(string=sent, repl="") # 絵文字削除
    sent = re.sub((r'[\(@][^あ-ん\u30A1-\u30F4\u2E80-\u2FDF\u3005-\u3007\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\U00020000-\U0002EBEF]+?\)'), '', sent) # 顔文字削除
    sent = re.sub(r"[m|c|u|o|つ|っ|ノ|シ|ヽ|人|⊂|＜|＞|<|>|≦|≧|≲|≳|≪|≫|《|》|\^|⌍|♪|＾|⌒|@|σ|α|β|∂|≡|=|-|ー|~|～|ﾟ|・|·|་|\*|＊|'|`|｀|´|❛|;|:|!|\?|,|\.|\(|\)|\||｜|￥|\\|\/|＼|／]*(Д|д|ω|ᴗ|∇|_ _|σ|α|β)[m|c|u|o|つ|っ|ノ|シ|ヽ|人|⊂|＜|＞|<|>|≦|≧|≲|≳|≪|≫|《|》|\^|⌍|♪|＾|⌒|@|σ|α|β|∂|≡|=|-|ー|~|～|ﾟ|・|·|་|\*|＊|'|`|｀|´|❛|;|:|!|\?|,|\.|\(|\)|\||｜|￥|\\|\/|＼|／]*", '', sent) # 顔文字削除
    sent = re.sub(r'\^\^|orz', '', sent)
    sent = re.sub(r'[\t\n\r]', '', sent) # remove tab, newline
    sent = re.sub(r'[>]+[0-9]+', '', sent) # remove res anker
    sent = re.sub(r'[@][0-9]+', '', sent) # remove res anker
    #null_bite = re.compile(r'\x00')
    sent = re.sub(r'https?:?//[-_.!~*\'()a-zA-Z0-9;/?:@&=+$,%#]+', '', sent) # remove URL
    sent = re.sub(r'[\u0900-\u0FFF\u2500-\u2E52]', '', sent) #記号の削除
    sent = re.sub(r'[0-9,]+[0-9]+', '0', sent) # numbers to 0
    sent = re.sub(r'[0-9]+\.?[0-9]*', '0', sent) # decimal to 0
    sent = re.sub(r'[【】\[\]]', '', sent)
    sent = re.sub(r'0:0', '0時', sent)
    sent = re.sub(r'0d', '0日', sent)
    sent = re.sub(r'\|0\|0|0-0', '0', sent)
    return sent.strip()


def replace_term(sent: str) -> str:
    names = [
        "(?V1)行商人 アルビン|アルビン",
        "行商人|商人",
        "村長 ヴァルター|そんちょ|村長|ヴァル(ター)*|ヴェル",
        "仕立て屋 エルナ|エルナ|エルゥ|エレナ",
        "パン屋 オットー|オットー*|パン屋",
        "羊飼い カタリナ|カタリナ|カタりん|羊飼い",
        "司書 クララ|クララ|クラりん|司書|クララん",
        "楽天家 ゲルト|楽天家",
        "神父 ジムゾン|ジムゾ|ジム=ゾン|ジム!ゾン!|神父",
        "負傷兵 シモン|シモン*|兵隊|負傷兵",
        "ならず者 ディーター|Ｄ太|ディーター|ディッタン|ディダ|ならず者",
        "木こり トーマス|トー*マス*|トム|機関車トー○ス|肉妖精|木こり",
        "旅人 ニコラス|スナフキン|薄緑|旅人|ニコ(ラス|ルン)+|ニコラ",
        "村娘 パメラ|村娘|娘|パメラ*",
        "シスター フリーデル|シスター|リデル",
        "少年 ペーター|ペータ(ー)*|ピーター|ベーター|ペー太|ぺーた(ー|ん)*|ぺー(君|にゃん)|ペタ|少年|ペーター",
        "老人 モーリッツ|おじいちゃん|じーさん|じいちゃん|じじい|お爺さん|もーじ[い|ぃ]|長老|老人|モーリッツ|モーリツ|モっちん",
        "農夫 ヤコブ|ヤコ(ブ|ビン|ぷー)*|やこびー|農夫", "青年 ヨアヒム|ヨア(ヒム)*|ヨアヒ|よあひー|よあひむ|青年",
        "少女 リーザ|リ(ーズ|ーザ|ザ)+|少女",
        "宿屋の女主人 レジーナ|おばさん|マダム|レジ(ーナ)*|レジナ|姐さん|姐御|宿屋の女主人|宿屋のおかみ|女将|小母様|レジーナ"
    ]
    name_char = "楽|村|長|老|爺|翁|神|木|樵|旅|者|年|尼|少|妙|女|好|行|商|羊|屋|青|娘|農|宿|姐|修|尼|仕|服|司|書|兵|傷|負"
    nick_names = '|'.join([
        "エル",
        "ルナ",
        "オト",
        "カタ", "リナ",
        "クラ", "ララ",
        "ゲル",
        "ジム",
        "ディタ",
        "ディ",
        "フリ",
        "モリ", "モー",
    ])
    # TODO: モーリツ以降もチェック。

    particles = ["は","も","を","が","か","で","と","に","の","や","へ"]
    honorific_titles = [
        "姉", "ね[え|ぇ]", "お姉", "おね[え|ぇ]", "お嬢"
        "兄", "に[い|ぃ]", "お兄", "おに[え|ぇ]",
        "おじいちゃん", "爺", "翁", "じ[い|ぃ]",
        "さん","くん", "君", "ちゃん", "坊", "子",
        "様", "さま", "殿", "どの",
        "どん", "さぁ", "はん",
        "ち", "っち", "ぴょん", "ポン", "にゃん", "わん", "たん", "らん", "りん", "猫ちゃん", "ワンちゃん"
    ]
    person_particles = '|'.join([f"<person>{p}" for p in particles])
    person_honorific_titles = '|'.join([f"<person>{h}" for h in honorific_titles])
    particles = '|'.join(particles)
    honorific_titles = '|'.join(honorific_titles)
    role_names = "狼|人狼|羊|灰|偽|白|黒|真|占|狂|狩|霊|共"
    symbols = "→|-|⇒|or|=|>|≧|×|「|:|。|\(|\)|・|0"

    reg1 = regex.compile('|'.join(names))
    sent = regex.sub(reg1, '<person>', sent)

    ignore_prefix = "共有|能力|霊能|微|進|続|奇|この|確定|うちの|勝|色々|流|隣"
    ignore_suffix = "い|う|な|し|性|っ|[き-け]|騙り|の開始|に泊|器|品"
    # reg2 = regex.compile(rf"(?V1)(?<!({ignore_prefix}))({name_char}|{nick_name})(?=(どっち|的には|0|のように|視点|確定|確白|確黒|希望|両狼|確認|考察|襲撃|護衛|直吊|吊|即吊|ライン|連打|不在|パンダ|ぱんだ|墓|ごめん|ありがとう|食[っ|べ|う]|{symbols}|<person>|{role_names}|{name_char}|{particles}|{honorific_titles}|のco|の判定|の発言|の占い|の内心|[{particles}{symbols}][{name_char}|{role_names}]))")
    reg2 = regex.compile(rf"(?V1)(?<!({ignore_prefix}))({name_char}|{nick_names})(?=(どっち|的には|のように|視点|確定|確白|確黒|希望|両狼|確認|考察|襲撃|護衛|直吊|吊|即吊|ライン|連打|不在|パンダ|ぱんだ|墓|ごめん|ありがとう|食[っ|べ|う]|{symbols}|<person>|{role_names}|{name_char}|{particles}|{honorific_titles}|のco|の判定|の発言|の占い|の内心|([{particles}{symbols}]|{nick_names})[{name_char}|{role_names}]))")
    # reg3 = regex.compile(rf"(?V1)(?<=([{name_char}{role_names}][{particles}|{honorific_titles}]|<person>|{person_particles}|{person_honorific_titles}|{symbols}|{role_names}|{name_char}|希望出し))({name_char}|{nick_name})(?!({ignore_suffix}))")
    reg3 = regex.compile(rf"(?V1)(?<=((([{name_char}{role_names}]|{nick_names})[{particles}|{honorific_titles}]|<person>|{person_particles}|{person_honorific_titles}|{symbols}|{role_names}|{name_char}|希望出し))({name_char}|{nick_names})(?!({ignore_suffix}))")
    # TODO names/nickname particles/honorific_titlesでちゃんと変換されるかチェック。

    sent = regex.sub(r'旅行(?=(に|へ|でもいかない|でもいかない|行く|いく))', '&旅&行&', sent)
    sent = regex.sub(r'負傷(?=(した|して|がひどい|こそ|具合))', '&負&傷&', sent)
    sent = regex.sub(r'(?=(足|手|体))負傷', '&負&傷&', sent)
    sent = regex.sub(r'宿屋(?=(に|の))', '&宿&屋&', sent)
    sent = regex.sub(r'村人(?=(|なら|は|の|を|))', '&村&人&', sent)
    sent = regex.sub(r'村(?=(|のために|として|[は|の]負け|を滅ぼ|[は|の]勝))', '@村@', sent)

    sent = regex.sub(reg2, '<person>', sent)
    sent = regex.sub(reg3, '<person>', sent)

    sent = re.sub('&旅&行&', '旅行', sent)
    sent = re.sub('&負&傷&', '負傷', sent)
    sent = re.sub('&宿&屋&', '宿屋', sent)
    sent = re.sub('&村&人&', '村人', sent)
    sent = re.sub('@村@', '村', sent)

    sent = re.sub(r'灰{2,}', '灰', sent)
    sent = re.sub(r'白{2,}', '白', sent)
    sent = re.sub(r'黒{2,}', '黒', sent)

    # sent = regex.sub(r'リ<person>|<person>ナ|<person>ブ|お<person>(?=(ちゃん))|モリお<person>', '<person>', sent)
    #sent = re.sub(r'\(*<person>\)', '<person>', sent)
    #sent = re.sub(r'(<person>){2,}', '<person>', sent)
    #sent = regex.sub(r'(<person>|<|>)+', '<person>', sent)
    return sent