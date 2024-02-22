import neologdn
import demoji
import regex


def add_to_replacement_track_dict(sent: str, pattern_name: str, matched_text: str, replacement: str, replacement_track_dict: dict):
    if matched_text not in replacement_track_dict:
        replacement_track_dict[matched_text] = {
            'count': 0,
            'pattern': pattern_name,
            'replacement': replacement,
            'examples': set()
        }
    replacement_track_dict[matched_text]['count'] += 1
    if len(replacement_track_dict[matched_text]['examples']) < 3:
        replacement_track_dict[matched_text]['examples'].add(sent)


def track_replacements(sent, pattern_name, pattern, replacement, replacement_track_dict):
    def replace_func(match):
        matched_text = match.group(0)
        add_to_replacement_track_dict(sent=sent, pattern_name=pattern_name, matched_text=matched_text, replacement=replacement, replacement_track_dict=replacement_track_dict)
        return replacement

    if len(sent) == 0:
        return sent

    modified_sent = regex.sub(pattern, replace_func, sent)
    remained_ratio = len(modified_sent) / len(sent)
    if pattern_name in ["remove_terms_with_pipe", "remove_with_char_first_letter"] and remained_ratio < 0.8:
        add_to_replacement_track_dict(sent=sent, pattern_name="removed", matched_text="removed", replacement="", replacement_track_dict=replacement_track_dict)
        return ""

    return modified_sent


def clean_sent(sent: str, replacement_track_dict: dict) -> str:
    sent = neologdn.normalize(sent, repeat=3) # 表記揺れを無くす
    sent = sent.lower()
    sent = demoji.replace(string=sent, repl="") # 絵文字削除
    
    japanese_unicode_ranges = r'[^あ-ん\u30A1-\u30F4\u2E80-\u2FDF\u3005-\u3007\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\U00020000-\U0002EBEF]+'
    left_right_wing = r"[m|c|u|o|つ|っ|ノ|シ|ヽ|人|⊂|＜|＞|<|>|≦|≧|≲|≳|≪|≫|《|》|⊂|彡|☆||\^|⌍|♪|＾|⌒|@|σ|α|β|∂|≡|=|-|~|～|ﾟ|・|·|་|\*|＊|'|`|｀|´|❛|;|:|,|\.|\(|\)|\||｜|￥|\\|\/|＼|／]*"
    mouse = r"(Д|д|ω|ᴗ|∇|_ _|σ|α|β)"
    left_cheek = r'[\(（\|]@*'
    right_cheek = r'[\)）\|]@*'
    
    patterns_replacements = [
        ("remove_blankets", r'[\(（](\d+|\?)[\)）]', '?'),
        ("remove_kaomoji_other", r'[\^|＾][\^|＾][;；]*|orz|\|*[＿|_]\|￣\|[○|◯]\|*', ''), # それ以外
        ("remove_kaomoji_both", left_right_wing + left_cheek + japanese_unicode_ranges + right_cheek + left_right_wing, ''), # ()等で囲まれているもの
        ("remove_kaomoji_left", left_right_wing + left_cheek + mouse + left_right_wing , ''), # 左頬あり
        ("remove_kaomoji_right", left_right_wing + mouse + right_cheek + left_right_wing, ''), # 右頬あり
        ("remove_.o0", r"\.+o0", ''),
        ("remove_whitespace", r'[\t\n\r]', ''), 
        ("remove_res_anker_numeric", r'[>]+[0-9]+', ''),
        ("remove_res_anker_at", r'[@][0-9]+', ''),
        ("remove_url", r'https?:?//[-_.!~*\'()a-zA-Z0-9;/?:@&=+$,%#]+', ''),
        ("remove_symbols", r'[\u0900-\u0FFF\u2500-\u2E52]', ''),
        ("normalize_numbers", r'[0-9,]+[0-9]+', '0'),
        ("normalize_decimal", r'[0-9]+\.?[0-9]*', '0'),
        ("remove_brackets", r'[【】\[\]]', ''),
        ("replace_0_colon_0", r'0:0', '0時'),
        ("replace_0d", r'0d', '0日'),
        ("replace_prob", r'0/0', ''),
        ("replace_0_pipe_0", r'\|0\|0|0-0', '0'),
        ("replace_space", ' |　', '')
    ]
    
    for args in patterns_replacements:
        sent = track_replacements(sent, *args, replacement_track_dict)
        
    return sent


def replace_term(sent: str, replacement_track_dict: dict) -> str:
    # キャラクター名を置換
    names = [
        "行商人 アルビン|アルビン|行商人|商人",
        "村長 ヴァルター|ヴァル(ター)*|村長|そんちょ|ヴェル",
        "仕立て屋 エルナ|エルナ|仕立て屋|エルゥ|エレナ",
        "パン屋 オットー|オットー*|パン屋",
        "羊飼い カタリナ|カタリナ|羊飼い|カタりん",
        "司書 クララ|クララ|司書|クラりん|クラ[ら|ラ]ん",
        "楽天家 ゲルト|ゲルト|楽天家",
        "神父 ジムゾン|ジムゾン|神父|ジムゾ|ジム=ゾン|ジムスン|ジム!ゾン!",
        "負傷兵 シモン|シモン*|シモーン|兵隊|負傷兵",
        "ならず者 ディーター|Ｄ太|ディータ(ー)*|ディッタン|ディダ|ならず者",
        "木こり トーマス|トー*マス*|トム|機関車トー○ス|肉妖精|木こり",
        "旅人 ニコラス|スナフキン|薄緑|旅人|ニコ(ラス|ルン)+|ニコラ",
        "村娘 パメラ|村娘|パメラ*",
        "シスター フリーデル|フリーデル|シスター|リデル",
        "少年 ペーター|ペータ(ー)*|少年|ピーター|ベーター|ペー太|ぺーた(ー|ん)*|ぺー(君|にゃん)|ペタ",
        "老人 モーリッツ|モーリツ|老人|おじいちゃん|じーさん|じいちゃん|じじい|お爺さん|もーじ[い|ぃ]|長老|モーリッツ|モっちん",
        "農夫 ヤコブ|ヤコ(ブ|ビン|ぷー)*|やこびー|農夫",
        "青年 ヨアヒム|ヨアヒム|ヨアムヒ|ヨアヒ|よあひー|よあひむ|青年",
        "少女 リーザ|リ(ーズ|ーザ|ザ){1,}|少女",
        "宿屋の女主人 レジーナ|レジーナ|おばさん|マダム|レジナ|姐さん|姐御|宿屋の女主人|宿屋のおかみ|女将|小母様",
    ]

    name_char = "楽|村|長|老|爺|翁|神|木|樵|旅|者|年|少|妙|女|好|行|商|羊|屋|青|娘|農|宿|姐|修|尼|仕|服|司|書|兵|傷|負"
    
    nick_names = "アル|エル|ルナ|オト|カタ|リナ|クラ|ララ|ゲル|ジム|ディタ|ディ|ニコ|フリ|モリ|モー|ヨア|レジ"

    honorific_titles = '|'.join([
        "姉", "ね[え|ぇ]", "お姉", "おね[え|ぇ]", "お嬢",
        "兄", "に[い|ぃ]", "お兄", "おに[い|ぃ]",
        "おじいちゃん", "爺", "翁", "じ[い|ぃ]",
        "さん|サン","くん|クン|君|san", "ちゃん|チャン", "坊|ぼう", "子",
        "様", "さま", "殿", "どの",
        "どん", "さぁ", "はん",
        "ち", "っち", "ぴょん", "ぽん|ポン","ぷ", "にゃん", "にょ", "わん", "ー*たん", "らん", "りん", "猫ちゃん", "ワンちゃん", "パンマン", "[ぴ|ピ]ー"
    ])

    full_role_names = r"人狼|狂人|真?占い師|真?占師|真?霊媒師|真?霊能者|真?狩人|共有者|ハムスター人間|村人"
    short_role_names = "狼|狂|占|霊|狩|共"
    short_term = "真|白|黒|灰"
    short_general_term1 = "非|票|吊|偽"
    short_general_term2 = "羊|人|敵|偽|死"
    first_letters = r"ア|ヴ|エ|オ|カ|ク|ゲ|ジ|シ|デ|ディ|ト|ニ|ス|パ|フ|ペ|モ|ヤ|ヨ|リ|レ"
    
    person_char_pattern = rf"{name_char}|{first_letters}|<person>"
    
    short_char_terms = (
        rf"{short_role_names}|{short_term}|{short_general_term1}|{short_general_term2}"
    )
    
    particles = "は|も|を|が|か|で|と|に|の|や|へ|なら"

    symbols = "→|^|-|⇒|⇔|or|vs|…|=|>|<|_|≧|≠|\.|×|\+|「|:|。|、|－|ー|\(|\)|\||\\|/|/|\"|・|0|\?|\!"
    connections = "\||→|-|⇒|⇔|or|vs|\=|\-|:|・"

    freq_words = "どっち|的には|のように|co|判定|発言|内心|視点|目線|確定|確白|確黒|希望|希望出し|両狼|確認|考察|襲撃|護衛|直吊|吊|即吊|対抗|ライン|連打|不在|パンダ|ぱんだ|ステルス|墓|ごめん|ありがとう|食[っ|べ|う]|噛|なのー|寄り|じゃなく|って|投票|sg|gj|pp|スケープゴート|ゴート|生か|潜|優秀|更新|見え(る|た|て)|入れ(て|た|る)|いや|覆[ら|る]|切る|勝|負|本命|来た|切(る|っ|ろ|り)|候補|守護|今日|明日|昨日"

    color = "黒|白|灰|グレー|グレイ|グレ|グレ|真|偽|"
    patterns_replacements = [
        ("replace_char_name_1", regex.compile('|'.join(names)), '<person>'),
        # TF-IDF値が高かった人狼用語を削除
        ("remove_full_role_name", full_role_names, ''),
        ("game_strategies", rf'(が[{color}]|は[{color}]|[{color}]っぽい|[{color}]寄り|[{color}]と思う|[{color}]だと思う|[{color}]でいい|[{color}]決め打ち|[{color}]で見|[{connections}][{color}]|[{color}]確定?|が偽|偽っぽい|白判定|白確定?|白く|は白|黒判定|黒確定?|黒い|黒く|グレースケール|グレーゾーン|グレー|能力者|役職者|人外|人狼|人間|突然死|仮決定)', ''),
        ("game_actions", r'(寡黙|狼が|狼を|狼の|が狼|占い先|占い対象|占われた|占わ|吊っちゃった|吊った|占っ|占う|共吊り|吊り|吊る|吊って|吊られそう|吊られる|吊ろうとした|吊ろう|吊れ|吊ら|co|護衛|占い|襲撃|潜伏|対抗|狩り|狩って|狩る|襲い|襲う|襲った|ローラー|ステルス|ロラ|スライド|発表順了解|発表順も了解)', ''),
        ("general_game_terms", r'(占い希望|ライン|灰考察|本決定|議事録|仮決定|墓下|投票|fo|log|gj|gs|sg|rp)', ''),
        ("remove_char1", rf'非(?=({short_char_terms}))', ''),
        
        ("preserve", r'旅行(?=(に|へ|でもいかない|でもいかない|行く|いく))', '@旅@行@'),
        ("preserve", "長旅", "@長@旅@"),
        ("preserve", r'旅(?=(の|に|の途中|の途中|に出る|に出る|に|が好き|好き))', '@旅@'),
        ("preserve", r'修行(?=(に|へ|する|に行く|にいく))', '@修@行@'),
        ("preserve", r'負傷(?=(した|して|がひどい|こそ|具合))', '@負@傷@'),
        ("preserve", r'(?<=(足|手|体))負傷', '@負@傷@'),
        ("preserve", r'宿屋(?=(に|の))', '@宿@屋@'),
        ("preserve", r'宿(?=(売上げ|に泊|課税))', '@宿@'),
        ("preserve", r'村人(?=(なら|は|の|を|か|として))', '@村@人@'),
        ("preserve", rf'村(?=(のために|として|の総意|[はの]負け|を滅ぼ|[は|の]勝|へ|々|{particles}))', '@村@'),
        ("preserve", r'(?<=(この|私たちの))村', '@村@'),
        ("preserve", "若者", "@若@者@"),
        ("preserve", "長年", "@長@年@"),
        ("preserve", r'女好(?=き)', '@女@好@き@'),

        ("remove_char_with_connections", f"({name_char}|{first_letters}|{short_role_names}|{short_term})+({connections}|0){1,}({name_char}|{first_letters}|{short_role_names}|{short_term}|{connections}|0|<person>)*", ''),

        ("remove_name_person_char1", rf"(?<=({person_char_pattern}|{short_char_terms}))({name_char})+",  ""),
        ("remove_name_person_char2", rf"({name_char})+(?=({person_char_pattern}|{short_char_terms}))", ""),
        
        ("remove_name_person_kana1", rf"(?<=({name_char}|<person>|{short_char_terms}))({first_letters})+",  ""),
        ("remove_name_person_kana2", rf"({first_letters})+(?=({name_char}|<person>|{short_char_terms}))", ""),
        ("replace_name", rf"({nick_names}|{name_char})(?=([{particles}|{symbols}|{freq_words}|{short_char_terms}|{nick_names}|{honorific_titles}]))", '<person>'),
        
        ("preserve", '@旅@行@', '旅行'),
        ("preserve", '@長@旅@', '長旅'),
        ("preserve", '@旅@', '旅'),
        ("preserve", '@修@行@', '修行'),
        ("preserve", '@負@傷@', '負傷'),
        ("preserve", '@宿@屋@', '宿屋'),
        ("preserve", '@宿@', '宿'),
        ("preserve", '@村@人@', '村人'),
        ("preserve", '@村@', '村'),
        ("preserve", '@者@', '者'),
        ("preserve", "@若@者@", "若者"),
        ("preserve", "@長@年@", "長年"),
        ("preserve", "@女@好@き@", "女好き"),

        ("reduce_repeated_patterns", rf"(<person>|<person>0|<person>>|<person>・)\1+", r'\1'),

        ("preserve", "言霊", "@言@霊@"),
        ("preserve", "全然", "@全@然@"),
        
        ("remove_char2", rf'(?:は|が)(?:{short_char_terms})|(?:{short_char_terms})(?:ならば|なら|だ|かも|確定|確|が確定)', ''),
        ("remove_term_char1", rf"(?<=({person_char_pattern}|{short_char_terms}))({short_char_terms})+", ""),
        ("remove_term_char2", rf"({short_char_terms})+(?=({person_char_pattern}|{short_char_terms}))", ""),
        
        ("preserve", "@言@霊@", "言霊"),
        ("preserve", "@全@然@", "全然"),

        ("remove_char3", r"[狼占吊狂狩灰霊→\(\)\|]", ''),
        ("remove_pipe_slash", r"[\|\\\/_]{2,}", ''),
    ]
    
    for args in patterns_replacements:
        sent = track_replacements(sent, *args, replacement_track_dict)
    
    return sent


def clean_murder_mystery_data(sent, replacement_track_dict):
    patterns_replacements = [
        ("見やるは、誰ぞ、", r"(八木|ヤギ|マトン|夜見|ヨミ|オコジョ|仁江|ニモ|モズ|持衰|ジサイ|サシミ)", "<person>"),
        ("ワタナベ1", r'(南極の(渡辺|わたなべ|ワタナベ)|沖縄の(渡部|わたなべ|ワタナベ)|スウェーデンの(渡邉|わたなべ|ワタナベ))', "<person>"),
        ("ワタナベ2",  r'[”|"]?(わたなべ|ワタナベ|わたべ|ワタベ)[”|"]?', "<person>"),
        ("つづりあわせの流儀", r"(ザンジュ|スワヒリ|バントゥー|サバナ)", "<person>"),
        ("死者の湯加減", r"(ダイヴ侍|ダイヴ|ダイブ|侍|小泉潤|小泉|潤|花咲ラン|花咲|ラン|記本文章|記本|基本|赤木巧|赤木|巧|小泉充之介|充之介)", "<person>"),
        ("mention", r"<@\w+>", "<person>"),
        ("mention utter", r"<>", "")
    ]

    sent = clean_sent(sent, replacement_track_dict=replacement_track_dict)

    for args in patterns_replacements:
        sent = track_replacements(sent, *args, replacement_track_dict=replacement_track_dict)
    
    return sent
