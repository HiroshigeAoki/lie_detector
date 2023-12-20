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
        
    sent = replace_term(sent, replacement_track_dict)
    
    return sent


def replace_term(sent: str, replacement_track_dict: dict) -> str:
    # キャラクター名を置換
    names = [
        "(?V1)行商人 アルビン|アルビン|行商人|商人",
        "村長 ヴァルター|ヴァル(ター)*|村長|そんちょ|ヴェル",
        "仕立て屋 エルナ|エルナ|仕立て屋|エルゥ|エレナ",
        "パン屋 オットー|オットー*|パン屋",
        "羊飼い カタリナ|カタリナ|羊飼い|カタりん",
        "司書 クララ|クララ|司書|クラりん|クラ[ら|ラ]ん",
        "楽天家 ゲルト|ゲルト|楽天家",
        "神父 ジムゾン|ジムゾン|神父|ジムゾ|ジム=ゾン|ジム!ゾン!",
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
        "少女 リーザ|リ(ーズ|ーザ|ザ)+|少女",
        "宿屋の女主人 レジーナ|レジーナ|おばさん|マダム|レジナ|姐さん|姐御|宿屋の女主人|宿屋のおかみ|女将|小母様",
    ]

    name_char = '|'.join([
        "(?<!(田|気|音|苦|安|娯|哀|快|極|洋|邦))楽(?!(器|園|観|市|座|天|勝|過ぎ|天家|し[いむく]|団|典))",
        "(?<!(この|漁|山|寒|隣))村(?!(起こし|を上げて|落|八分|雨|のみんな|の総意))",
        "(?<!(身|発話|発言|館|機|店|校|学科|委員|総|番|首|家|冗|助|成|百薬の))長(?!(い|短|兄|姉|身|所|いも|芋|編|蛇|閑))",
        "(?<!(不|養))老(?!(獪|い|け|化|害|眼|若男女))",
        "爺",
        "翁",
        "(?<!(ゼウス|ぜうす|太陽|恵みの|鬼|疫病|禍|氏))神(?!(社|宮|様|のみぞ|聖|に祈る|速|罰|秘|童|出鬼没|経|[無|在]月))",
        "(?<!(材|(スギ|杉|柳|桜|サクラ|いちょう)の|檜|朽ち*|猿も|樹|土|版|ほう))木(?!(材|製|星|イチゴ|苺|いちご|戸|枯らし|陰|魚|綿|琴))",
        "樵(?!(り))",
        "(?<!((かわいい|可愛い)子には))旅(?!((に|へ)(出る|行く)|程|順|路|券|団|籠|客))",
        "(?<!(共有|能力|霊能|強|弱|拙|読|影武|(回|まわ)し|患|たわけ|戯け|去る|記|業|賢|口達|三|使|指揮|識|前任|後継|前|後|達|著|忍|覇|亡|役|勇|有権))者(?!(択一))",
        "(?<!(昨|去|来|今|往|永|各|享|凶|更|周|若|熟|積|中|定|晩|忘|没|歴|暦|留|豊|厄|[一二三四五六七八九十百千万億兆京]))年(?!(内|間|始|初|末|明|度|頭|譜|頃|寄|賀|端|功|中|来|齢|次|貢|金|号|輩|配|輪|をとった))",
        "(?<!(極|希|減|稀|最|些|過|多|幼|少))少(?!(々|し|ない|数|額|量|数))",
        "(?<!(絶|微|奇|巧|軽|珍|言い得て))妙(?!(に|な(?!(ら|の))|薬|策|計|齢|手|味|法))",
        "(?<!(男|悪|下|乙|鬼|次|海|山|売|彼))女(?!(性|の子|権|系|王))",
        "(?<!(同|嗜|恰|愛|絶|選り|えり|大|物|良|格))好(?!(意|感|み|悪|機|奇|例|物|調|適|転|敵手|日|事|評|む|ま|き))",
        "(?<!(同|続|鈍|徐|退|進|運|逆|血|徐|随|両|並|平|飛|夜|連|紀|危|銀|品|洋|所|諸|移|横|愚|蛇|凶|飛|非|動|苦|急|強|慣|刊|奉|奥|横|奇|逆|現|孝|興|試|施|執|巡|代|単|知|通|発|尾|犯|蛮|道|徳|断|昼|草|素|千里の|遂|心|巡|順|実))行(?!(動|政|い|う|わ|お|っ|[か-こ]|間|列|儀|幸|事|為|軍|程|楽|使|為|文|雲|跡|政|状|水|脚|火))",
        "(?<!(工|画|外|豪|通))商(?!(品|い|才|業|売|う|魂|略|標))",
        "(?<!(群|山|麺|綿|亡))羊(?!(小屋|毛|水|羹|かん|皮紙))",
        "(?<!(うどん|そば|楽|一軒|家|寺子|陣|部|茶|岩|的|床|廃|八百|魚|肉|母|納|小|平|牢|ろう|庄))屋(?!(台|敷|根))",
        "(?<!(群|刺|真っ|緑|紺))青(?!(物|空|天|銅|写真|二才|天井|菜|鬼|嵐|柳|魚|果|眼|雲|海|山|酸|蠅|蝿|磁|春))",
        "(?<!(モーニング|もーにんぐ|愛|生|うちの))娘",
        "(?<!(酪|小作|豪|士|労))農(?!(場|家|業|隙|薬|奴|耕|芸|園))",
        "(?<!(下|寄|合|定|木賃|民))宿(?!(泊|縁|敵|命|題|る|す|意))",
        "姐",
        "(?<!(学|阿|研|監|専|補|必|独))修(?!(める|学|復|士|業|辞|築|道|復|了|正|整|飾|得|理|験道|練|錬|竹|羅|繕|法))",
        "尼",
        "(?<!(お|奉|出))仕(?!(方|え|事|様|合|置|官|出|える|度|損|立|業|る|分|舞い))",
        "(?<!(春|夏|秋|冬|一|圧|感|屈|呉|私|克|制|喪|体操|呉|降|着|被|礼|洋|和|紋|征))服(?!(飾|役|従|毒|薬|装|用|従|する|務))",
        "(?<!(上|曹|下))司(?!(る|会|令))",
        "(?<!(覚え|封|遺|手|願|文|楷|公文|辞|肩|清|聖|図|代|蔵|葉|端|著|調|白|勅|読|請|秘|封|裏|投))書(?!(簡|物|き|く|い|院|留|家|簡|見|式|状|写|斎|面))",
        "(?<!(雑|出))兵(?!(役|法|隊|舎))",
        "(?<!(古|火|外|感|軽|殺|死|挫|手|深|浅|損|凍|生|中|破|食))傷(?!(口|害|跡|む|い|心|痕))",
        "(?<!(勝(ち)*|誇|自|気|請け|背|抱|手))負(?!(債|担|荷|託|う|い目|け|い))",
    ])
    nick_names = '|'.join([
        "(?<!(リ|レ))アル(?!(コール|よ|ヨ|フォート|中|ミ|ゴン|バイト))",
        "エル",
        "ルナ",
        "オト(?!(コ|ナ))",
        "(?<!(ド|ミ))カタ(?!(パルト|リシス|チ))",
        "リナ(?!(ックス))",
        "(?<!(クラ|オ))クラ(?!(イシス|クラ|リネット|ス))",
        "(?<!(ラ))ララ(?!(ラ))",
        "ゲル(?!(状))",
        "ジム",
        "(?<!(エ))ディタ",
        "(?<!(アイ))ディ(?!(ナー))",
        "(?<!(ニコ))ニコ(?!(ニコ))",
        "(?<!(フリ))フリ(?!(ートーク|ー|ーズ|ース|フリ))",
        "(?<!(タ|イ|トク|モリ))モリ(?!(モリ))",
        "モー",
        "ヨア",
        "レジ(?!(スター))",
    ])

    honorific_titles = '|'.join([
        "姉", "ね[え|ぇ]", "お姉", "おね[え|ぇ]", "お嬢",
        "兄", "に[い|ぃ]", "お兄", "おに[い|ぃ]",
        "おじいちゃん", "爺", "翁", "じ[い|ぃ]",
        "さん|サン","くん|クン|君|san", "ちゃん|チャン", "坊|ぼう", "子",
        "様", "さま", "殿", "どの",
        "どん", "さぁ", "はん",
        "ち", "っち", "ぴょん", "ぽん|ポン","ぷ", "にゃん", "にょ", "わん", "ー*たん", "らん", "りん", "猫ちゃん", "ワンちゃん", "パンマン", "[ぴ|ピ]ー"
    ])

    full_role_names = "人狼|狂人|占い師|占師|霊媒師|狩人|共有者|ハムスター人間|村人"
    short_role_names = "狼|狂|占|霊|狩|共|村|羊"
    short_term = "真|偽|白|黒|灰|吊"
    short_general_term = "非|票|死|全|敵"
    
    particles = "は|も|を|が|か|で|と|に|の|や|へ|なら"

    symbols = "→|^|-|⇒|⇔|or|vs|…|=|>|<|_|≧|≠|\.|×|\+|「|:|。|、|－|ー|\(|\)|\||\\|/|/|\"|・|0|\?|\!"

    reg1 = regex.compile('|'.join(names))

    freq_words = "どっち|的には|のように|co|判定|発言|内心|視点|目線|確定|確白|確黒|希望|希望出し|両狼|確認|考察|襲撃|護衛|直吊|吊|即吊|対抗|ライン|連打|不在|パンダ|ぱんだ|ステルス|墓|ごめん|ありがとう|食[っ|べ|う]|噛|なのー|寄り|じゃなく|って|投票|sg|gj|pp|スケープゴート|ゴート|生か|潜|優秀|更新|見え(る|た|て)|入れ(て|た|る)|いや|覆[ら|る]|切る|勝|負|本命|来た|切(る|っ|ろ|り)|候補|守護|今日|明日|昨日"
    #reg2 = regex.compile(rf"(?V1)(?<!({ignore_prefix}))({name_char}|{nick_names})(?=([{particles}]*{freq_subsequent_words}|{symbols}|<person>|{role_names}|{name_char}|{particles}|{honorific_titles}|(({particles}|{symbols})|({nick_names}|{name_char}|{role_names}))))")
    reg2 = regex.compile(rf"(?V1)({name_char}|{nick_names})(?=(({particles}|{symbols})*({freq_words}|{full_role_names}|{short_role_names}|{short_term}|{name_char}|{nick_names})|{symbols}|<person>|{particles}|{honorific_titles}))")
    #reg3 = regex.compile(rf"(?V1)(?<=(({name_char}|{role_names}|{nick_names})({particles}|{honorific_titles})|<person>|{person_particles}|{person_honorific_titles}|{symbols}|{role_names}|{name_char}|希望出し))({name_char}|{nick_names})(?!({ignore_suffix}))")
    reg3 = regex.compile(rf"(?V1)(?<=({particles}|{freq_words}|((<person>|{full_role_names}|{short_role_names}|{short_term}|{symbols})({particles}|{honorific_titles}|{symbols})*)))({name_char}|{nick_names})")
    #reg3 = regex.compile(rf"(?V1)(?<=(<person>))({name_char}|{nick_names})")
    
    reg4 = regex.compile(
        rf"({particles})\s*({short_term})\s*"
        rf"|({short_term})\s*({particles})"
    )

    patterns_replacements = [
        ("replace_char_name_1", reg1, '<person>'),
        
        ("preserve", r'旅行(?=(に|へ|でもいかない|でもいかない|行く|いく))', '@旅@行@'),
        ("preserve", r'修行(?=(に|へ|する|に行く|にいく))', '@修@行@'),
        ("preserve", r'負傷(?=(した|して|がひどい|こそ|具合))', '@負@傷@'),
        ("preserve", r'(?<=(足|手|体))負傷', '@負@傷@'),
        ("preserve", r'宿屋(?=(に|の))', '@宿@屋@'),
        ("preserve", r'宿(?=(売上げ|に泊|課税))', '@宿@'),
        ("preserve", r'村人(?=(なら|は|の|を|か|として))', '@村@人@'),
        ("preserve", r'村(?=(のために|として|の総意|[は|の]負け|を滅ぼ|[は|の]勝|へ|々))', '@村@'),
        ("preserve", r'(?<=(この|私たちの))村', '@村@'),
        ("preserve", "若者", "若@者@"),

        ("replace_char_name_2", reg2, '<person>'),
        ("replace_char_name_3", reg3, '<person>'),
        
        ("preserve", '@旅@行@', '旅行'),
        ("preserve", '@修@行@', '修行'),
        ("preserve", '@負@傷@', '負傷'),
        ("preserve", '@宿@屋@', '宿屋'),
        ("preserve", '@宿@', '宿'),
        ("preserve", '@村@人@', '村人'),
        ("preserve", '@村@', '村'),
        ("preserve", "若@者@", "若者"),

        ("reduce_repeated_chars", rf'([{short_role_names}|{short_term}])\1+', r'\1'),
        ("remove_terms_with_pipe", rf"\|*[{short_role_names}|{short_term}|{short_general_term}|<person>|\d+|\\|\|]+(\|[{short_role_names}|{short_term}|{short_general_term}|<person>|\d+|\\|\|]+)+\|*" , ''),

        # TF-IDF値が高かった人狼用語を削除
        ("remove_full_role_name", full_role_names, ''),
        ("game_strategies", 'が真|は真|真っぽい|が偽|偽っぽい|白判定|白く|は白|黒判定|黒い|黒く|グレースケール|グレーゾーン|グレー|能力者|人外|人間', ''),
        ("game_actions", '寡黙|狼が|狼を|狼の|が狼|占い対象|占わ|占っ|占う|吊り|吊る|吊って|吊れ|吊ら|co|護衛|占い|襲撃|潜伏|対抗|ローラー|ステルス|ロラ|スライド', ''),
        ("general_game_terms", '占い希望|ライン|灰考察|本決定|議事録|仮決定|墓下|投票|fo|log|gj|gs|sg|rp', ''),
        
        ("preserve", "言霊", "@言@霊@"),
        ("preserve", "突然死", "突然@死@"),
        ("preserve", "全然", "@全@然@"),
        
        ("replace_short_role_name", reg4, ''),
        
        ("preserve", "@言@霊@", "言霊"),
        ("preserve", "突然@死@", "突然死"),
        ("preserve", "@全@然@", "全然"),
        
        ("remove_with_char_first_letter", r'([狼占狂霊全人敵]*[=ー_-・&:\s\|]*[アヴエオカクゲジシデディトニスパフペモヤヨリレ\d]|<person>)+[=ー_-・&:\s\|]*[狼占狂霊全人敵][=:|\\&-。、・]*[狼占狂霊全人敵]*', ''),
        
        ("remove_wolf_char", "狼", ''),
        ("remove_pipe_slash", r"[\|\\\/_]{2,}", ''),
    ]
    
    for args in patterns_replacements:
        sent = track_replacements(sent, *args, replacement_track_dict)
    
    return sent
