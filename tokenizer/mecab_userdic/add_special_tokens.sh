#!/bin/sh

/usr/local/libexec/mecab/mecab-dict-index -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd -u ./special.dic -f utf-8 -t utf-8 special_tokens.csv
