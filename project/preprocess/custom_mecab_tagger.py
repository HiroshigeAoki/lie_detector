import MeCab
import pandas as pd

class CustomMeCabTagger(MeCab.Tagger):

    COLUMNS = ['表層形', '品詞', '品詞細分類1', '品詞細分類2', '品詞細分類3', '活用型', '活用形', '原形', '読み', '発音']

    def __init__(self, option=''):
        self.option = option
        self.tagger = MeCab.Tagger(option)

    def __getstate__(self):
        return {'option': self.option}

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)

    def __getnewargs__(self):
        return self.option,

    def __reduce_ex__(self, proto):
        func = CustomMeCabTagger
        args = self.__getnewargs__()
        state = self.__getstate__()
        listitems = None
        dictitems = None
        rv = (func, args, state, listitems, dictitems)
        return rv

    def __call__(self, text):
        ret = self.tagger.parse(text).rstrip()
        return ret

    def parseToDataFrame(self, text: str) -> pd.DataFrame:
        """テキストを parse した結果を Pandas DataFrame として返す"""
        results = []
        for line in self(text).split('\n'):
            feature = [None for _ in range(len(type(self).COLUMNS) - 1)]
            if line == 'EOS':
                break
            surface, _feature = line.split('\t')
            for i, f in enumerate(_feature.split(',')):
                feature[i] = f if f!='*' else None
            results.append([surface, *feature])
        return pd.DataFrame(results, columns=type(self).COLUMNS)