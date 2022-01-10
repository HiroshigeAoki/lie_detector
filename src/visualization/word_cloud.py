import pandas as pd
import os

import matplotlib.pyplot as plt
from wordcloud import WordCloud
font_path = '/usr/share/fonts/truetype/fonts-japanese-gothic.ttf'

def create_wordcloud(df):
    d = {}
    df = df.dropna()
    for a, x in df.values:
        a = a.replace('▁', '_')
        d[a] = x
    wordcloud = WordCloud(font_path=font_path, background_color='white')
    wordcloud.generate_from_frequencies(frequencies=d)
    return wordcloud


def load_data(input_dir, mode):
    FN_60_50 = pd.read_csv(os.path.join(input_dir, f'FN_60_50_vital_word_{mode}.csv'))
    FP_60_50 = pd.read_csv(os.path.join(input_dir, f'FP_60_50_vital_word_{mode}.csv'))
    TN_60_50 = pd.read_csv(os.path.join(input_dir, f'TN_60_50_vital_word_{mode}.csv'))
    TP_60_50 = pd.read_csv(os.path.join(input_dir, f'TP_60_50_vital_word_{mode}.csv'))
    FN_70 = pd.read_csv(os.path.join(input_dir, f'FN_70_vital_word_{mode}.csv'))
    FP_70 = pd.read_csv(os.path.join(input_dir, f'FP_70_vital_word_{mode}.csv'))
    TN_70 = pd.read_csv(os.path.join(input_dir, f'TN_70_vital_word_{mode}.csv'))
    TP_70 = pd.read_csv(os.path.join(input_dir, f'TP_70_vital_word_{mode}.csv'))
    FN_80 = pd.read_csv(os.path.join(input_dir, f'FN_80_vital_word_{mode}.csv'))
    FP_80 = pd.read_csv(os.path.join(input_dir, f'FP_80_vital_word_{mode}.csv'))
    TN_80 = pd.read_csv(os.path.join(input_dir, f'TN_80_vital_word_{mode}.csv'))
    TP_80 = pd.read_csv(os.path.join(input_dir, f'TP_80_vital_word_{mode}.csv'))
    FN_90 = pd.read_csv(os.path.join(input_dir, f'FN_90_vital_word_{mode}.csv'))
    FP_90 = pd.read_csv(os.path.join(input_dir, f'FP_90_vital_word_{mode}.csv'))
    TN_90 = pd.read_csv(os.path.join(input_dir, f'TN_90_vital_word_{mode}.csv'))
    TP_90 = pd.read_csv(os.path.join(input_dir, f'TP_90_vital_word_{mode}.csv'))

    TP_90_80 = pd.concat((TP_90, TP_80), axis=0).groupby('token').sum().sort_values(by=mode, ascending=False).reset_index()
    TN_90_80 = pd.concat((TN_90, TN_80), axis=0).groupby('token').sum().sort_values(by=mode, ascending=False).reset_index()
    FP_90_80 = pd.concat((FP_90, FP_80), axis=0).groupby('token').sum().sort_values(by=mode, ascending=False).reset_index()
    FN_90_80 = pd.concat((FN_90, FN_80), axis=0).groupby('token').sum().sort_values(by=mode, ascending=False).reset_index()

    TP_90_50 = pd.concat((TP_90, TP_80, TP_70, TP_60_50), axis=0).groupby('token').sum().sort_values(by=mode, ascending=False).reset_index()
    TN_90_50 = pd.concat((TN_90, TN_80, TN_70, TN_60_50), axis=0).groupby('token').sum().sort_values(by=mode, ascending=False).reset_index()
    FP_90_50 = pd.concat((FP_90, FP_80, FP_70, FP_60_50), axis=0).groupby('token').sum().sort_values(by=mode, ascending=False).reset_index()
    FN_90_50 = pd.concat((FN_90, FN_80, FN_70, FN_60_50), axis=0).groupby('token').sum().sort_values(by=mode, ascending=False).reset_index()

    return dict(
        data_60_50=(TP_60_50, TN_60_50, FP_60_50, FN_60_50),
        data_70=(TP_70, TN_70, FP_70, FN_70),
        data_80=(TP_80, TN_80, FP_80, FN_80),
        data_90=(TP_90, TN_90, FP_90, FN_90),
        data_90_80=(TP_90_80, TN_90_80, FP_90_80, FN_90_80),
        data_90_50=(TP_90_50, TN_90_50, FP_90_50, FN_90_50)
    )


def unique(TP, TN, FP, FN, mode):
    set_TP = set(TP['token'])
    set_TN = set(TN['token'])
    set_FP = set(FP['token'])
    set_FN = set(FN['token'])

    u_TP = ((set_TP - set_TN) - set_FP) - set_FN
    u_TN = ((set_TN - set_TP) - set_FP) - set_FN
    u_FP = ((set_FP - set_TN) - set_TP) - set_FN
    u_FN = ((set_FN - set_TN) - set_FP) - set_TP
    u_TP = TP[TP['token'].isin(list(u_TP))].sort_values(by=mode, ascending=False)
    u_TN = TN[TN['token'].isin(list(u_TN))].sort_values(by=mode, ascending=False)
    u_FP = FP[FP['token'].isin(list(u_FP))].sort_values(by=mode, ascending=False)
    u_FN = FN[FN['token'].isin(list(u_FN))].sort_values(by=mode, ascending=False)

    return u_TP, u_TN, u_FP, u_FN


def plot_TP_TN_FP_FN(wordcloud, pred_class, file_path, suptitle):
    row=2
    col=2
    fig, ax = plt.subplots(nrows=row, ncols=col,figsize=(18,10))
    fig.suptitle(suptitle, fontsize=20, color='black')
    for i, img in enumerate(wordcloud):
        _r= i//col
        _c= i%col
        ax[_r,_c].set_title(pred_class[i], fontsize=16, color='black')
        ax[_r,_c].axes.xaxis.set_visible(False) # X軸を非表示に
        ax[_r,_c].axes.yaxis.set_visible(False) # Y軸を非表示に
        ax[_r,_c].imshow(img, interpolation="bilinear") # 画像を表示
    fig.savefig(file_path)
    plt.clf()
    plt.close()


def plot_positive_negative(wordcloud, pred_class, file_path, suptitle):
    row=1
    col=2
    fig, ax = plt.subplots(nrows=row, ncols=col,figsize=(15,5))
    fig.suptitle(suptitle, fontsize=20, color='black')
    for i, img in enumerate(wordcloud):
        ax[i].set_title(pred_class[i], fontsize=16, color='black')
        ax[i].axes.xaxis.set_visible(False) # X軸を非表示に
        ax[i].axes.yaxis.set_visible(False) # Y軸を非表示に
        ax[i].imshow(img, interpolation="bilinear") # 画像を表示
    fig.savefig(file_path)
    plt.clf()
    plt.close()


def main():
    n_grams = ['uni', 'bi', 'tri']
    modes = ['freq', 'weight']
    DCs = ['90', '80', '70', '60_50', '90_80', '90_50']

    for n_gram in n_grams:
        base_dir = f'outputs/nested/HAN/baseline/200_dim200_sp/plot_attention_{n_gram}-gram'

        for mode in modes:
            input_dir = os.path.join(base_dir, f'csv/{mode}')
            data = load_data(input_dir, mode=mode)

            for DC in DCs:
                save_dir = os.path.join(base_dir.replace(f'plot_attention_{n_gram}-gram', f'wordcloud/{n_gram}-gram'), f'{DC}%/{mode}')
                os.makedirs(save_dir, exist_ok=True)

                suptitle = f'{n_gram}-gram {DC}% {mode}'

                TP, TN, FP, FN = data[f'data_{DC}']
                u_TP, u_TN, u_FP, u_FN = unique(TP, TN, FP, FN, mode=mode)

                wordcloud = [create_wordcloud(pred_class_df) for pred_class_df in [TP, FN, FP, TN]]
                pred_class = ['TP', 'FN', 'FP', 'TN']
                plot_TP_TN_FP_FN(wordcloud=wordcloud, pred_class=pred_class, file_path=os.path.join(save_dir, f'TP_TN_FP_FN_{DC}%_{mode}.png'), suptitle=suptitle)

                wordcloud = [create_wordcloud(pred_class_df) for pred_class_df in [u_TP, u_FN, u_FP, u_TN]]
                pred_class = ['TP-(TN+FP+FN) (A:werewolf)', 'FN-(TP+TN+FP) (A:citizen)', 'FP-(TP+TN+FN) (A:werewolf)', 'TN-(TP+FP+FN) (A:citizen)']
                plot_TP_TN_FP_FN(wordcloud=wordcloud, pred_class=pred_class, file_path=os.path.join(save_dir, f'uTP_uTN_uFP_uFN_{DC}%_{mode}.png'), suptitle=suptitle)

                wordcloud = [create_wordcloud(pred_class_df) for pred_class_df in [u_TP, u_TN]]
                pred_class = ['TP-(TN+FP+FN) (A:werewolf)', 'TN-(TP+FP+FN) (A:citizen)']
                plot_positive_negative(wordcloud=wordcloud, pred_class=pred_class, file_path=os.path.join(save_dir, f'uTP_uTN_{DC}%_{mode}.png'), suptitle=suptitle)

                wordcloud = [create_wordcloud(pred_class_df) for pred_class_df in [u_FP, u_FN]]
                pred_class = ['FP-(TP+TN+FN) (A:werewolf)', 'FN-(TP+TN+FP) (A:citizen)']
                plot_positive_negative(wordcloud=wordcloud, pred_class=pred_class, file_path=os.path.join(save_dir, f'uFP_uFN_{DC}%_{mode}.png'), suptitle=suptitle)


if __name__ == "__main__":
    main()