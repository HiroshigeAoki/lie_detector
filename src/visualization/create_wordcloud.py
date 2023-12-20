import argparse
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm

def main(data_dir, top_n):
    font_path = '/usr/share/fonts/truetype/fonts-japanese-gothic.ttf'
    for n_gram in tqdm([1, 2, 3, 4, 5]):
        label_0 = pd.read_csv(os.path.join(data_dir, f"diff_{n_gram}_gram_0.csv"), index_col=0)
        label_1 = pd.read_csv(os.path.join(data_dir, f"diff_{n_gram}_gram_1.csv"), index_col=0)
        
        top_n_0 = label_0[:top_n].to_dict()['difference']
        top_n_1 = label_1[:top_n].to_dict()['difference']
        
        top_n_0 = {k.replace(' ', ''): v for k, v in top_n_0.items()}
        top_n_1 = {k.replace(' ', ''): v for k, v in top_n_1.items()}

        wordcloud_0_jp = WordCloud(font_path=font_path, width=800, height=400, background_color='white').generate_from_frequencies(top_n_0)
        wordcloud_1_jp = WordCloud(font_path=font_path, width=800, height=400, background_color='white').generate_from_frequencies(top_n_1)

        # 背景を白に設定
        plt.figure(figsize=(16, 8), facecolor='white')

        plt.subplot(1, 2, 1)
        plt.imshow(wordcloud_0_jp, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Top {top_n} {n_gram}-grams for Label 0')

        plt.subplot(1, 2, 2)
        plt.imshow(wordcloud_1_jp, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Top {top_n} {n_gram}-grams for Label 1')

        combined_image_file = os.path.join(data_dir, f'wordcloud_{n_gram}_top_{top_n}.png')
        plt.savefig(combined_image_file, bbox_inches='tight', facecolor='white')
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate word clouds from CSV data.")
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--top_n', type=int, default=100)
    
    args = parser.parse_args()
    data_dir = os.path.join("data", args.data, "ngram")
    main(data_dir, args.top_n)
