from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import os, sys
from tqdm import tqdm
from glob import glob
import argparse

sys.path.append("./src")
from utils.make_ngram_diff_cm import get_model_data_dir_test

def main(data_dir, top_n):
    font_path = '/usr/share/fonts/truetype/fonts-japanese-gothic.ttf'
    
    # normed_diffディレクトリ内のCSVファイルの一覧を取得
    csv_files = glob(os.path.join(data_dir, "normed_diff", "*.csv"))
    
    for file_path in tqdm(csv_files):
        df = pd.read_csv(file_path)
        top_n_words = df[:top_n].set_index('ngram')['difference'].to_dict()
        
        # 空白を取り除く
        top_n_words = {str(k).replace(' ', ''): v for k, v in top_n_words.items()}

        # WordCloudを生成
        wordcloud_jp = WordCloud(font_path=font_path, width=800, height=400, background_color='white').generate_from_frequencies(top_n_words)

        # WordCloudを表示
        
        plt.figure(figsize=(8, 4), facecolor='white')
        plt.imshow(wordcloud_jp, interpolation='bilinear')
        plt.axis('off')
        # plt.title(f'Top {top_n} Words in {os.path.basename(file_path)}')

        # WordCloudを保存
        file_name = os.path.basename(file_path)
        wordcloud_image_file = os.path.join(data_dir, "wordclouds", f'wordcloud_{file_name}.png')
        plt.savefig(wordcloud_image_file, bbox_inches='tight', facecolor='white')
        plt.close()

def create_for_outputs():
    for data in ["nested_day", "nested_day_twitter", "murder_mystery"]:
        for model in ["bigbird", "hierbert"]:
            data_dir, _ = get_model_data_dir_test(data, model)
            os.makedirs(os.path.join(data_dir, "wordclouds"), exist_ok=True)
            main(data_dir, 200)
            
def create_for_data():
    data_dir = "data/nested_day_twitter/hierbert/ngram"
    os.makedirs(os.path.join(data_dir, "wordclouds"),exist_ok=True)
    main(data_dir, 200)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", action='store_true')
    args = parser.parse_args()
    print(args.data)
    if args.data:
        create_for_data()
    else:
        create_for_outputs()