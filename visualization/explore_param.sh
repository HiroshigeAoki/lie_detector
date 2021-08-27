#!/home/lyriatest/haoki/Documents/vscode/venv/bin python
rm -fr ./HTML/aozora_wiki_bert/heatmap/last

# color_level
for color_level in $(seq 2.5 .1 2.7);
do
  export CUDA_VISIBLE_DEVICES="1"
  python mk_heatmap.py --yoshinobu --attention_type="last" --sample --color_level=$color_level --temp=0.04 --threshold=0.025
done

# temp
#for temp in $(seq .03 .001 .05);
#do
#  export CUDA_VISIBLE_DEVICES="7"
#  python mk_heatmap.py --yoshinobu --attention_type="last" --sample --color_level=3.5 --temp=$temp --threshold=0.035
#done


# threshold
#for threshold in $(seq .025 .005 .04);
#do
#  export CUDA_VISIBLE_DEVICES="7"
#  python mk_heatmap.py --yoshinobu --attention_type="last" --sample --color_level=3.5 --temp=0.031 --threshold=$threshold
#done

