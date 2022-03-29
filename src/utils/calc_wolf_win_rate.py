import json
import glob

filePath = '../../corpus/BBSjsons/*/*.json'

files = glob.glob(filePath)

cnt_wolf_win = 0
for file in files:
    with open(file) as f:
        data = json.load(f)
    if data.get('result')!='human':
        cnt_wolf_win += 1

print(f'{ cnt_wolf_win / len(files) * 100:.2f}%')

