import pandas as pd
import os
import re
import pathlib


def main():
    JDC_corpus = pd.read_csv('../../corpus/JDC/all_dialog_modify.csv')
    dataset = []

    utters = []
    is_LIE = False
    lie_count = 0
    cur_subject = None
    cur_TOPIC = None

    save_path = 'data/JDC'
    os.makedirs(save_path, exist_ok=True)


    with open(os.path.join(save_path, 'debug.txt'), 'w') as f:

        for row in JDC_corpus.itertuples():
            if cur_subject != row.subject and str(row.subject) != 'nan':
                prev_subject = cur_subject
                cur_subject = row.subject

            if str(row.TOPIC) == 'nan' or row.TOPIC == 'N':
                continue
            elif cur_TOPIC == None: # First time
                cur_TOPIC = row.TOPIC

            if cur_TOPIC != row.TOPIC:
                lie_rate = lie_count / len(utters)
                dataset.append({'nested_utters': pd.DataFrame(utters, columns=['raw_nested_utters']), 'labels': is_LIE, 'lie_rate': lie_rate})
                print(f'subject:{cur_subject if prev_subject==None else prev_subject}, TOPIC:{cur_TOPIC}, is_LIE:{is_LIE}, lie_rate:{lie_rate}\n{utters}\n\n', file=f)
                prev_subject = None
                cur_TOPIC = row.TOPIC
                utters = []
                lie_count = 0
                is_LIE = 1 if row.LIE == 'LIE' else 0

            utter = re.sub(r'<SN>|<UNIN>|<MP>|<LG>', '', row._4)
            if row.LIE == 'LIE':
                lie_count += 1
            utters.append(utter)

        lie_rate = lie_count / len(utters)
        dataset.append({'nested_utters': pd.DataFrame(utters, columns=['raw_nested_utters']), 'labels': is_LIE, 'lie_rate': lie_rate})
        print(f'subject:{cur_subject if prev_subject==None else prev_subject}, TOPIC:{cur_TOPIC}, is_LIE:{is_LIE}, lie_rate:{lie_rate}\n{utters}\n\n', file=f)

    original_df = pd.DataFrame(dataset)
    pd.concat((original_df.query('labels==1').query('lie_rate>0.4'),original_df.query('labels==0').query('lie_rate<0.2')), axis=0).reset_index().to_pickle(os.path.join(save_path, 'test.pkl'))


if __name__ == "__main__":
    main()