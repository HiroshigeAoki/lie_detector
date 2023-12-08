import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from dotenv import load_dotenv
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import ast

load_dotenv()

SAVE_DIR = "./data/scam_game"
ANNOTATION_DIR = os.path.join(SAVE_DIR, "annotation")

# 経過時間計測
def calculate_elapsed_time(df):
    df['ts'] = pd.to_datetime(df['ts'])
    df['elapsed_time'] = df['ts'].diff().dt.total_seconds()
    total_seconds = df['elapsed_time'].sum()
    return str(timedelta(seconds=int(total_seconds)))


def main():
    os.makedirs(ANNOTATION_DIR, exist_ok=True)

    # Ggoogle API認証
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive',
             'https://www.googleapis.com/auth/drive.file']
    creds = ServiceAccountCredentials.from_json_keyfile_name(
        os.getenv('GCP_API_KEY_FILE_PATH'), scope)
    google_client = gspread.authorize(creds)

    # マスターシートを取得し、データフレームに変換
    master_spreadsheet = google_client.open_by_key(os.getenv("MASTER_SHEET_KEY"))
    master_sheet = master_spreadsheet.sheet1.get_all_values()
    master_df = pd.DataFrame(master_sheet[1:], columns=master_sheet[0])
    # "game"から始まるチャンネル名のみ抽出
    # master_df = master_df[master_df['channel_name'].str.lower().str.startswith('game')]
    
    # target_channels = ['test18', 'test19', 'test20', 'test21', 'test22', 'test23', 'test24', 'test25']
    target_channels = ['test26', 'test27', 'test28', 'test29']
    master_df = master_df[master_df['channel_name'].isin(target_channels)]
    
    master_df = master_df[master_df["finish"] == "TRUE"]
    master_df = master_df.drop(["customer_email", "sales_email"], axis=1)

    # アノテーションシートを取得し、データフレームに変換し保存
    annotation_spreadsheet = google_client.open_by_key(os.getenv("SPREAD_SHEET_KEY"))

    liar_dfs, honest_dfs = [], []

    # masterシートにchannel_nameと対応させて統計量と経過時間を保存するため、辞書を利用
    num_utterances_dict, num_sales_utters_dict, num_customers_utters_dict = {}, {}, {}
    num_lie_dict, num_suspicious_dict, num_detected_lie_dict, num_undetected_lie_dict = {}, {}, {}, {}
    elapsed_time_dict = {}

    for worksheet in annotation_spreadsheet.worksheets():
        # "game"から始まるチャンネル名のアノテーションのみ抽出
        if worksheet.title not in master_df["channel_name"].to_list():
            continue
        sheet = worksheet.get_all_values()
        df = pd.DataFrame(sheet[1:], columns=sheet[0])
        df.to_csv(os.path.join(ANNOTATION_DIR, f"{worksheet.title}.csv"), index=False)
        
        # 統計量を保存
        num_utterances = len(df)
        num_sales_utters = len(df[df["role"] == "sales"])
        num_customers_utters = len(df[df["role"] == "customer"])
        num_lie = len(df[df["lie"] == "TRUE"])
        num_suspicious = len(df[df["suspicious"] == "TRUE"])
        num_detected_lie = len(df[(df["lie"] == "TRUE") & (df["suspicious"] == "TRUE")])
        num_undetected_lie = num_lie - num_detected_lie
        
        num_utterances_dict[worksheet.title] = num_utterances
        num_lie_dict[worksheet.title] = num_lie
        num_sales_utters_dict[worksheet.title] = num_sales_utters
        num_customers_utters_dict[worksheet.title] = num_customers_utters
        num_suspicious_dict[worksheet.title] = num_suspicious
        num_detected_lie_dict[worksheet.title] = num_detected_lie
        num_undetected_lie_dict[worksheet.title] = num_undetected_lie
        
        elapsed_time_dict[worksheet.title] = calculate_elapsed_time(df)

        is_liar = master_df.loc[master_df["channel_name"]== worksheet.title, "is_liar"]
        assert is_liar.shape[0] == 1
        if is_liar.values[0] == "TRUE":
            liar_dfs.append(df)
        else:
            honest_dfs.append(df)

    # masterシートに統計量を追加
    master_df["num_utterances"] = master_df['channel_name'].map(num_utterances_dict)
    master_df["num_sales_utters"] = master_df['channel_name'].map(num_sales_utters_dict)
    master_df["num_customers_utters"] = master_df['channel_name'].map(num_customers_utters_dict)
    master_df["num_lie"] = master_df['channel_name'].map(num_lie_dict)
    master_df["num_suspicious"] = master_df['channel_name'].map(num_suspicious_dict)
    master_df["num_detected_lie"] = master_df['channel_name'].map(num_detected_lie_dict)
    master_df["num_undetected_lie"] = master_df['channel_name'].map(num_undetected_lie_dict)
    master_df["elapsed_time"] = master_df['channel_name'].map(elapsed_time_dict)
    master_df.reset_index(drop=True)
    master_df.to_csv(os.path.join(SAVE_DIR, "master.csv"), index=False)


    result = (
        f"正例数:{len(master_df[master_df['is_liar']=='FALSE'])}\n"
        f"負例数:{len(master_df[master_df['is_liar']=='TRUE'])}\n"
        "発話統計\n"
        f"総発話数: {master_df['num_utterances'].sum()}\n"
        f"\t- 営業役: {master_df['num_sales_utters'].sum()}\n"
        f"\t- 客役: {master_df['num_customers_utters'].sum()}\n"
        f"嘘の発話数: {master_df['num_lie'].sum()}\n"
        f"怪しまれた発話数: {master_df['num_suspicious'].sum()}\n"
        f"捉えられた嘘: {master_df['num_detected_lie'].sum()}\n"
        f"捉えられなかった嘘: {master_df['num_undetected_lie'].sum()}\n"
        f"総発話数に対する嘘の発話数の割合: {master_df['num_lie'].sum()/master_df['num_utterances'].sum():.2f}\n"
        f"総発話数に対する怪しまれた発話数の割合: {master_df['num_suspicious'].sum()/master_df['num_utterances'].sum():.2f}\n"
        #f"発話を嘘かどうか判定できたmatrix\n"
        #f"\t- Accuracy: {accuracy_score(master_df['num_lie'].sum(), master_df['num_suspicious'].sum()):.2f}\n"
        #f"\t- Precision: {precision_score(master_df['num_lie'].sum(), master_df['num_suspicious'].sum()):.2f}\n"
        #f"\t- Recall: {recall_score(master_df['num_lie'].sum(), master_df['num_suspicious'].sum()):.2f}\n"
        #f"\t- F1: {f1_score(master_df['num_lie'].sum(), master_df['num_suspicious'].sum()):.2f}\n"
        #f"confusion matrix\n"
        #f"{confusion_matrix(master_df['num_lie'].sum(), master_df['num_suspicious'].sum())}\n"
    )

    with open("./data/scam_game/stats.txt", "w") as f:
        f.write(result)
    
    nested_utters, labels = [], []
    judges, judge_reason = [], []
    lie_columms, suspicious_columns, suspicious_reasons, channel_names = [], [], [], []
    for channel_name in  master_df["channel_name"].to_list():
        df = pd.read_csv(os.path.join(ANNOTATION_DIR, f"{channel_name}.csv"))
        is_liar = master_df.loc[master_df["channel_name"]==channel_name, "is_liar"].values[0]
        label = 1 if is_liar=='TRUE' else 0
        df_sales = df[df['role'] == 'sales']
        nested_utters.append(df_sales['message'].to_list())
        labels.append(label)
        judge = master_df.loc[master_df["channel_name"]==channel_name, "judge"].values[0]
        judge = 0 if judge=='trust' else 1
        judges.append(judge)
        judge_reason.append(master_df.loc[master_df["channel_name"]==channel_name, "reason"].values[0])
        lie_columms.append(df_sales['lie'].to_list())
        suspicious_columns.append(df_sales['suspicious'].to_list())
        suspicious_reasons.append(list(map(str, df_sales['reason'].to_list())))
        channel_names.append(channel_name)

    df = pd.DataFrame({
        'nested_utters': nested_utters,
        'labels': labels,
        'judge': judges,
        'judge_reason': judge_reason,
        'lie': lie_columms,
        'suspicious': suspicious_columns,
        'suspicious_reasons': suspicious_reasons,
        'channel_name': channel_names})
    #df = pd.DataFrame({'nested_utters': nested_utters, 'labels': labels})
    df.to_csv(os.path.join(SAVE_DIR, "test.csv"), index=False)
    df['nested_utters'] = [pd.DataFrame({"raw_nested_utters": utter}) for utter in df['nested_utters']]
    df.to_pickle(os.path.join(SAVE_DIR, "test.pkl"))


if __name__ == "__main__":
    main()
