import os
import argparse
import pandas as pd
import json
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


def objective(trial, train, valid):
    # ハイパーパラメータの範囲を定義
    param = {
        'tree_method': 'gpu_hist',  # GPUを使用
        'device': 'cuda:1',
        'lambda': trial.suggest_float('lambda', 1e-3, 10.0),
        'alpha': trial.suggest_float('alpha', 1e-3, 10.0),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.6, 0.7, 0.8, 0.9, 1.0]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.01, 0.02, 0.05, 0.1]),
        'n_estimators': trial.suggest_categorical('n_estimators', [100, 200, 300, 400, 500]),
        'max_depth': trial.suggest_categorical('max_depth', [3, 4, 5, 6, 7, 8, 9]),
        'random_state': 42,
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
    }
    train_x = train.drop(["labels"], axis=1)
    train_y = train["labels"]
    valid_x = valid.drop(["labels"], axis=1)
    valid_y = valid["labels"]

    # モデルの作成と訓練
    model = XGBClassifier(**param)
    model.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], verbose=True)
    model.set_params(early_stopping_rounds=50)

    # 検証データでの予測と精度の計算
    preds = model.predict(valid_x)
    accuracy = accuracy_score(valid_y, preds)
    return accuracy
    

def main(args):
    data = args.data
    
    data_dir = os.path.join("data", data)
    
    train = pd.read_pickle(os.path.join(data_dir, "train.pkl"))
    valid = pd.read_pickle(os.path.join(data_dir, "valid.pkl"))
    
    print("start optuna")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, train, valid), n_trials=100, n_jobs=10, show_progress_bar=True)
    
    best_params = study.best_params
    with open(os.path.join(data_dir, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="nested_day_mecab_parsed_ngram_100")
    
    args = parser.parse_args()
    main(args)