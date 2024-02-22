import os
import pandas as pd
import pickle
from xgboost import XGBClassifier, plot_importance
import torch
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score, ConfusionMatrix
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pickle
import shap
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"

# model = XGBClassifier()
# model.load_model("outputs/nested_day_xgboost_160000_100/xgboost/baseline/2024-01-22_145248/model.pkl")

class XGBoost:
    def __init__(self, data_dir, best_params=None, trained_model_path=None):
        if trained_model_path:
            self.model = XGBClassifier(**best_params)
            self.model.load_model(trained_model_path)
        else:
            self.model = XGBClassifier(**best_params)
        
        self.data_dir = data_dir
        self.test_metrics = MetricCollection([
            Accuracy(task="binary", num_classes=2, average='macro'),
            Precision(task="binary", num_classes=2, average='macro'),
            Recall(task="binary", num_classes=2, average='macro'),
            F1Score(task="binary", num_classes=2, average='macro')
        ])
        self.cm = ConfusionMatrix(task="binary", num_classes=2)
    
    def train(self, X, y):
        self.model.fit(X, y, verbose=True)
        self.model.save_model("model.ubj")
        self.model.save_model("model.json")
    
    def test(self, X, y, postfix=""):
        y_pred = self.model.predict(X)
        if hasattr(self.model, 'feature_importances_'):
            self.save_results(y, y_pred, postfix)
            # self.shape_summary_plot(X, y)
    
    def save_results(self, test_y, y_pred, postfix=""):
        test_y = torch.tensor(test_y)
        y_pred = torch.tensor(y_pred)
        self.test_metrics(y_pred, test_y)
        self.cm(y_pred, test_y)
        test_metrics = self.test_metrics.compute()
        pd.DataFrame([metrics.cpu().numpy() for metrics in test_metrics.values()], index=test_metrics.keys()).to_csv(f'scores{postfix}.csv')
        cm = self.cm.compute().numpy()
        pd.DataFrame(cm).to_csv("confusiond_matrix.csv", index=False)
        accuracy = accuracy_score(test_y, y_pred)
        precision_recall_f1_support_df = pd.DataFrame(list(precision_recall_fscore_support(test_y, y_pred)), index=["precision", "recall", "f1", "support"]).T
        confusion_matrix_df = pd.DataFrame(confusion_matrix(test_y, y_pred))
        
        print(f"Test Data Accuracy{postfix}: {accuracy}")
        print(precision_recall_f1_support_df)
        print(confusion_matrix_df)
        
        with open(f"result{postfix}.txt", "w") as f:
            f.write(f"accuracy: {accuracy}\n")
            f.write(f"{precision_recall_f1_support_df}\n")
            f.write(f"{confusion_matrix_df}\n")
        
        # weight: 特徴量が分岐に使われた回数。
        # gain: 特徴量が使用された際にどれだけ効果があったか。
        # cover: 特徴量によるデータのカバレッジ。
        importance_types = ['weight', 'gain', 'cover']
        for importance_type in importance_types:
            importance = self.model.get_booster().get_score(importance_type=importance_type)
            importance_df = pd.DataFrame(list(importance.items()), columns=['Feature', importance_type])
            importance_df.sort_values(by=importance_type, ascending=False, inplace=True)
            importance_df.to_csv(f'feature_importance_{importance_type}{postfix}.csv', index=False)
        
        importance_types = ['weight', 'gain', 'cover']
        for importance_type in importance_types:
            fig, ax = plt.subplots(figsize=(10, 8))
            plot_importance(self.model, ax=ax, importance_type=importance_type, title=f'Feature Importance{postfix} ({importance_type})')
            plt.show()

        # 画像ファイルとして保存
        fig.savefig(f"feature_importance_{importance_type}.png")
        self.test_metrics.reset()
        self.cm.reset()

    def shape_summary_plot(self, X, y):
        try:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_interaction_values(X, y)
            shap.summary_plot(shap_values, X, plot_type="bar", max_display=100)
            plt.savefig("shap_summary_bar.png")
            plt.clf()
            shap.summary_plot(shap_values, X)
            plt.savefig("shap_summary_default.png")
            plt.clf()
        except Exception as e:
            raise e
