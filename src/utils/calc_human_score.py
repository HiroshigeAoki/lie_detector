import os
import json
import pandas as pd
import torch
import numpy as np
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score, ConfusionMatrix
from sklearn.metrics import precision_recall_fscore_support


def main():
    data = "murder_mystery"
    output_dir = f"outputs/{data}/human"
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = MetricCollection([
            Accuracy(task="binary", num_classes=2, average='macro'),
            Precision(task="binary", num_classes=2, average='macro'),
            Recall(task="binary", num_classes=2, average='macro'),
            F1Score(task="binary", num_classes=2, average='macro')
        ])
    cm = ConfusionMatrix(task="binary", num_classes=2)
    
    # 犯人モ含め、全員の予測先をpredsには入れる。←だめ
    preds, labels, channel_names = [], [], []
    with open("data/murder_mystery/whole_log.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            criminal = data["criminal"]
            channel_name = data["channel_name"]
            vote_targets = [value for key, value in data.items() if key.startswith("vote_target_")]
            speakers = list(set(data["speakers"]))
            # Oversampling
            num_non_criminal = len(speakers) - 2
            speakers.extend([criminal] * num_non_criminal)
            for speaker in speakers:
                label = 1 if speaker==criminal else 0
                for vote_target in vote_targets:
                    pred = 1 if vote_target==speaker else 0
                    preds.append(pred)
                    labels.append(label)
                    channel_names.append(channel_name)
                    
    preds = torch.tensor(preds)
    labels = torch.tensor(labels)
    metrics(preds, labels)
    cm(preds, labels)
    metrics = metrics.compute()
    pd.DataFrame([metrics.cpu().numpy() for metrics in metrics.values()], index=metrics.keys()).to_csv(f'{output_dir}/scores.csv')
    cm = cm.compute().numpy()
    pd.DataFrame(cm).to_csv(f"{output_dir}/confusiond_matrix.csv", index=False)
    scores_df = pd.DataFrame(
        np.array(precision_recall_fscore_support(labels.numpy(), preds.numpy())).T,
        columns=["precision", "recall", "f1", "support"])
    scores_df.to_csv(f'{output_dir}/precision_recall_fscore_support.csv')
    pd.DataFrame(dict(preds=preds, labels=labels, channel_names=channel_names)).to_csv(f'{output_dir}/preds_labels.csv', index=False)


if __name__ == "__main__":
    main()