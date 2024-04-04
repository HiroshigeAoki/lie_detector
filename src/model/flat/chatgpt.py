import json
import torch
import pandas as pd
import numpy as np
from torchmetrics import (
    MetricCollection,
    Accuracy,
    Precision,
    Recall,
    F1Score,
    ConfusionMatrix,
)
from sklearn.metrics import precision_recall_fscore_support
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

import json
import dotenv

dotenv.load_dotenv()


class ChatGPT:
    def __init__(
        self,
        model_name,
        max_tokens,
        model_outputs_dict=dict(
            indices=[], preds=[], labels=[], reasons=[]
        ),  # system_fingerprint=[]),
    ):
        self.model = ChatOpenAI(
            model_name=model_name,
            max_tokens=max_tokens,
        )
        self.model_outputs_dict = model_outputs_dict
        self.model = ChatOpenAI(
            model_name=model_name,
            max_tokens=max_tokens,
        )

        metrics = MetricCollection(
            [
                Accuracy(task="binary", num_classes=2, average="macro"),
                Precision(task="binary", num_classes=2, average="macro"),
                Recall(task="binary", num_classes=2, average="macro"),
                F1Score(task="binary", num_classes=2, average="macro"),
            ]
        )
        self.test_metrics = metrics.clone(prefix="test_")
        self.cm = ConfusionMatrix(task="binary", num_classes=2)

    def generate_prompt(self, utterances):
        prompt_template = PromptTemplate.from_template(
            template="".join(
                [
                    "#指示\n",
                    "あなたは、発言のスタイルから相手が嘘を付いているかどうかがわかるAIです。\n",
                    "これから、下記のある人のゲーム内での[発言]を読んで、嘘をついているかどうかを判断して頂きます。\n",
                    "下記の[出力形式]と[制約条件]に従い、嘘をついている場合は「1」を、嘘をついていない場合は「0」を出力してください。またその根拠も合わせて出力してください。\n\n",
                    "# 発言\n",
                    "{utterances}\n\n",
                    "# 出力形式\n",
                    "回答は、`|,|`で区切って、[回答]|,|[理由]の潤に出力してください。例: `1|,|~が嘘付きの典型的な特徴であるから。`\n\n",
                    "#制約条件\n",
                    "*嘘つきかどうか決める際、発話の内容(矛盾点等)ではなく、文体や表現、言い回し等を基に判断してください。",
                ]
            )
        )
        prompt = prompt_template.format(
            utterances=utterances,
        )
        return prompt

    def forward(self, utterances):
        try:
            output = self.model.invoke(
                self.generate_prompt(utterances),
            )
            answer, reason = output.content.split("|,|")
            return dict(
                answer=int(answer),
                reason=reason,
            )
        except ValueError as e:
            raise ValueError(f"Error: {e}\nOutput: {output}")

        except Exception as e:
            raise Exception(f"Error: {e}\nutterances: {utterances}")

    def test_step(self, index, utterances, label):
        try:
            output = self.forward(utterances)
            self.model_outputs_dict["indices"].append(index)
            self.model_outputs_dict["preds"].append(output["answer"])
            self.model_outputs_dict["labels"].append(label)
            self.model_outputs_dict["reasons"].append(output["reason"])
            # self.model_outputs_dict["system_fingerprint"].append(output["system_fingerprint"])
            with open("output.json", "w", encoding="utf-8") as f:
                json.dump(self.model_outputs_dict, f, ensure_ascii=False, indent=4)
        except Exception as e:
            raise e

    def on_test_end(self):
        preds = torch.tensor(self.model_outputs_dict["preds"])
        labels = torch.tensor(self.model_outputs_dict["labels"])
        self.test_metrics(preds, labels)
        self.cm(preds, labels)
        test_metrics = self.test_metrics.compute()
        pd.DataFrame(
            [metrics.cpu().numpy() for metrics in test_metrics.values()],
            index=test_metrics.keys(),
        ).to_csv("scores.csv")
        cm = self.cm.compute().numpy()
        pd.DataFrame(cm).to_csv("confusiond_matrix.csv", index=False)
        scores_df = pd.DataFrame(
            np.array(precision_recall_fscore_support(labels.numpy(), preds.numpy())).T,
            columns=["precision", "recall", "f1", "support"],
        )
        scores_df.to_csv("precision_recall_fscore_support.csv")
