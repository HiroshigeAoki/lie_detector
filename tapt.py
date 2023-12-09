from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import LineByLineTextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import torch
import os
import ast
import argparse

from src.model.add_special_tokens_and_initialize import add_special_tokens_and_initialize


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='micro')  # 'binary'から'micro'に変更
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default=[0,1], nargs='+')
    parser.add_argument("--data_dir_name", default="nested")
    parser.add_argument("--block_size", default=128, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--resume_from_checkpoint", action="store_true")
    args = parser.parse_args()
    
    # pathの設定
    MODEL_NAME = "nlp-waseda/bigbird-base-japanese"
    DATA_PATH = f"data/{args.data_dir_name}/bbs.txt"
    _model_name = MODEL_NAME.replace("/", "-")
    SAVE_DIR = f"model/{_model_name}/original"
    LOG_DIR = f"logs/{_model_name}_tapt"
    OUTPUT_DIR = SAVE_DIR.replace("_original", "_tapt")
    for _dir in [SAVE_DIR, LOG_DIR, OUTPUT_DIR]:
        os.makedirs(_dir, exist_ok=True)
    additional_special_tokens: list = ['<person>']
    personal_pronouns: list = ['君', 'きみ', 'あなた' ,'彼', '彼女']
    gpus = ast.literal_eval(args.gpus[0])

    # GPUの指定
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    device = torch.device('cuda')
    # 事前学習モデルの設定
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_tokens(additional_special_tokens)
    tokenizer.save_pretrained(SAVE_DIR)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    add_special_tokens_and_initialize(model, pretrained_model_name_or_path=MODEL_NAME, mode="mlm", additional_special_tokens=additional_special_tokens, personal_pronouns=personal_pronouns)
    model.to(device)

    # データセットの作成
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=DATA_PATH,
        block_size=args.block_size,
    )

    # データコレーターの作成
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )
    
    # トレーニングの設定
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=20,
        per_device_train_batch_size=args.batch_size,
        save_steps=5000,
        save_total_limit=5,
        logging_steps=10,
        logging_dir=LOG_DIR,
    )

    # トレーナーの作成と事前学習の実行
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        compute_metrics=compute_metrics
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(SAVE_DIR)

if __name__ == "__main__":
    main()
