from transformers import AutoTokenizer
import torch


def add_special_tokens_and_initialize(
        model, 
        pretrained_model_name_or_path, 
        additional_special_tokens: list = ['<person>'], 
        personal_pronouns: list = ['君', 'きみ', 'あなた' ,'彼', '彼女'], 
        mode="fine-tuning",
    ):
    if additional_special_tokens is not None and '<person>' in additional_special_tokens and len(additional_special_tokens) == 1:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, additional_special_tokens=list(additional_special_tokens))
        model.resize_token_embeddings(len(tokenizer))
        
        # 個人代名詞の埋め込みの平均を取得
        # '<person>' トークンの埋め込みを初期化
        if mode == "fine-tuning":
            personal_pronouns_weights = torch.stack([model.embeddings.word_embeddings.weight[i, :] for i in tokenizer.convert_tokens_to_ids(personal_pronouns)])
            model.embeddings.word_embeddings.weight.data[-1, :] = personal_pronouns_weights.mean(dim=0)
        elif mode == "mlm":
            personal_pronouns_weights = torch.stack([model.base_model.embeddings.word_embeddings.weight[i, :] for i in tokenizer.convert_tokens_to_ids(personal_pronouns)])
            model.base_model.embeddings.word_embeddings.weight.data[-1, :] = personal_pronouns_weights.mean(dim=0)
            
    else:
        raise ValueError(f"Additional tokens:{additional_special_tokens} except for the '<person>' token are currently not supported.")
