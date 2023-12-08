from typing import Any
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from src.model.AbstractModel import AbstractModel
from src.model.add_special_tokens_and_initialize import add_special_tokens_and_initialize


class HFModel(AbstractModel):
    def __init__(
        self, 
        pretrained_model_name_or_path,
        additional_special_tokens: list = ['<person>'], 
        personal_pronouns: list = ['君', 'きみ', 'あなた' ,'彼', '彼女'], 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()
    
        self.model = AutoModel.from_pretrained(pretrained_model_name_or_path)
        self.fn = nn.Linear(self.model.config.hidden_size, 2)
        self.criterion = nn.CrossEntropyLoss()
        
        # Resize for the <person> token
        if additional_special_tokens is not None and '<person>' in additional_special_tokens and len(additional_special_tokens) == 1:
            add_special_tokens_and_initialize(self.model, pretrained_model_name_or_path, additional_special_tokens, personal_pronouns)
        
    def forward(
        self, 
        input_ids, 
        attention_mask,
        output_attentions=False,
        labels=None,
    ):
        try:
            outputs = self.model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                output_attentions=output_attentions
            )
            preds = self.fn(outputs.pooler_output)
        except RuntimeError as e:
            self.logger.error(f"RuntimeError has occurred. {e}")
            raise e
        loss = None
        if labels is not None:
            loss = self.criterion(preds, labels)
        return dict(loss=loss, preds=preds, attentions=outputs.attentions)
