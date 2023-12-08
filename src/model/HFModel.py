from typing import Any
import torch
from torch import nn
import hydra
from transformers import AutoTokenizer
from src.model.AbstractModel import AbstractModel


class HFModel(AbstractModel):
    def __init__(
        self, 
        model_params, 
        additional_special_tokens: list = ['<person>'], 
        personal_pronouns: list = ['君', 'きみ', 'あなた' ,'彼', '彼女'], 
        **kwargs
    ):
        super(HFModel).__init__(**kwargs)
        self.save_hyperparameters()
    
        self.model = hydra.utils.instantiate(model_params)
        self.classifier = nn.Linear(self.model.config.hidden_size, 2)
        
        # Resize for the <person> token
        if additional_special_tokens is not None and '<person>' in additional_special_tokens and len(additional_special_tokens) == 1:
            tokenizer = AutoTokenizer.from_pretrained(model_params["pretrained_model"], additional_special_tokens=list(additional_special_tokens))
            self.model.resize_token_embeddings(len(tokenizer))
            # initialize  <person> token by the average of some personal_pronouns's weights.
            personal_pronouns_weights = torch.stack([self.model.embeddings.word_embeddings.weight[i, :] for i in tokenizer.convert_tokens_to_ids(personal_pronouns)])
            self.model.embeddings.word_embeddings.weight.data[-1, :] = personal_pronouns_weights.mean(dim=0)
        else:
            raise ValueError(f"Additional tokens:{additional_special_tokens} except for the '<person>' token are currently not supported.")
        
    def forward(
        self, 
        input_ids, 
        attention_mask,
        token_type_ids=None,
        position_ids=None, 
        output_attentions=False,
        labels=None,
    ):
        outputs = self.model(
            input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids, 
            position_ids=position_ids,
            output_attentions=output_attentions
        )
        preds = self.classifier(outputs.pooler_output)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(preds, labels)
        return dict(loss=loss, preds=preds, attentions=outputs.attentions)
