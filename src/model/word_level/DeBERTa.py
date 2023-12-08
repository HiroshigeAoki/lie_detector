import torch
import pytorch_lightning as pl
from transformers import DebertaV2Model, AutoTokenizer
from transformers.models.deberta_v2.modeling_deberta_v2 import ContextPooler


class DeBERTa(pl.LightningModule):
    def __init__(
        self,
        pretrained_model: str = 'ku-nlp/deberta-v2-base-japanese',
        additional_special_tokens: list = None,
        personal_pronouns: list = ['君', 'きみ', 'あなた' ,'彼', '彼女'],
        ):
        super(DeBERTa, self).__init__()
        
        self.model: DebertaV2Model = DebertaV2Model.from_pretrained(pretrained_model, return_dict=True)
        self.pooler = ContextPooler(self.model.config)
        self.dropout = torch.nn.Dropout(self.model.config.hidden_dropout_prob)

        # Resize for the <person> token
        if additional_special_tokens is not None and '<person>' in additional_special_tokens and len(additional_special_tokens) == 1:
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model, additional_special_tokens=list(additional_special_tokens))
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
        output_attentions=False
    ):
        outputs = self.model(
            input_ids, 
            token_type_ids=token_type_ids,
            attention_mask=attention_mask, 
            position_ids=position_ids,
            output_attentions=output_attentions)
        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        pooled_dropout = self.dropout(pooled_output)
        
        return dict(pooled_output=pooled_dropout, attentions=outputs.attentions)
    

import pytest
import torch
from transformers import AutoTokenizer

@pytest.fixture
def model():
    return DeBERTa(additional_special_tokens=['<person>'])

def test_initialization_with_person_token(model):
    # Ensure model is initialized
    assert isinstance(model, DeBERTa)

def test_initialization_with_invalid_token():
    # Ensure model raises an error for invalid additional tokens
    with pytest.raises(ValueError):
        DeBERTa(additional_special_tokens=['<invalid>'])

def test_forward_method(model):
    tokenizer = AutoTokenizer.from_pretrained('ku-nlp/deberta-v2-base-japanese', additional_special_tokens=['<person>'])
    input_text = ["これはテスト文です。", "これはもう一つのテスト文です。"]
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    
    outputs = model(**inputs)
    
    # Ensure output has the expected keys
    assert "pooled_output" in outputs
    assert "attentions" in outputs

    # Ensure output shape is as expected
    assert outputs["pooled_output"].shape == (len(input_text), model.model.config.hidden_size)

if __name__ == "__main__":
    pytest.main([__file__])
