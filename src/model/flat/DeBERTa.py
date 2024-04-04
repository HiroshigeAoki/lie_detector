import torch
import pytorch_lightning as pl
from transformers import DebertaV2Model
from src.model.pooling import MeanPooling
from src.model.add_special_tokens_and_initialize import (
    add_special_tokens_and_initialize,
)


class DeBERTa(pl.LightningModule):
    def __init__(
        self,
        pretrained_model_name_or_path: str = "ku-nlp/deberta-v2-base-japanese",
        additional_special_tokens: list = None,
        personal_pronouns: list = ["君", "きみ", "あなた", "彼", "彼女"],
        dropout: float = 0.1,
        use_attention: bool = False,
    ):
        super(DeBERTa, self).__init__()

        self.model: DebertaV2Model = DebertaV2Model.from_pretrained(
            pretrained_model_name_or_path, return_dict=True
        )
        self.pooler = MeanPooling()
        self.dropout = torch.nn.Dropout(dropout)
        self.use_attention = use_attention

        if (
            additional_special_tokens is not None
            and "<person>" in additional_special_tokens
            and len(additional_special_tokens) == 1
        ):
            add_special_tokens_and_initialize(
                self.model,
                pretrained_model_name_or_path,
                additional_special_tokens,
                personal_pronouns,
            )

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        position_ids=None,
        output_attentions=False,
    ):
        outputs = self.model(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=True,
        )
        hidden_layers = outputs.hidden_states[-1]
        # hidden_layers = torch.cat((outputs.hidden_states[-1], outputs.hidden_states[-2], outputs.hidden_states[-3], outputs.hidden_states[-4]), dim=-1)

        pooled_output = self.pooler(hidden_layers, attention_mask)
        pooled_dropout = self.dropout(pooled_output)

        return dict(
            pooled_output=pooled_dropout.unsqueeze(1), attentions=outputs.attentions
        )
