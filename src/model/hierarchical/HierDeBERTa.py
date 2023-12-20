from torch import nn
from src.model.hierarchical.AbstractHierModel import AbstractHierModel
from src.model.word_level.DeBERTa import DeBERTa
from src.model.sent_level.lstm import LSTM


class HierDeBERTa(AbstractHierModel):
    def __init__(
        self, 
        wordattennet_params: dict,
        sentattennet_params: dict,
        optim: dict, 
        dropout: int = 0.1,
        num_class: int = 2,
        use_gmail_notification: bool = False, 
        is_scam_game: bool = False,
        is_murder_mystery: bool = False,
        ):
        super().__init__(optim=optim, use_gmail_notification=use_gmail_notification, is_scam_game=is_scam_game, is_murder_mystery=is_murder_mystery)
        self.save_hyperparameters()

        self.build_wordattennet(**wordattennet_params)
        self.build_sentattennet(**sentattennet_params, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(sentattennet_params.hidden_dim*4, num_class)
        self.criterion = nn.CrossEntropyLoss()

    def build_wordattennet(self, *args, **kwargs):
        self.wordattennet = DeBERTa(*args, **kwargs)
    
    def build_sentattennet(self, hidden_dim: int, dropout: float):
        self.sentattennet = LSTM(
            input_dim=hidden_dim*4,
            hidden_dim=hidden_dim*4,
            dropout=dropout,
            batch_first=True,
        )
    
    def forward(self, input_ids, attention_mask, labels=None):
        wordattennet_outputs = self.wordattennet(input_ids, attention_mask)
        sentattennet_outputs = self.sentattennet(wordattennet_outputs["pooled_output"])
        dropout_outputs = self.dropout(sentattennet_outputs["last_hidden_state"]).squeeze()
        preds = self.fc(dropout_outputs)
        loss = self.criterion(preds, labels) if labels is not None else None
        
        return dict(
            loss=loss, 
            preds=preds, 
            word_attentions=wordattennet_outputs['attentions'], 
            sent_attentions=sentattennet_outputs['attentions']
        )
