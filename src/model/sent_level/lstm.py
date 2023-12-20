from torch import nn
from src.model.pooling import MeanPooling


class LSTM(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        dropout: float, 
        batch_first: bool,
        num_layers: int = 1, 
        bidirectional: bool = False, 
        use_attention: bool = False
        ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            bidirectional=bidirectional, 
            dropout=dropout, 
            batch_first=batch_first
        )
        #self.pooling =  MeanPooling()
    
    def forward(self, x):
        outputs = self.lstm(x)
        #pooled_outputs = self.pooling(outputs[0])
        return dict(
            #pooled_outputs=pooled_outputs,
            last_hidden_state=outputs[0], 
            last_cell_states=outputs[1],
            attentions=None # TODO
        )
