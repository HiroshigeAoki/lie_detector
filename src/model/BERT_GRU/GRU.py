import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from src.model.AttentionWithContext import AttentionWithContext


class GRU(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        dropout: float, 
        batch_first: bool,
        num_layers: int = 1, 
        bidirectional: bool = False, 
        use_attention: bool = False,
        ):
        super().__init__()
        
        self.lstm = nn.GRU(
            input_dim, 
            hidden_dim, 
            num_layers, 
            bidirectional=bidirectional, 
            dropout=dropout, 
            batch_first=batch_first
        )
        
        self.attention = AttentionWithContext(hidden_dim)
    
    def forward(self, input_ids, lengths):
        # packed_data = rnn_utils.pack_padded_sequence(input_ids, lengths, batch_first=True, enforce_sorted=False)
        # packed_output = self.lstm(packed_data)
        # packed_output, (hidden, cell) = self.lstm(packed_data)
        # output, output_lengths = rnn_utils.pad_packed_sequence(packed_output, batch_first=True)
        packed_data = rnn_utils.pack_padded_sequence(input_ids, lengths, batch_first=True, enforce_sorted=False)
        output, h_n = self.lstm(packed_data)
        output, output_lengths = rnn_utils.pad_packed_sequence(output, batch_first=True)
        
        batch_size = output.shape[0]
        max_len = output.shape[1]
        attention_mask = (torch.arange(max_len).expand(batch_size, max_len) >= output_lengths.unsqueeze(1)).to(input_ids.device)
        attentions, weighted_outputs = self.attention(output, attention_mask)
        
        return dict(
            weighted_outputs=weighted_outputs,
            attentions=attentions
        )
