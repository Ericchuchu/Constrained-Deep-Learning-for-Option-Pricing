import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from core.models_convlstm import ConvLSTMCell
from core.models_multi_patch_former_adjusted import MultiPatchFormer

class DatePositionalEncoding(nn.Module):
    """Date tensor positional encoding optimized by precomputing constants."""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # 將 div_term 預先計算並作為 buffer 儲存
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)

    def forward(self, date_positions):
        """
        date_positions: [batch_size, seq_len]
        returns: [batch_size, seq_len, d_model]
        """
        # 利用 broadcasting 自動對齊形狀
        # date_positions.unsqueeze(-1): [B, channel, seq_len, 1]
        # self.div_term: [d_model/2]，經 broadcasting 變成 [B, channel, seq_len, d_model/2]
        pe = torch.zeros(date_positions.size(0), date_positions.size(1), date_positions.size(2), self.d_model, device=date_positions.device)
        # 使用 broadcasting 計算 sin 和 cos，不需要手動 unsqueeze div_term
        pe[:, :, :, 0::2] = torch.sin(date_positions.unsqueeze(-1).float() * self.div_term)
        pe[:, :, :, 1::2] = torch.cos(date_positions.unsqueeze(-1).float() * self.div_term)
        return pe

class PositionalEncoding(nn.Module):
    """Standard sequence positional encoding with precomputed pe."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 預計算所有位置的編碼
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, 1, max_len, d_model)
        pe[0, :, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        returns: [batch_size, seq_len, d_model] with positional encoding added.
        """
        seq_len = x.size(1)
        # 直接從預計算的 pe 中取前 seq_len 個
        x = x + self.pe[:, :, :seq_len, :]
        return self.dropout(x)
    
class ConstrainedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.raw_weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
    def forward(self, x):
        weight = torch.nn.functional.softplus(self.raw_weight)
        return torch.nn.functional.linear(x, weight, self.bias)

class MultPatchFormerEncoder(nn.Module):
    def __init__(self, d_model=16, nhead=4, num_layers=2, dim_feedforward=64, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Constrained linear input projection for moneyness
        self.moneyness_proj = ConstrainedLinear(1, d_model) 
        
        # Date positional encoding
        self.date_pos_encoder = DatePositionalEncoding(d_model) 
        
        # Sequence positional encoding
        self.seq_pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Fusion projection: 將 concat 後的向量從 2*d_model 投影回 d_model
        self.fusion_proj = nn.Linear(2 * d_model, d_model)
        
        # Transformer encoder layer
        self.multi_patch_former = MultiPatchFormer(patch_configs=[(2, 1), (3, 1)],
                                                    feature_shape=d_model,
                                                    embed_dim=32,
                                                    in_channels=1,
                                                    temporal_layers=1,
                                                    temporal_heads=4,
                                                    channel_heads=4,
                                                    decoder_patch_length=2,
                                                    decoder_steps=4,
                                                    decoder_hidden_dim=128,
                                                    decoder_out_dim=d_model,
                                                    dropout=0.1)
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, 2] (first vector: moneyness, second vector: date_position)
        moneyness = x[:, 0:1, :, 0:1]  # shape: [batch_size, channel, seq_len, 1]
        date_positions = x[:, 0:1, :, 1].long()  # shape: [batch_size, channel, seq_len]
        
        # Project moneyness
        moneyness_encoded = self.moneyness_proj(moneyness)  # [batch_size, chnnel, seq_len, d_model]
        
        # Get date positional encodings
        date_pos = self.date_pos_encoder(date_positions)  # [batch_size, channel, seq_len, d_model]
        
        # Concat moneyness features with date positional encoding 
        x = torch.cat([moneyness_encoded, date_pos], dim=-1)  # [batch_size, channel,seq_len, 2*d_model]
        
        # Project the concatenated features back to d_model
        x = self.fusion_proj(x)  # [batch_size, channel,seq_len, d_model]
        
        # Add sequence positional encoding
        x = self.seq_pos_encoder(x)

        # Pass through transformer encoder
        output = self.multi_patch_former(x)  # [batch_size, d_model]
        
        return output

class ConvLSTM(nn.Module):
    def __init__(self, input_channels=3, hidden_channels=[16, 8, 1], kernel_size=3, in_dim=5, out_dim=25, step=10):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self._all_layers = []
        self.linear1 = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.linear2 = nn.Linear(in_features=in_dim+2, out_features=out_dim)

        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input):
        # input: (N, C, T, D_in)
        input = input.requires_grad_()
        bsize, _, seq_len, _ = input.size()
        device = input.device
        
        # initialize hidden states of all layers
        hidden_states = []
        for i in range(self.num_layers):
            cell = getattr(self, f'cell{i}')
            h, c = cell.init_hidden(bsize, cell.hidden_channels, input.size(-1))
            hidden_states.append((h.to(device), c.to(device)))

        outputs = []
        t_tensor = None
        S_tensor = None
        for time_idx in range(seq_len):
            x = input[:, :, time_idx, :]  # shape: [batch_size, channels, D_in]

            for i in range(self.num_layers):
                h, c = hidden_states[i]
                h_new, c_new = getattr(self, f'cell{i}')(x, h, c)
                hidden_states[i] = (h_new, c_new)
                x = h_new  

            if time_idx == (seq_len - 1):
                # Extract t and S directly from input to maintain gradient chain
                t_tensor = input[:, 0, time_idx, 2].view(-1, 1, 1)  # shape: [batch_size, 1, 1]
                S_tensor = input[:, 2, time_idx, 2].view(-1, 1, 1)  # shape: [batch_size, 1, 1]

                # Ensure these tensors require gradients
                t_tensor.requires_grad_(True)
                S_tensor.requires_grad_(True)
                x = torch.cat([x, t_tensor, S_tensor], dim=-1)
                outputs.append(self.linear2(x))
            else:   
                outputs.append(self.linear1(x))

        final_output = torch.stack(outputs, dim=1)[:, -1, :].squeeze()
        return (final_output, t_tensor, S_tensor)

class DualBranchNetwork(nn.Module):
    def __init__(self, convlstm_channels=[16, 8, 1], transformer_dim=32, transformer_heads=4,
                 transformer_layers=2, transformer_ff_dim=128, dropout=0.1):
        super().__init__()
        
        # ConvLSTM branch
        self.convlstm = ConvLSTM(
            input_channels=3,
            hidden_channels=convlstm_channels,
            kernel_size=3,
            in_dim=5,  
            out_dim=25  # Keep same dimension
        )
        
        # Multi patch former  branch
        self.transformer = MultPatchFormerEncoder(
            d_model=transformer_dim,
            nhead=transformer_heads,
            num_layers=transformer_layers,
            dim_feedforward=transformer_ff_dim,
            dropout=dropout
        )
        
        # Fusion layers
        convlstm_output_dim = convlstm_channels[-1] * 25  # Assuming 5x5 spatial dimension
        self.fusion = nn.Sequential(
            nn.Linear(convlstm_output_dim + transformer_dim , 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
    def forward(self, x_convlstm, x_transformer):
        # Process ConvLSTM branch
        convlstm_out,t,S = self.convlstm(x_convlstm)  # [batch_size, out_dim]
        
        # Process Transformer branch
        transformer_out = self.transformer(x_transformer)  # [batch_size, transformer_dim]
        
        # Concatenate and fuse
        combined = torch.cat([convlstm_out, transformer_out], dim=1)
        output = self.fusion(combined)

        # Set retain_grad for t and S
        t.retain_grad()
        S.retain_grad()
        
        return (output.squeeze(-1),t,S)

# test
if __name__ == '__main__':
    input_convlstm = torch.normal(0, 1, size=(64, 3, 10, 5))
    input_transformer = torch.normal(0, 1, size=(64, 1, 10, 2))
    model = DualBranchNetwork()
    output = model(input_convlstm, input_transformer)
    print(f'Output shape : {output[0].shape}' )
