import torch
import torch.nn as nn
import torch.nn.functional as F_func
import math

import os
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from core.models_fanformer import FANformerLayer

# scaling gradient
class StochasticDepth(nn.Module):
    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        
        # 隨機深度正則化
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        noise.floor_()  # 二值化噪聲
        return x.div(keep_prob) * noise

# 1.Multi-Scale Embedding (Adjusted for input shape (B, C, S, F))
class MultiScaleEmbedding(nn.Module):
    def __init__(self, patch_configs, embed_dim, in_channels, dropout = 0.1):
        """
        patch_configs: list of tuples，每個 tuple 為 (kernel_size, stride)
        embed_dim: 每個 patch 經過 conv1d 映射後的維度
        in_channels: 輸入的通道數 (C)
        input shape : (B, C, S, F) where C is channels, S is sequence length, F is features
        output shape : (B, total_tokens, F, embed_dim)
        """
        super(MultiScaleEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.patch_convs = nn.ModuleList()
        # add batch normalizatios
        self.batch_norms = nn.ModuleList()

        # add GELU activation functio
        self.act = nn.GELU()

        # Increase frequency sensoring abiliyty
        self.freq_weights = nn.Parameter(torch.ones(len(patch_configs) + 1, 1, 1, 1))

        # 添加初始化方法提高穩定性
        self.apply(self._init_weights)

        for kernel_size, stride in patch_configs:
            # 直接使用 C 作為 in_channels
            padding = kernel_size//2
            conv = nn.Conv1d(in_channels=in_channels, out_channels=embed_dim,
                             kernel_size=kernel_size, stride=stride, padding = padding, padding_mode = 'reflect')
            self.patch_convs.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(embed_dim))
            
        # 添加全局平均池化分支
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_proj = nn.Sequential(
            nn.Linear(in_channels, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        
    def forward(self, x):
        """
        x: (B, C, S, F)
        process : 
          1. 對每個 feature 單獨處理，將 x 轉換成 (B*F, C, S)
          2. 添加全局平均池化分支 (B, 1, F, embed_dim) 增加到 output
          3. 分別對每個 patch config 使用 conv1d 得到 (B*F, embed_dim, L)
          4. 重塑成 (B, F, embed_dim, L) 並調整順序為 (B, L, F, embed_dim)
          5. 將各尺度的結果在 tokens 維度連接，得到 (B, total_tokens + 1, F, embed_dim)
        """
        B, C, S, F = x.shape
        outputs = []
        
        # 轉換到 (B*F, C, S) - 每個 feature 單獨處理，但保留 channel 維度作為 conv1d 的輸入通道
        x_reshaped = x.permute(0, 3, 1, 2)  # (B, F, C, S)
        x_reshaped = x_reshaped.reshape(B * F, C, S)  # (B*F, C, S)

        # add global average pool
        global_feat = self.global_pool(x_reshaped).squeeze(-1) # (B*F, C)
        global_embed = self.global_proj(global_feat).view(B, F, self.embed_dim, 1)
        global_out = global_embed.permute(0, 3, 1, 2)  # (B, 1, F, embed_dim)
        outputs.append(global_out * self.freq_weights[0])  # 應用可學習的頻率權重

        for i, conv in enumerate(self.patch_convs):
            out = conv(x_reshaped)  # (B*F, embed_dim, L)
            out = self.act(self.batch_norms[i](out))  # 添加批次正規化和GELU激活
            L = out.shape[-1]
            out = out.view(B, F, self.embed_dim, L).permute(0, 3, 1, 2)
            outputs.append(out * self.freq_weights[i+1])
            
        out = torch.cat(outputs, dim=1)  # (B, total_tokens+1, F, embed_dim)
        return out
    
# 2. Positional Encoder
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

        # 添加可學習的縮放因子
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # x: [B, seq_len, ...]
        pos_enc = self.pe[:, :x.size(1), :]
        x = x + self.scale * pos_enc.unsqueeze(-2)
        return x  # (B, seq_len, F, dim)

# 3. Temporal Encoder 
class TemporalEncoder(nn.Module):
    def __init__(self, embed_dim, num_layers=1, num_heads=4, dropout=0.1):
        """
        針對每個 feature 對 token 序列進行 Transformer 編碼
        Input shape: (B, tokens, F, embed_dim)
        output shape: (B, tokens, F, embed_dim)
        """
        super(TemporalEncoder, self).__init__()
        # Multi layers Transformer
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(embed_dim),
                FANformerLayer(
                    hidden_dim=embed_dim, 
                    num_heads=num_heads, 
                    feedforward_dim=embed_dim*4,
                    attn_dropout=dropout,
                    proj_dropout=dropout,
                    dropout=dropout
                ),
                StochasticDepth(dropout)
            ]))
            
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        1. 將 x 從 (B, tokens, F, embed_dim) 轉為 (B, F, tokens, embed_dim)
        2. 合併 B 與 F 得到 (B*F, tokens, embed_dim)
        3. Transformer 編碼後再重塑回原狀： (B, tokens, F, embed_dim)
        """
        B, tokens, F, E = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * F, tokens, E)

        # Transformer encoder
        for norm, transformer, drop_path in self.layers:
            x = x + drop_path(transformer(norm(x)))

        x = self.out_norm(x)
        x = x.view(B, F, tokens, E).permute(0, 2, 1, 3)  # (B, tokens, F, E)
        return x
    
# 4.Channel-wise Encoder 
class ChannelWiseEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        """
        對於每個 token，跨 feature (channel) 進行多頭自注意力
        input shape shape: (B, tokens, F, embed_dim)
        output shape: (B, tokens, F, embed_dim)
        """
        super(ChannelWiseEncoder, self).__init__()
        # pre-normalization
        self.pre_norm1 = nn.LayerNorm(embed_dim)
        self.pre_norm2 = nn.LayerNorm(embed_dim)

        # Multi-head attention
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,
                                         dropout=dropout, batch_first=True)
        
        # gate mechanism
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )
        
        # Feedforward network
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        # Stochastic depth
        self.drop_path1 = StochasticDepth(dropout)
        self.drop_path2 = StochasticDepth(dropout)

    def forward(self, x):
        """
        將 x (B, tokens, F, embed_dim) 重塑為 (B*tokens, F, embed_dim)，
        然後進行 self-attention 及前饋網路，最後還原形狀。
        """
        B, tokens, F, E = x.shape
        x_reshaped = x.reshape(B * tokens, F, E)  # (B*tokens, F, E)

        # First layer : Multi head attention
        res = x_reshaped
        x_norm = self.pre_norm1(x_reshaped)
        attn_output, _ = self.mha(x_norm, x_norm, x_norm)
        gate_values = self.gate(attn_output)
        x_reshaped = res + self.drop_path1(gate_values * attn_output)
        
        # Second layer : Feedforward network
        res = x_reshaped
        x_norm = self.pre_norm2(x_reshaped)
        ff_output = self.ff(x_norm)
        x_reshaped = res + self.drop_path2(ff_output)

        x_out =  x_reshaped.reshape(B, tokens, F, E)

        return x_out

# 5. Multi-step Decoder 
class MultiStepDecoder(nn.Module):
    def __init__(self, embed_dim, patch_length, num_steps, hidden_dim=128):
        """
        將預測長度分為多個 patch (例如 patch_length=2)
        num_steps * patch_length = T_pred
        input shape: (B, F, embed_dim)
        output shape: (B, T_pred, F)
        """
        super(MultiStepDecoder, self).__init__()
        self.patch_length = patch_length
        self.num_steps = num_steps
        self.fc_in = nn.Linear(embed_dim + patch_length, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, patch_length)
        self.relu = nn.ReLU()

    def forward(self, context):
        """
        context: (B, F, embed_dim)
        逐步生成預測，每個 patch 輸出 shape (B, F, patch_length)
        最後輸出 shape 經轉置為 (B, T_pred, F)
        """
        B, F, E = context.shape
        device = context.device
        # 初始上一個預測 patch，形狀 (B, F, patch_length
        prev_patch = torch.zeros(B, F, self.patch_length, device=device, dtype=context.dtype)
        predictions = []
        for _ in range(self.num_steps):
            # 拼接 context 與上一個預測： (B, F, embed_dim + patch_length)
            inp = torch.cat([context, prev_patch], dim=-1)
            hidden = self.relu(self.fc_in(inp))  # (B, F, hidden_dim)
            patch_pred = self.fc_out(hidden)       # (B, F, patch_length)
            predictions.append(patch_pred)
            prev_patch = patch_pred  # 更新上一個 patch
        # 將各步預測連接： (B, F, patch_length * num_steps)
        out = torch.cat(predictions, dim=-1)
        # 轉換為 (B, T_pred, F) 其中 T_pred = patch_length * num_steps
        out = out.transpose(1, 2)
        return out

# 6. Single step decoder 
class SingleStepDecoder(nn.Module):
    def __init__(self, embed_dim, feature_shape, hidden_dim=128, out_dim=16, dropout = 0.1):
        """
        用於生成單一預測值
        input shape: (B, F, embed_dim)
        output shape: (B, out_dim)
        """
        super(SingleStepDecoder, self).__init__()
        # 使用多頭注意力處理特徵
        self.feature_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=min(4, embed_dim // 8),  # 確保頭數合適
            batch_first=True,
            dropout=dropout
        )
        
        self.feature_norm = nn.LayerNorm(embed_dim)
        
        # 預測層
        self.prediction = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),  # 降低最後的 dropout
            nn.Linear(hidden_dim // 2, out_dim)
        )
        
        # 輸出限制（用於穩定訓練）
        self.output_activation = nn.Tanh()
        self.output_scale = nn.Parameter(torch.ones(1) * 5.0)  # 可學習的輸出縮放
     

    def forward(self, context):
        """
        Args : 
            contexy : (B, F, embed_dim)
        Return:
            Output : (B, out_dim)
        """

        # 特徵注意力
        context_norm = self.feature_norm(context)
        attn_output, _ = self.feature_attn(
            context_norm, context_norm, context_norm
        )
        
        # 加權並聚合
        weighted_context = context + attn_output
        context_pooled = weighted_context.mean(dim=1)  # 平均池化
        
        # 預測
        output = self.prediction(context_pooled)
        
        # 訓練時不限制輸出範圍，測試時使用縮放的 tanh
        if not self.training:
            output = self.output_scale * self.output_activation(output / self.output_scale)
            
        return output

# 7. MultiPatchFormer (Adjusted for input shape (B, C, S, F))
class MultiPatchFormer(nn.Module):
    def __init__(self, patch_configs, feature_shape, embed_dim, in_channels=3,
                 temporal_layers=1, temporal_heads=4,
                 channel_heads=4, decoder_patch_length=2, decoder_steps=4, 
                 decoder_hidden_dim=128, decoder_out_dim=16, dropout=0.1):
        """
        本模型接受 (B, C, S, F) 格式的輸入
        input shape: (B, C, S, F) where C is channels, S is sequence length, F is features
        output shape: (B, out_dim) for single step decoder or (B, T_pred, F) for multi-step decoder
        
        patch_configs: e.g., [(2,1), (3,1)] 適用於短序列 S
        decoder_patch_length 與 decoder_steps 決定 T_pred = patch_length * num_steps
        """
        super(MultiPatchFormer, self).__init__()
        self.embedding = MultiScaleEmbedding(patch_configs, embed_dim, in_channels)
        self.pos_encoder = PositionalEncoding(embed_dim)
        self.temporal_encoder = TemporalEncoder(embed_dim, num_layers=temporal_layers,
                                                num_heads=temporal_heads, dropout=dropout)
        self.channel_encoder = ChannelWiseEncoder(embed_dim, num_heads=channel_heads, dropout=dropout)
        
        # tokens combined using attention method
        self.token_mha = nn.MultiheadAttention(
            embed_dim=feature_shape*embed_dim,  num_heads=4,    
            dropout=0.1,batch_first=True
        )

        # LayerNorm用於token注意力
        self.token_norm = nn.LayerNorm(feature_shape*embed_dim)

        # Multi step decoder:
        # self.decoder = MultiStepDecoder(embed_dim, patch_length=decoder_patch_length, num_steps=decoder_steps, hidden_dim=embed_dim*2)
        # Single value decoder:
        self.decoder = SingleStepDecoder(embed_dim, feature_shape=feature_shape, hidden_dim=decoder_hidden_dim, out_dim=decoder_out_dim)

    def forward(self, x):
        """
        x: (B, C, S, F)
        In process：
          1. 多尺度嵌入：輸出 (B, total_tokens, F, embed_dim)
          2. 時間編碼器：輸出 (B, total_tokens, F, embed_dim)
          3. 通道式編碼器：輸出 (B, total_tokens, F, embed_dim)
          4. 於 tokens 維度做加權平均，獲得 (B, F, embed_dim)
          5. 解碼器：輸出 (B, out_dim) 或 (B, T_pred, F)
        """
        x = self.embedding(x)                # (B, tokens, F, embed_dim)
        x = self.pos_encoder(x)              # (B, tokens, F, embed_dim)
        x = self.temporal_encoder(x)         # (B, tokens, F, embed_dim)
        x = self.channel_encoder(x)          # (B, tokens, F, embed_dim)
        
        # 添加注意力權重 weighted sum of tokens
        B, tokens, F, E = x.shape
        x_flat = x.reshape(B, tokens, -1)    # (B, tokens, F*E)
        x_flat = self.token_norm(x_flat)
        attn_output, attn_weights = self.token_mha(
            x_flat, x_flat, x_flat,
            need_weights=True,
            average_attn_weights=False  
        )
        weights = F_func.softmax(attn_weights.sum(dim=-2).mean(dim=1), dim=1)
        weighted_x = (x * weights.view(B, tokens, 1, 1))  # (B, tokens, F, E)
        context = weighted_x.sum(dim=1)      # (B, F, E)

        out = self.decoder(context)          # (B, out_dim) or (B, T_pred, F)
        return out.squeeze(-1)

if __name__ == '__main__':
    # Example usage with input shape (batch, channel, sequence, feature) = (64, 3, 10, 5)
    batch_size = 64
    C = 3  # channels
    S = 10  # sequence length
    F = 5  # features
    x = torch.randn(batch_size, C, S, F)
    
    # Example patch configs for sequence length S=10
    patch_configs = [(2, 1), (3, 1)]
    embed_dim = 32
    
    model = MultiPatchFormer(
        patch_configs=patch_configs,
        feature_shape=F,
        embed_dim=embed_dim,
        in_channels=C,
        temporal_layers=1,
        temporal_heads=4,
        channel_heads=4,
        decoder_patch_length=2,
        decoder_steps=4,
        decoder_hidden_dim=128,
        decoder_out_dim=1,
        dropout=0.1
    )
    
    output = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
