import torch
import torch.nn as nn
import torch.nn.functional as F

# 1.Multi-Scale Embedding (Adjusted for input shape (B, C, S, F))
class MultiScaleEmbedding(nn.Module):
    def __init__(self, patch_configs, embed_dim, in_channels):
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
        for kernel_size, stride in patch_configs:
            # 直接使用 C 作為 in_channels
            conv = nn.Conv1d(in_channels=in_channels, out_channels=embed_dim,
                             kernel_size=kernel_size, stride=stride)
            self.patch_convs.append(conv)

    def forward(self, x):
        """
        x: (B, C, S, F)
        process : 
          1. 對每個 feature 單獨處理，將 x 轉換成 (B*F, C, S)
          2. 分別對每個 patch config 使用 conv1d 得到 (B*F, embed_dim, L)
          3. 重塑成 (B, F, embed_dim, L) 並調整順序為 (B, L, F, embed_dim)
          4. 將各尺度的結果在 tokens 維度連接，得到 (B, total_tokens, F, embed_dim)
        """
        B, C, S, F = x.shape
        outputs = []
        
        # 轉換到 (B*F, C, S) - 每個 feature 單獨處理，但保留 channel 維度作為 conv1d 的輸入通道
        x_reshaped = x.permute(0, 3, 1, 2)  # (B, F, C, S)
        x_reshaped = x_reshaped.reshape(B * F, C, S)  # (B*F, C, S)
        
        for conv in self.patch_convs:
            out = conv(x_reshaped)  # (B*F, embed_dim, L)
            L = out.shape[-1]
            # 重塑回 (B, F, embed_dim, L) 並調整順序 -> (B, L, F, embed_dim)
            out = out.view(B, F, self.embed_dim, L).permute(0, 3, 1, 2)
            outputs.append(out)
            
        # 沿 tokens 維度連接
        out = torch.cat(outputs, dim=1)  # (B, total_tokens, F, embed_dim)
        return out

# 2. Temporal Encoder (No changes needed)
class TemporalEncoder(nn.Module):
    def __init__(self, embed_dim, num_layers=1, num_heads=4, dropout=0.1):
        """
        針對每個 feature 對 token 序列進行 Transformer 編碼
        Input shape: (B, tokens, F, embed_dim)
        output shape: (B, tokens, F, embed_dim)
        """
        super(TemporalEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """
        1. 將 x 從 (B, tokens, F, embed_dim) 轉為 (B, F, tokens, embed_dim)
        2. 合併 B 與 F 得到 (B*F, tokens, embed_dim)
        3. Transformer 編碼後再重塑回原狀： (B, tokens, F, embed_dim)
        """
        B, tokens, F, E = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B * F, tokens, E)
        x = self.transformer_encoder(x)  # (B*F, tokens, E)
        x = x.view(B, F, tokens, E).permute(0, 2, 1, 3)  # (B, tokens, F, E)
        return x

# 3.Channel-wise Encoder (No changes needed)
class ChannelWiseEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        """
        對於每個 token，跨 feature (channel) 進行多頭自注意力
        input shape shape: (B, tokens, F, embed_dim)
        output shape: (B, tokens, F, embed_dim)
        """
        super(ChannelWiseEncoder, self).__init__()
        # 使用 batch_first=True 方便操作 (B, seq, E)
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,
                                         dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        將 x (B, tokens, F, embed_dim) 重塑為 (B*tokens, F, embed_dim)，
        然後進行 self-attention 及前饋網路，最後還原形狀。
        """
        B, tokens, F, E = x.shape
        x_reshaped = x.reshape(B * tokens, F, E)  # (B*tokens, F, E)
        # 使用 self-attention：查詢、鍵、值皆為 x_reshaped
        attn_output, _ = self.mha(x_reshaped, x_reshaped, x_reshaped)  # (B*tokens, F, E)
        x_res = self.norm1(x_reshaped + attn_output)
        ff_output = self.ff(x_res)
        x_ff = self.norm2(x_res + ff_output)  # (B*tokens, F, E)
        x_out = x_ff.reshape(B, tokens, F, E)
        return x_out

# 4. Multi-step Decoder (No changes needed)
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
        # 初始上一個預測 patch，形狀 (B, F, patch_length)，全 0
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

# 5. Single step decoder (No changes needed)
class SingleStepDecoder(nn.Module):
    def __init__(self, embed_dim, feature_shape, hidden_dim=128, out_dim=16):
        """
        用於生成單一預測值
        input shape: (B, F, embed_dim)
        output shape: (B, out_dim)
        """
        super(SingleStepDecoder, self).__init__()
        # 使用 MLP 處理整個特徵集
        self.feature_fusion = nn.Sequential(
            nn.Linear(feature_shape * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, out_dim)
        )

    def forward(self, context):
        # (B, F, embed_dim) -> (B, F*embed_dim)
        flat_context = context.reshape(context.size(0), -1)
        out = self.feature_fusion(flat_context)  # (B, out_dim)
        return out

# 6. MultiPatchFormer (Adjusted for input shape (B, C, S, F))
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
        self.temporal_encoder = TemporalEncoder(embed_dim, num_layers=temporal_layers,
                                                num_heads=temporal_heads, dropout=dropout)
        self.channel_encoder = ChannelWiseEncoder(embed_dim, num_heads=channel_heads, dropout=dropout)
        # tokens combined using attention method
        self.token_attention = nn.Sequential(
            nn.Linear(feature_shape * embed_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
        # Multi step decoder:
        # self.decoder = MultiStepDecoder(embed_dim, patch_length=decoder_patch_length, num_steps=decoder_steps, hidden_dim=embed_dim*2)
        # Single value decoder:
        self.decoder = SingleStepDecoder(embed_dim, feature_shape=feature_shape, hidden_dim=decoder_hidden_dim, out_dim=decoder_out_dim)

    def forward(self, x):
        """
        x: (B, C, S, F)
        流程：
          1. 多尺度嵌入：輸出 (B, total_tokens, F, embed_dim)
          2. 時間編碼器：輸出 (B, total_tokens, F, embed_dim)
          3. 通道式編碼器：輸出 (B, total_tokens, F, embed_dim)
          4. 於 tokens 維度做加權平均，獲得 (B, F, embed_dim)
          5. 解碼器：輸出 (B, out_dim) 或 (B, T_pred, F)
        """
        x = self.embedding(x)                # (B, tokens, F, embed_dim)
        x = self.temporal_encoder(x)         # (B, tokens, F, embed_dim)
        x = self.channel_encoder(x)          # (B, tokens, F, embed_dim)
        
        # 添加注意力權重 weighted sum of tokens
        B, tokens, F, E = x.shape
        x_flat = x.reshape(B, tokens, -1)    # (B, tokens, F*E)
        attn_weights = self.token_attention(x_flat)  # (B, tokens, 1)
        weighted_x = (x * attn_weights.unsqueeze(-1))  # (B, tokens, F, E)
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
