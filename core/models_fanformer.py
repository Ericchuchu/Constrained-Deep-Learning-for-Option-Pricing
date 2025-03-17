import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FANLayer(nn.Module):
    """
    FANLayer : The fundamental building block for fourier analysis network, forward fourier transform operation
    """
    def __init__(self, input_dim , output_dim, p_ratio = 0.25, activation = None, use_p_bias = True):
        super(FANLayer, self).__init__()
        
        # Ensure the p_ratio is within a valid range
        assert 0 <= p_ratio <= 0.5, "p_ratio must be between 0 and 0.5"
        
        self.p_ratio = p_ratio
        p_output_dim = int(output_dim * self.p_ratio)
        g_output_dim = output_dim - p_output_dim * 2  # Account for cosine and sine terms
        
        self.input_linear = nn.Linear(input_dim, p_output_dim + g_output_dim, bias=use_p_bias)
        
        self.fused_dims = (p_output_dim, g_output_dim)
        
        # Set the activation function
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation if activation else lambda x: x
            
    def forward(self, src, norm_g=None):
        pg = self.input_linear(src)
        
        # Split into periodic (p) and general (g) components
        p, g = pg.split(self.fused_dims, dim=-1)
        
        # Apply optional normalization to g if provided
        if norm_g is not None:
            g = norm_g(g)
        
        # Concatenate cos(p), sin(p), and activated g along the last dimension
        # This implements the Fourier series representation
        output = torch.cat((torch.cos(p), torch.sin(p), self.activation(g)), dim=-1)
        
        return output

# RoPE positional embedding
def rotary_pos_emb(q, k, cos, sin, position_ids = None):
    """
    Implement RoPE positional embedding
    Args:
        q: shape [batch_size, seq_len, n_heads, head_dim] 
        k: shape [batch_size, seq_len, n_heads, head_dim] 
        cos: 餘弦位置編碼，形狀為 [seq_len, head_dim/2]
        sin: 正弦位置編碼，形狀為 [seq_len, head_dim/2]
        position_ids: 可選的位置索引覆蓋
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 應用了 RoPE 的查詢和鍵
    """
    batch_size, seq_len, n_heads, hidden_dims = q.shape

    # assign position
    if position_ids is None:
        position_ids = torch.arange(seq_len, device = q.device)

    cos = cos[position_ids].unsqueeze(1)  # [seq_len, 1, head_dim/2]
    sin = sin[position_ids].unsqueeze(1)  # [seq_len, 1, head_dim/2]
    
    q_embed_dim_half = q.shape[-1] // 2
    k_embed_dim_half = k.shape[-1] // 2
    q_real, q_imag = q[..., :q_embed_dim_half], q[..., q_embed_dim_half:]
    k_real, k_imag = k[..., :k_embed_dim_half], k[..., k_embed_dim_half:]
    
    # 複數旋轉：(a+bi)(cos+sin*i) = (a*cos-b*sin) + (a*sin+b*cos)i
    q_out_real = q_real * cos - q_imag * sin
    q_out_imag = q_real * sin + q_imag * cos
    k_out_real = k_real * cos - k_imag * sin
    k_out_imag = k_real * sin + k_imag * cos
    
    # 重新組合實部和虛部
    q_out = torch.cat([q_out_real, q_out_imag], dim=-1)
    k_out = torch.cat([k_out_real, k_out_imag], dim=-1)
    
    return q_out, k_out


# ATF attention
class ATFattention(nn.Module):
    def __init__(self, hidden_dim, num_heads, p_ratio = 0.25, qkv_bias = False, 
                 attn_dropout = 0.0, proj_dropout = 0.0, max_position_embeddings = 512, base = 10000):
        super(ATFattention, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # RoPE requirements
        assert self.head_dim % 2 == 0, "Head dimension must be even for RoPE"

        # FANLayers for Q, K, V projections
        # Note: no activation for FANLayer as mentioned in the description
        self.q_fan = FANLayer(hidden_dim, hidden_dim, p_ratio, activation=None, use_p_bias=qkv_bias)
        self.k_fan = FANLayer(hidden_dim, hidden_dim, p_ratio, activation=None, use_p_bias=qkv_bias)
        self.v_fan = FANLayer(hidden_dim, hidden_dim, p_ratio, activation=None, use_p_bias=qkv_bias)
        
        # Linear projections for Q, K, V after FAN transformation
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=qkv_bias)
        
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.proj_dropout = nn.Dropout(proj_dropout)

        # RoPE 相關設置
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # 生成 RoPE 的正弦和餘弦表
        self.register_buffer("cos_cached", self.compute_cos_sin_table()[0])
        self.register_buffer("sin_cached", self.compute_cos_sin_table()[1])

    def compute_cos_sin_table(self):
        half_dim = self.head_dim // 2
        emb = torch.arange(half_dim, dtype = torch.float32)
        emb = self.base ** (-2 * emb/ half_dim)
        
        # 生成位置序列
        positions = torch.arange(self.max_position_embeddings, dtype=torch.float32).reshape(-1, 1)
        
        # 計算旋轉角度
        angles = positions * emb
        
        # 計算正弦和餘弦值
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        
        return cos, sin


    def forward(self, x,mask = None, position_ids = None):
        batch_size, seq_len, hidden_dim = x.shape

        # Apply FAN transformation followed by linear projection
        # This implements XF = FANLayer'(X) followed by QF, KF, VF computation
        q = self.q_proj(self.q_fan(x))  # (batch, seq_len, hidden_dim)
        k = self.k_proj(self.k_fan(x))  # (batch, seq_len, hidden_dim)
        v = self.v_proj(self.v_fan(x))  # (batch, seq_len, hidden_dim)
        
        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Use RoPE
        q, k = rotary_pos_emb(
            q.transpose(1,2),
            k.transpose(1,2),
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
            position_ids
        )

        # Reshape to ccalculate attention score
        q = q.transpose(1, 2)  # (batch, n_heads, seq_len, head_dim)
        k = k.transpose(1, 2)  # (batch, n_heads, seq_len, head_dim)
    
        # Calculate attention scores
        # ATF(X) = softmax(QF·KF^T/√d_h)·VF
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attn_weights, v)
        
        # Reshape back to original dimensions
        context = context.permute(0, 2, 1, 3).reshape(batch_size, seq_len, hidden_dim)
        
        # Final projection
        output = self.proj(context)
        output = self.proj_dropout(output)
        
        return output
    
# SwiGLU activation function
class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        
        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w2 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)
        
    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = F.silu(x1) * x2  # SiLU (Swish) activation with gating
        return self.w3(hidden)

# FANformer layer        
class FANformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, feedforward_dim, p_ratio = 0.25, qkv_bias = False,
                 dropout = 0.0, attn_dropout = 0.0, proj_dropout = 0.0, max_position_embeddings = 512):
        super(FANformerLayer, self).__init__()
        
        # 預層正規化
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # ATF attention layer
        self.atf = ATFattention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            p_ratio=p_ratio,
            qkv_bias=qkv_bias,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            max_position_embeddings=max_position_embeddings
        )
        
        # feed forward network
        self.ffn = SwiGLU(
            in_features=hidden_dim,
            hidden_features=feedforward_dim,
            out_features=hidden_dim,
            bias=True
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None, position_ids=None):
        # layer normalization, ATF atyention, and residual connection
        attn_output = self.atf(self.norm1(x), mask, position_ids)
        x = x + attn_output
        
        # layer normalization, feedforward network, and residual connection
        ffn_output = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_output)
        
        return x

if __name__ == '__main__':
    # Example usage with input shape (batch, channel, sequence, feature) = (64, 3, 10, 5)
    batch_size = 64
    S = 10  # sequence length
    embed_dim = 32
    x = torch.randn(batch_size, S, embed_dim)
    
    model = FANformerLayer(
        hidden_dim=embed_dim,
        num_heads=4,
        feedforward_dim=embed_dim
    )
    
    output = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)


        

