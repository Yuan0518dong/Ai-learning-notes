import torch
import torch.nn as nn
from sympy.polys.densearith import dmp_ff_div
from torch.nn.functional import dropout

from model.attention import *

class FeedForward(nn.Module):
    """
        GPT 的'肌肉'：负责记忆和非线性变换
        结构：Linear -> GELU -> Linear
        """
    def __init__(self, d_model, expansion_factor = 4, dropout = 0.1):
        super().__init__()
        # expansion_factor 默认是 4，即中间层维度是 4 倍
        d_ff = d_model * expansion_factor
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),# GPT 关键细节：使用 GELU 而非 ReLU
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """
        一个标准的 GPT Decoder Block
        结构：Input -> LN -> Attn -> Add -> LN -> FFN -> Add
        特性：Pre-Norm (层归一化在子层之前)
        """
    def __init__(self, d_model, n_head, dropout = 0.1):
        super().__init__()
        # 1. 核心组件
        self.attention = MultiHeadAttention(d_model, n_head)
        self.ff = FeedForward(d_model)

        # 2. 归一化层 (LayerNorm)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # 3. Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x,mask = None):
        # === Part 1: Attention ===
        # Pre-Norm 结构: 先 LN，再 Attention
        # 残差连接 (Residual): x = x + sublayer(LN(x))
        x_norm= self.ln1(x)
        # 注意：这里我们传入 x, x, x 是因为它是 Self-Attention
        # Decoder 训练时必须要有 mask (下三角掩码)，不过今天先传 None 跑通维度即可
        atten_out = self.attention(x_norm,x_norm,x_norm,mask)
        x = x + self.dropout(atten_out)

        # === Part 2: FeedForward ===
        x_norm = self.ln2(x)
        ff_out = self.ff(x_norm)
        x = x + self.dropout(ff_out)

        return x
