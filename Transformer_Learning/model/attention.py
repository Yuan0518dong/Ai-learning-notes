import torch
import torch.nn as nn
import  math
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    [打工仔]
    只负责纯数学运算。
    输入: 已经是分好头的 q, k, v
    输出: 加权后的 values, attention weights
    """

    def __init__(self, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # q, k, v 维度: [batch, n_heads, seq_len, d_k]

        d_k = q.size(-1)

        # --- 数学公式步骤 ---

        # 1. Q * K.T / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)

        # 2. Mask (如果有)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 3. Softmax
        attn_weights = F.softmax(scores, dim=-1)

        # 4. Dropout
        attn_weights = self.dropout(attn_weights)

        # 5. Weight * V
        output = torch.matmul(attn_weights, v)

        return output


class MultiHeadAttention(nn.Module):
    """
    [包工头]
    负责：投影(Linear) -> 分头(Split) -> 指挥打工仔计算 -> 拼接(Concat) -> 输出
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"

        # 定义4个线性层: Q, K, V 的投影，以及最后的输出投影
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

        # 雇佣打工仔
        self.attention = ScaledDotProductAttention(dropout=dropout)

    def forward(self, x, mask=None):
        batch_size = x.size(0)

        # 1. [投影] 把输入变个身
        query = self.w_q(x)
        key   = self.w_k(x)
        value = self.w_v(x)

        # 2. [分头] 最难的一步：Split & Transpose
        # view: 把 d_model 拆成 n_heads * d_k
        # transpose: 把 n_heads 放到前面，方便并行计算
        # 形状变化: [batch, seq, d_model] -> [batch, n_heads, seq, d_k]
        query = query.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        key   = key.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        value = value.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 3. [计算] 交给打工仔处理
        # 这里的参数必须是3个分开的 q, k, v
        out = self.attention(query, key, value, mask=mask)

        # 4. [拼接] Concat
        # transpose: 把 n_heads 换回来 -> [batch, seq, n_heads, d_k]
        # contiguous: 内存整理 (必须做，否则 view 报错)
        # view: 拼回原来的形状 -> [batch, seq, d_model]
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 5. [收尾] 最后的线性变换
        out = self.fc(out)

        return out

