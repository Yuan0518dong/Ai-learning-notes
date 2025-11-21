import torch
import torch.nn as nn
import  math
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.d_model = d_model # 模型的总维度，例如 512
        self.n_heads = n_heads # 注意力头的数量，例如 8
        # 计算每个头的维度 d_head
        self.d_head = d_model // n_heads

        # 【重要检查】确保能整除，否则无法平均分配
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        # --- Q, K, V 线性投影层 ---
        # 解释：这三层是独立的，各自有自己的权重和偏置
        # 它们的作用是把输入的 d_model 维向量，转换成 Query (Q), Key (K), Value (V) 的 d_model 维表示
        # 即使输入输出维度相同，但因为权重不同，它们会将相同的原始输入映射到三个不同的语义空间
        # 使得 Q, K, V 能够专注于不同的信息提取任务。
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # --- 最终输出线性层 ---
        # 解释：多头注意力计算完成后，会得到一个拼接起来的 d_model 维向量。
        # 这个线性层进一步融合和转换这个向量，作为自注意力机制的最终输出
        self.w_o = nn.Linear(d_model, d_model)

        # Dropout 层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, mask=None):
        # x 的输入形状: [batch_size, seq_len, d_model]
        # 举例：[32, 10, 512] -> 32个句子，每个句子10个词，每个词用512维向量表示
        batch_size , seq_len, _ = x.shape # 提取批次大小和序列长度

        # ==============================================================================
        # 步骤 1: 线性投影 (Linear Projections)
        # ==============================================================================
        # 解释：把输入 x 分别通过我们之前定义的 w_q, w_k, w_v 三个线性层。
        # 这就是将原始输入转换成 Query (Q), Key (K), Value (V) 的过程。
        # 形状变化：
        #   x: [batch_size, seq_len, d_model]
        #   Q, K, V: 经过线性层后，形状仍然是 [batch_size, seq_len, d_model]
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        # ==============================================================================
        # 步骤 2: 拆分多头 (Split Heads) - 【维度变换核心】
        # ==============================================================================
        # 解释：这是将 d_model 维度的表示“切分”给 n_heads 个头，并调整维度以方便并行计算。
        # 1. `.view(batch_size, seq_len, self.n_heads, self.d_head)`
        #    - 把 d_model (例如 512) 维度“逻辑上”重塑为 (n_heads, d_head) (例如 8, 64)。
        #    - 形状从 [B, S, D] 变为 [B, S, H, Dh]
        # 2. `.transpose(1, 2)`
        #    - 交换第1维 (seq_len) 和第2维 (n_heads)。
        #    - 目的：让 `n_heads` 这个维度在 `seq_len` 前面，这样后续的矩阵乘法可以针对每个头独立并行计算。
        #    - 形状从 [B, S, H, Dh] 变为 [B, H, S, Dh]
        # 举例：[32, 10, 512] -> [32, 10, 8, 64] -> [32, 8, 10, 64]
        # 想象一下：现在我们有 8 个独立的“视角”或“小组”，每个小组处理着自己的 Q, K, V。
        Q=Q.view(batch_size,seq_len,self.n_heads,self.d_head).transpose(1,2)
        K=K.view(batch_size,seq_len,self.n_heads,self.d_head).transpose(1,2)
        V=V.view(batch_size,seq_len,self.n_heads,self.d_head).transpose(1,2)

        # ==============================================================================
        # 步骤 3: 缩放点积注意力 - 计算分数 (Scaled Dot-Product Attention Scores)
        # ==============================================================================
        # 公式核心: Attention(Q, K, V) = softmax( (Q * K^T) / sqrt(d_k) ) * V
        # 这一步计算 Q * K^T 部分
        # 1. `K.transpose(-1, -2)`
        #    - 对 K 的最后两个维度 (seq_len 和 d_head) 进行转置。
        #    - K 原形状: [B, H, S, Dh]
        #    - K.transpose(-1, -2) 后形状: [B, H, Dh, S]
        # 2. `torch.matmul(Q, K.transpose(-1, -2))`

        #    - 进行矩阵乘法这个乘法是在批次 (B) 和头 (H) 维度上独立进行的
        #    - [..., S, Dh] * [..., Dh, S] -> [..., S, S]

        #    - 所以 `scores` 的形状是: [batch_size, n_heads, seq_len, seq_len]
        #    - 这个 [S, S] 矩阵就是每个词对序列中所有其他词的“相似度”或“关注度”分数
        # 3. `/ math.sqrt(self.d_head)`
        #    - 缩放因子 为了防止 Q 和 K 的点积结果过大，导致 softmax 后梯度过小
        #      除以 `sqrt(d_head)` 进行缩放 这是 Transformer 的一个重要设计
        scores = torch.matmul(Q, K.transpose(-1,-2)) / math.sqrt(self.d_head)

        # ==============================================================================
        # 步骤 4: 应用掩码 (Apply Mask)
        # ==============================================================================
        # 解释：处理特殊情况，例如：
        #   - Padding Mask: 如果句子有填充（为了凑齐长度），填充的词不应该被关注。
        #   - Look-ahead Mask (Decoder): 在解码器中，一个词不应该看到它后面的词。
        #   mask 的形状通常是 [batch_size, 1, seq_len, seq_len] 或 [batch_size, 1, 1, seq_len]。
        #   当 mask 值为 0 (表示需要遮蔽) 的位置，`scores` 会被填充为 -1e9 (一个非常小的负数)。
        #   这样在下一步的 softmax 之后，这些位置对应的注意力权重就会趋近于 0。
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # ==============================================================================
        # 步骤 5: Softmax 和 Dropout
        # ==============================================================================
        # 1. `F.softmax(scores, dim=-1)`
        #    - 在 `scores` 的最后一个维度 (seq_len) 上进行 softmax 操作
        #    - 将分数转换为概率分布，确保每个词对其他词的关注度总和为 1
        #    - `attention_weights` 形状: [batch_size, n_heads, seq_len, seq_len]
        # 2. `self.dropout(attention_weights)`
        #    - 应用 dropout，随机丢弃一部分注意力权重，增强模型的泛化能力
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # ==============================================================================
        # 步骤 6: 加权求和 (Weighted Sum with V)
        # ==============================================================================
        # 解释：用计算出的注意力权重去加权 Value (V) 的信息。
        # 1. `attention_weights`: [B, H, S, S] (表示每个词对其他词的关注概率)
        # 2. `V`: [B, H, S, Dh] (每个词的实际内容信息)
        # 3. `torch.matmul(attention_weights, V)`
        #    - 矩阵乘法结果 `context` 形状: [B, H, S, Dh]
        #    - 含义：对于序列中的每个词，它现在是一个融合了序列中所有相关词 (根据权重) 信息的新表示。
        context = torch.matmul(attention_weights, V)

        # ==============================================================================
        # 步骤 7: 拼接多头 (Concatenate Heads) - 【维度变换逆操作】
        # ==============================================================================
        # 解释：这是步骤 2 “拆分多头”的逆操作，将各个头独立计算的结果重新拼接起来。
        # 1. `context.transpose(1, 2)`
        #    - 交换第1维 (n_heads) 和第2维 (seq_len) 回来。
        #    - 形状从 [B, H, S, Dh] 变为 [B, S, H, Dh]
        # 2. `.contiguous()`
        #    - 【非常重要】`transpose` 操作会改变张量在内存中的存储顺序。
        #    - `view` 操作要求张量的内存是连续的。所以，在 `view` 之前，通常需要调用 `contiguous()` 来确保内存连续。
        # 3. `.view(batch_size, seq_len, self.d_model)`
        #    - 将 (n_heads, d_head) 这两维重新“压扁”成 d_model (例如 512)。
        #    - 形状从 [B, S, H, Dh] 变为 [B, S, D]
        # 举例：[32, 8, 10, 64] -> [32, 10, 8, 64] -> [32, 10, 512]
        # 想象一下：所有小组的报告都汇总起来，形成了一个完整的、更丰富的词表示。
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # ==============================================================================
        # 步骤 8: 最终线性投影 (Final Linear Projection)
        # ==============================================================================
        # 解释：将拼接后的多头输出再进行一次线性变换。
        # 这个线性层可以进一步融合来自不同头的信息，并将其投影到模型所需的最终输出空间。
        # 输入形状: [batch_size, seq_len, d_model]
        # 输出形状: [batch_size, seq_len, d_model]
        output = self.w_o(context)

        return output # 返回自注意力机制处理后的词表示


# ==============================================================================
# 测试代码
# ==============================================================================
if __name__ == "__main__":
    print("--- 正在测试 SelfAttention 模块 ---")

    # 设定参数
    d_model = 512  # 模型的隐藏层维度
    n_heads = 8  # 注意力头的数量
    seq_len = 20  # 序列长度 (例如，一个句子有20个词)
    batch_size = 4  # 批次大小 (例如，一次处理4个句子)
    dropout = 0.1  # Dropout 比率

    # 1. 实例化 SelfAttention 模块
    print(f"初始化 SelfAttention (d_model={d_model}, n_heads={n_heads})...")
    attention_module = SelfAttention(d_model, n_heads, dropout)
    print("SelfAttention 模块初始化成功！")

    # 2. 创建一个模拟输入张量 x
    # 形状: [batch_size, seq_len, d_model]
    # 随机生成一些数据，模拟词嵌入
    print(f"创建模拟输入张量 x，形状: [{batch_size}, {seq_len}, {d_model}]...")
    x = torch.randn(batch_size, seq_len, d_model)  # 随机生成标准正态分布的张量
    print("输入张量 x 创建成功。")
    print(f"x 的初始形状: {x.shape}")

    # 3. (可选) 创建一个模拟掩码 mask
    # 假设我们有一个句子，实际长度是 15，后面 5 个是 padding
    # mask 形状通常是 [batch_size, 1, 1, seq_len] 或 [batch_size, 1, seq_len, seq_len]
    # 这里我们模拟一个简单的 padding mask:
    # 假设所有序列实际长度为 15，后面 5 个位置是 padding
    mask = torch.ones(batch_size, 1, seq_len, seq_len)
    for i in range(batch_size):
        # 假设每个句子的有效词长为 15，后面的为填充
        # 对于自注意力，mask通常是下三角矩阵或者根据padding来决定
        # 这里为了演示，我们只遮蔽padding部分
        mask[i, :, :, 15:] = 0  # 遮蔽掉序列中第15个词及之后的所有词
        # 为了演示自注意力，一个更典型的mask是下三角，防止看到未来信息
        # casual_mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
        # mask = mask * casual_mask[None, None, :, :] # 结合padding和causal mask

    print(f"创建模拟 mask 张量，形状: {mask.shape} (这里假设遮蔽了部分padding)...")

    # 4. 调用 SelfAttention 模块的前向传播
    print("调用 SelfAttention 模块的前向传播...")
    output = attention_module(x, mask=mask)
    print("前向传播完成！")

    # 5. 检查输出
    print(f"输出 output 的最终形状: {output.shape}")

    # 期望输出形状应该与输入 x 相同
    assert output.shape == x.shape, \
        f"输出形状不匹配! 预期: {x.shape}, 实际: {output.shape}"

    print("\n--- SelfAttention 模块测试成功！ ---")
    print("输入和输出形状匹配，模块功能正常。")











