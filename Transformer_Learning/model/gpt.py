import torch
import torch.nn as nn
from torch.nn.functional import dropout
import math
from model.attention import *


# 1. 工具函数：因果掩码 (Causal Mask)
#    这是 Decoder-only 架构的核心，确保模型在写代码时不能偷看后面
def create_causal_mask(seq_len):
    """
        生成下三角掩码。
        形状: [seq_len, seq_len]
        1 0 0
        1 1 0
        1 1 1
        """
    mask = torch.ones(seq_len, seq_len)
    mask = torch.tril(mask)

    return mask.bool()

# 2. 位置编码 (Positional Embedding)
#    代码对顺序极度敏感，def 必须在 return 前面
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # 1. 造一个“位置字典”
        # nn.Embedding 本质上就是一个大矩阵，形状是 [max_len, d_model]
        # 比如 [5000, 512]
        # 第 0 行存的是“位置0”的专属特征，第 1 行存的是“位置1”的...
        # 关键点：这些特征是“可学习的”(Learnable) 模型在训练过程中，会自己学会“位置0”长什么样最好
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        # x 是输入的 token 序列，假设形状是 [Batch=2, Seq_Len=10]
        batch_size,seq_len = x.size()

        # 2. 生成位置号牌
        # 用 torch.arange 生成一个序列: [0, 1, 2, ..., 9]
        # .unsqueeze(0) 是为了把形状变成 [1, 10]，方便后面和 Batch 维度自动对齐
        position = torch.arange(0, seq_len, device=x.device).unsqueeze(0)

        # 3. 查表并返回
        # 拿着 [0, 1, 2...] 去 self.pe 这个大表里查，返回对应的向量
        return self.pe(position)


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
        atten_out = self.attention(x_norm,mask)
        x = x + self.dropout(atten_out)

        # === Part 2: FeedForward ===
        x_norm = self.ln2(x)
        ff_out = self.ff(x_norm)
        x = x + self.dropout(ff_out)

        return x

class GPT(nn.Module):

    #我们需要定义整个网络的骨架：Embedding -> Blocks -> Final Norm -> Output Head

    def __init__(self, vocab_size, d_model, n_layer, n_head, max_len=1024):
        super().__init__()
        # === 1. 零件准备：入口 ===
        # 词嵌入：把 "101" 这种数字变成一个向量 [0.1, -0.5, ...] (代表"语义")
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # 位置嵌入：把 "第0个位置" 变成一个向量 (代表"顺序")
        # 你的 PositionalEmbedding 类就在这里被实例化
        self.position_embedding = PositionalEmbedding(d_model, max_len)

        # === 2. 核心引擎：Block 堆叠 ===
        # 深度学习之所以叫“深度”，就是因为这里层数多
        # 我们用 ModuleList 像装子弹一样，装入 n_layer 个 Block
        # 每个 Block 都能提取更高级的特征（语法 -> 语义 -> 逻辑）
        self.blocks = nn.ModuleList([
            Block(d_model, n_head) for _ in range(n_layer)
        ])

        # === 3. 零件准备：出口 ===
        # Final LayerNorm: 经过几十层计算，数据分布可能乱了，最后整理一下
        self.ln_f = nn.LayerNorm(d_model)

        # LM Head (Language Model Head):
        # 把隐藏层维度 (d_model) 映射回 词表维度 (vocab_size)
        # 这样才能知道下一个词是 "def" 还是 "import" 的概率最大
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # === 🌟 专家点拨：Weight Tying (权重共享) ===
        # 这是一个算法技巧。
        # 直觉：Token "def" 进入模型时用的向量，和模型想输出 "def" 时用的向量，
        # 在语义空间里应该是相似的。所以我们强制让它们共享参数。
        # 好处：省了大量显存，而且通常效果更好。
        self.token_embedding.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        # idx: [Batch, Seq_Len] (比如 2句话，每句10个词)
        B , T = idx.size()

        # === Step 1: 准备“眼罩” (Mask) ===
        # 此时生成一个 T x T 的下三角矩阵。
        # 为什么在这里生成？因为序列长度 T 是动态的，可能这次是10，下次是20
        mask = create_causal_mask(T).to(idx.device)

        # === Step 2: 融合信息 ===
        # 词的信息 + 位置的信息 = 完整的输入表示
        token_emb =  self.token_embedding(idx)
        position_emb = self.position_embedding(idx)
        x = token_emb + position_emb

        # === Step 3: 层层提炼 ===
        # 数据流经每一个 Block
        for block in self.blocks:
            # 这里的 mask 传进去，就是为了在 Attention 算分时
            # 把右上角（未来的词）变成 -inf，让 Softmax 概率为 0
            x = block(x, mask)

        # === Step 4: 最终输出 ===
        x = self.ln_f(x)  # 归一化
        logits = self.lm_head(x)  # [B, T, vocab_size] -> 每个位置预测下一个词的“分数”

        # === Step 5: 算分 (仅在训练时) ===
        loss = None
        if targets is not None:
            # targets 也是 [B, T]

            # PyTorch 的 CrossEntropyLoss 有个怪癖：
            # 它希望 Input 是 [样本数, 类别数]，即 2D 矩阵。
            # 但我们的 logits 是 3D 的 [Batch, Time, Vocab]。

            # 所以我们需要把 Batch 和 Time 捏在一起，变成一个“超级长”的序列。
            # view(-1, ...) 的意思是：把前两个维度合并，剩下的自动计算。

            # logits 变身: [B*T, vocab_size]
            B_T_logits = logits.view(-1, logits.size(-1))

            # targets 变身: [B*T]
            B_T_targets = targets.view(-1)

            # 这样就是标准的分类问题了：
            # 对这 B*T 个位置，算出预测值和真实值的差距。
            loss = F.cross_entropy(B_T_logits, B_T_targets)

        return logits, loss


