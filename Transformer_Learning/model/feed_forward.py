import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, d_model,d_ff,dropout=0.1):
        """
                前馈网络 (Feed-Forward Network) 初始化。
                这是 Transformer Encoder 和 Decoder 中每个子层之后的标准组件。

                Args:
                    d_model (int): 模型的隐藏层维度 (embedding dimension)，也是输入输出的维度。
                    d_ff (int): 前馈网络的内部隐藏层维度。通常是 d_model 的 4 倍。
                                例如：d_model=512, d_ff=2048。
                    dropout (float): Dropout 比率。
        """
        super(FeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.linear1 = nn.Linear(d_model, d_ff)

        self.linear2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        """
                前向传播

                Args:
                    x (torch.Tensor): 输入张量。
                        形状: [batch_size, seq_len, d_model]
                Returns:
                    torch.Tensor: 输出张量。形状与输入 x 相同: [batch_size, seq_len, d_model]
        """

        # ----------------------------------------------------------
        # TODO 3: 通过第一个线性层，然后应用 ReLU 激活函数
        # 形状变化: [B, S, D] -> [B, S, D_ff]
        # ----------------------------------------------------------
        x = F.relu(self.linear1(x))

        # ----------------------------------------------------------
        # TODO 4: 应用 Dropout
        # ----------------------------------------------------------
        x = self.dropout(x)

        # ----------------------------------------------------------
        # TODO 5: 通过第二个线性层
        # 形状变化: [B, S, D_ff] -> [B, S, D]
        # ----------------------------------------------------------
        x = self.linear2(x)

        return x




# ==============================================================================
# 测试代码
# ==============================================================================
if __name__ == "__main__":
    print("--- 正在测试 FeedForward 模块 ---")

    # 设定参数
    d_model = 512       # 模型的隐藏层维度
    d_ff = 2048         # 前馈网络的内部维度 (通常是 d_model 的 4 倍)
    seq_len = 20        # 序列长度 (例如，一个句子有20个词)
    batch_size = 4      # 批次大小 (例如，一次处理4个句子)
    dropout = 0.1       # Dropout 比率

    # 1. 实例化 FeedForward 模块
    print(f"初始化FeedForward （d_model={d_model}, d_ff={d_ff}）")
    model = FeedForward(d_model, d_ff, dropout)
    print("FeedForward模块初始化成功")

    # 2. 创建一个模拟输入张量 x
    # 形状: [batch_size, seq_len, d_model]
    # 随机生成一些数据，模拟词嵌入或Attention层的输出
    print(f"创建模拟输入张量 x，形状: [{batch_size}, {seq_len}, {d_model}]")
    x =torch.randn(batch_size, seq_len, d_model)
    print("输入张量 x 创建成功。")
    print(f"x 的初始形状: {x.shape}")

    # 3. 调用 FeedForward 模块的前向传播
    print("调用 FeedForward 模块的前向传播...")
    out = model(x)
    print("前向传播完成！")

    #4.检查输出形状
    print(f"输出 output 的最终形状: {out.shape}")

    assert out.shape == x.shape ,f"输出形状不匹配! 预期: {x.shape}, 实际: {out.shape}"

    print("\n--- FeedForward 模块测试成功！ ---")
    print("输入和输出形状匹配，模块功能正常。")






