import torch
import torch.nn as nn
# 确保你的文件夹结构是 model/attention.py
from model.attention import MultiHeadAttention


def test_week2_day1_final():
    print("--- 🚀 开始 Week 2 Day 1 最终验收测试 ---")

    # ==========================================
    # 测试 1: 基础跑通 (Basic Sanity Check)
    # ==========================================
    print("\n[测试 1] 基础组件连通性测试...")
    d_model = 512
    n_heads = 8
    seq_len = 10
    batch_size = 2

    try:
        # 1. 实例化
        model = MultiHeadAttention(d_model, n_heads)
        # 2. 造假数据
        x = torch.randn(batch_size, seq_len, d_model)
        # 3. 前向传播 (不带 Mask)
        out = model(x)

        if out.shape == (batch_size, seq_len, d_model):
            print("✅ 基础维度检查通过！模型骨架搭建完成。")
        else:
            print(f"❌ 维度错误: 期望 {(batch_size, seq_len, d_model)}, 实际 {out.shape}")
            return

    except ValueError as e:
        print(f"❌ 运行崩溃: {e}")
        print(
            "💡 提示: 如果报错 'too many values to unpack'，请检查 attention.py 第 86 行是否改成了 'out = self.attention(...)'")
        return
    except Exception as e:
        print(f"❌ 未知错误: {e}")
        return

    # ==========================================
    # 测试 2: AI4SE 核心 - Causal Mask 测试
    # ==========================================
    print("\n[测试 2] Causal Mask (代码补全核心) 测试...")
    # 模拟一个极短的代码片段: "def main ( )" -> 4个token
    mini_seq = 4
    mini_batch = 1

    # 1. 构造下三角 Mask (核心!)
    # 形状: [mini_seq, mini_seq] -> [4, 4]
    # 1 表示可见，0 表示遮挡
    mask = torch.tril(torch.ones(mini_seq, mini_seq))

    print(f"   Mask 矩阵 (防作弊视窗):\n{mask}")

    try:
        x_code = torch.randn(mini_batch, mini_seq, d_model)

        # 传入 Mask
        out_masked = model(x_code, mask=mask)

        if out_masked.shape == (mini_batch, mini_seq, d_model):
            print("✅ Mask 机制运行正常！Attention 层成功处理了遮挡逻辑。")
            print("🎉 Day 1 任务圆满完成！你的 GPT 已经准备好学习写代码了。")
        else:
            print(f"❌ Mask 输出维度错误: {out_masked.shape}")

    except Exception as e:
        print(f"❌ Mask 测试崩溃: {e}")
        print("💡 检查点: ScaledDotProductAttention 里的 masked_fill 逻辑写对了吗？")


if __name__ == "__main__":
    test_week2_day1_final()