import torch
import torch.nn as nn
from model.gpt import Block, FeedForward


def test_gpt_components():
    print("--- 🚀 开始 GPT 组件测试 (Week 2 抢跑验收) ---")

    # 1. 定义测试参数 (模拟 GPT-2 Small)
    batch_size = 2
    seq_len = 32
    d_model = 768
    n_head = 12

    print(f"⚙️  测试配置: Batch={batch_size}, Seq={seq_len}, Dim={d_model}, Head={n_head}")

    # 造一个假数据 (模拟输入 Tensor)
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"📦 输入数据形状: {x.shape}")

    print("\n--- [测试 1] FeedForward 模块 ---")
    try:
        # 实例化 FFN
        ff = FeedForward(d_model)
        # 前向传播
        out_ff = ff(x)

        if out_ff.shape == x.shape:
            print("✅ FeedForward 测试通过！输出维度正确。")
        else:
            print(f"❌ FeedForward 维度错误: {out_ff.shape}")
            return
    except Exception as e:
        print(f"❌ FeedForward 运行崩溃: {e}")
        return

    print("\n--- [测试 2] Block 模块 (核心) ---")
    try:
        # 实例化 Block
        block = Block(d_model, n_head)

        # 检查内部组件是否存在 (防止变量名写错)
        print(f"   - 检查子模块: Attn={hasattr(block, 'attn')}, FF={hasattr(block, 'ff')}, LN={hasattr(block, 'ln1')}")

        # 前向传播 (暂时不传 mask，下周一再搞 mask)
        out_block = block(x)

        if out_block.shape == x.shape:
            print(f"✅ Block 测试通过！输出维度: {out_block.shape}")
            print("🎉 恭喜！GPT 的躯干已经搭建完毕，且逻辑自洽！")
        else:
            print(f"❌ Block 维度错误: {out_block.shape}")

    except Exception as e:
        print(f"❌ Block 运行崩溃: {e}")
        print("💡 提示: 如果报错 'tuple object...'，请检查 model/attention.py 是否只返回了 output")


if __name__ == "__main__":
    test_gpt_components()