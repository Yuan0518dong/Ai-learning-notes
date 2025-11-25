# ... (上面是你写好的 GPT 类) ...
import torch

from model.gpt import GPT

if __name__ == "__main__":
    print("\n-------------------------------------------")
    print("🧪 开始 GPT 模型骨架测试 (Week 2 Day 2)")
    print("-------------------------------------------")

    try:
        # 1. 模拟超参数
        vocab_size = 100  # 假定词表只有100个词
        d_model = 64  # 嵌入维度 64
        n_layer = 2  # 2 层 Block
        n_head = 2  # 2 个头
        max_len = 20  # 最长序列 20

        # 2. 实例化模型
        model = GPT(vocab_size, d_model, n_layer, n_head, max_len)
        print("✅ [1/4] 模型实例化成功！")

        # 3. 创建模拟数据
        batch_size = 4
        seq_len = 10
        # 模拟输入 [4, 10]
        dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
        # 模拟目标 (Labels) [4, 10]
        dummy_target = torch.randint(0, vocab_size, (batch_size, seq_len))

        print(f"ℹ️  输入维度: {dummy_input.shape}")

        # 4. 前向传播 (Forward Pass)
        logits, loss = model(dummy_input, dummy_target)

        # 5. 验证输出维度
        expected_shape = (batch_size, seq_len, vocab_size)
        if logits.shape == expected_shape:
            print(f"✅ [2/4] 输出维度检查通过: {logits.shape}")
        else:
            print(f"❌ [2/4] 输出维度错误! 期望 {expected_shape}, 实际 {logits.shape}")
            exit()

        # 6. 验证 Loss
        if loss is not None and not torch.isnan(loss):
            print(f"✅ [3/4] Loss 计算成功: {loss.item():.4f}")
        else:
            print("❌ [3/4] Loss 计算失败 (是 None 或者是 NaN)")
            exit()

        # 7. 验证 Mask 是否生效 (简单验证)
        # 如果代码没报错，说明 create_causal_mask 形状匹配，且能传进 Attention
        print("✅ [4/4] Causal Mask 传递无报错")

        print("-------------------------------------------")
        print("🎉 恭喜！Week 2 Day 2 任务圆满完成！")
        print("   GPT 骨架已立，明天可以喂 Python 代码数据了！")
        print("-------------------------------------------")

    except Exception as e:
        print("\n❌ 测试过程中发生崩溃！")
        print(f"错误信息: {e}")
        import traceback

        traceback.print_exc()