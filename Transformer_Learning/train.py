import torch
import torch.nn as nn
from torch.nn import functional as F
import os

from model.gpt import GPT

# ==========================================
# 1. 配置参数 (Hyperparameters)
# ==========================================
batch_size = 32  # 一次看多少个片段
block_size = 64  # 上下文长度 (一次看多少个字符)
max_iters = 1000  # 训练多少步
learning_rate = 3e-4  # 学习率
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_interval = 200  # 每多少步评估一次
n_embd = 384  # 嵌入维度 (d_model)
n_head = 6  # 注意力头数
n_layer = 4  # Block 层数
dropout = 0.2

print(f"🔥 正在使用设备: {device}")

# ==========================================
# 2. 准备数据 (Data Pipeline)
# ==========================================
try:
    with open('data/input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"📚 成功加载数据! 长度: {len(text)} 字符")
except FileNotFoundError:
    print("❌ 错误: 找不到 data/input.txt。请确保你昨天完成了数据准备任务！")
    exit()

# --- 构建简单的字符级 Tokenizer ---
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"🔤 词表大小: {vocab_size}")
print(f"🔤 词表内容 (部分): {''.join(chars[:20])}...")

# 建立映射表
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # 文本 -> 数字
decode = lambda l: ''.join([itos[i] for i in l])  # 数字 -> 文本

# 划分训练集/验证集
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # 90% 训练
train_data = data[:n]
val_data = data[n:]


# --- 获取 Batch 的函数 ---
def get_batch(split):
    data_source = train_data if split == 'train' else val_data
    # 随机选 batch_size 个起始点
    ix = torch.randint(len(data_source) - block_size, (batch_size,))
    # 提取输入 x 和 目标 y (y 就是 x 向后移一位)
    x = torch.stack([data_source[i:i + block_size] for i in ix])
    y = torch.stack([data_source[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# ==========================================
# 3. 初始化模型
# ==========================================
model = GPT(vocab_size=vocab_size,
            d_model=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            max_len=block_size,  # 注意：这里要和 block_size 对齐
            dropout=dropout)
model = model.to(device)
print(f"🤖 模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

# 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ==========================================
# 4. 训练循环 (The Loop)
# ==========================================
print("🚀 开始训练...")

for iter in range(max_iters):

    # --- 评估阶段 (不更新参数) ---
    if iter % eval_interval == 0 or iter == max_iters - 1:
        model.eval()
        with torch.no_grad():
            # 简单估算一下当前的 Loss
            x_val, y_val = get_batch('val')
            logits = model(x_val)
            # 计算 CrossEntropyLoss
            # logits: [B, T, vocab_size] -> [B*T, vocab_size]
            # targets: [B, T] -> [B*T]
            loss = F.cross_entropy(logits.view(-1, vocab_size), y_val.view(-1))
            print(f"Step {iter}: Val Loss = {loss.item():.4f}")
        model.train()  # 切回训练模式

    # --- 训练阶段 ---
    # 1. 拿数据
    xb, yb = get_batch('train')

    # 2. 前向传播
    logits = model(xb)

    # 3. 算 Loss
    loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))

    # 4. 反向传播 (三板斧)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# ==========================================
# 5. 保存模型
# ==========================================
torch.save(model.state_dict(), 'model/nano_gpt_code.pth')
print("💾 模型已保存到 model/nano_gpt_code.pth")
