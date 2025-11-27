import torch
import torch.nn.functional as F
from model.gpt import GPT

# ==========================================
# 1. 配置 & 准备 (必须和训练时一致)
# ==========================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_path = 'data/input.txt'
model_path = 'model/nano_gpt_code.pth'

# 重新构建 Tokenizer (为了确保和训练时映射表一致，我们重读一遍数据)
# (注：更工程化的做法是把 stoi/itos 保存成 pkl 文件，但这里直接读文件更稳妥)
with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(f"🔥 设备: {device}")
print(f"🔤 词表重构完成，大小: {vocab_size}")

# ==========================================
# 2. 加载模型
# ==========================================
# ⚠️ 参数必须和 train.py 里一模一样！
n_embd = 384
n_head = 6
n_layer = 4
block_size = 64
dropout = 0.2

model = GPT(vocab_size=vocab_size,
            d_model=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            max_len=block_size,
            dropout=dropout)

# 加载训练好的权重
print(f"⏳ 正在加载模型权重: {model_path} ...")
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()  # 切换到评估模式 (关闭 Dropout)
print("✅ 模型加载成功！准备生成代码...")


# ==========================================
# 3. 定义生成函数
# ==========================================
def generate_code(start_text, max_new_tokens=200):
    # 把文本变成 tensor
    context_idxs = encode(start_text)
    idx = torch.tensor(context_idxs, dtype=torch.long, device=device).unsqueeze(0)  # [1, T]

    print(f"\n{'=' * 20} 生成结果 {'=' * 20}")
    print(f"🟢 起始提示: {start_text}")
    print(f"🤖 AI续写:\n")

    # 逐步生成
    for _ in range(max_new_tokens):
        # 截断 context，确保不超过 block_size
        idx_cond = idx[:, -block_size:]

        # 前向传播
        logits = model(idx_cond)

        # 只取最后一个时间步的预测
        logits = logits[:, -1, :]  # [1, vocab_size]

        # 算概率
        probs = F.softmax(logits, dim=-1)

        # 采样 (Multinomial Sampling) - 增加多样性
        idx_next = torch.multinomial(probs, num_samples=1)  # [1, 1]

        # 拼接到结果里
        idx = torch.cat((idx, idx_next), dim=1)

        # 实时打印出一个字符 (更有感觉)
        char = decode([idx_next.item()])
        print(char, end='', flush=True)

    print(f"\n\n{'=' * 20} 结束 {'=' * 20}")


# ==========================================
# 4. 玩耍时间！
# ==========================================
# Case 1: 让他写个 import
generate_code("import ", max_new_tokens=100)

# Case 2: 让他定义个函数
generate_code("def get_url(url):", max_new_tokens=200)

# Case 3: 让他写个类
generate_code("class Session:", max_new_tokens=200)