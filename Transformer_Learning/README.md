# NanoGPT for Code: A Minimalist Code Completion Model

![Status](https://img.shields.io/badge/Status-Prototype-green)
![Topic](https://img.shields.io/badge/Topic-LLM%20Training-blue)

这是一个基于 Transformer (Decoder-only) 架构的迷你生成式模型，专门从零开始训练用于 **Python 代码补全** 任务。
该项目是 **Week 2 AI4SE (AI for Software Engineering)** 的核心工程实践。

## 🏗️ 核心架构 (Architecture)
本项目**不依赖**任何高级封装库 (如 HuggingFace Trainer)，纯手工实现了 GPT 的核心组件：
* **Causal Masking:** 实现了下三角掩码，确保模型严格遵守因果推理（不偷看未来）。
* **Transformer Block:** 实现了 Pre-Norm 结构的 Attention + FeedForward 堆叠。
* **Positional Embedding:** 使用了可学习的位置编码。

## 📂 项目结构
```text
Transformer_Learning/
├── data/               # 训练数据 (Python 源码)
├── model/
│   ├── attention.py    # Multi-Head Attention + Mask 手写实现
│   ├── gpt.py          # GPT 模型主架构 (Block, FeedForward)
├── train.py            # 训练脚本 (Training Loop)
├── generate.py         # 推理脚本 (Token Sampling)
└── test_playground.py  # 单元测试 (Unit Tests)
```