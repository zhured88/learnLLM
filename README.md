# Learn LLM — 动手学大语言模型

> 参考李沐老师《动手学深度学习》，以"可运行的代码 > 公式推导"为原则，从零构建大语言模型相关算法。

## 项目哲学

- **代码先行**：每个算法都从零手写实现，不调包
- **逐层递进**：MLP → RNN → Transformer → LLM，每一步都有前序铺垫
- **可解释性**：每个项目附带技术详解文档和可视化分析
- **小白友好**：核心技术文档包含"白话先行"段落，用生活类比解释概念

## 学习路线

| 阶段 | 主题 | 项目 | 状态 |
|------|------|------|------|
| 0 | 深度学习基础 | [纯 NumPy 手写 MLP · MNIST 分类](stage0-mlp-numpy/) | ✅ |
| 1 | CNN + RNN | ResNet-18 + LSTM 文本生成 | 🔜 |
| 2 | NLP 基础 | [Word2Vec + Seq2Seq Attention](stage2-nlp-attention/) | ✅ |
| 3 | Transformer | [从零实现 Transformer + GPT-2 语言模型](stage3-transformer/) | ✅ |
| 4 | LLM 预训练 | ~100M 参数 GPT + LLaMA 架构 | 🔜 |
| 5 | 指令微调 | SFT + LoRA + DPO | 🔜 |
| 6 | 推理优化 | KV Cache + 量化 + 部署 | 🔜 |
| 7 | 前沿架构 | MoE + 多模态 | 🔜 |
| 8 | Agent 应用 | RAG + ReAct Agent | 🔜 |

## 快速开始

```bash
# 阶段 0：纯 NumPy MLP
cd stage0-mlp-numpy
python main.py          # 训练模型（MNIST 上 98%+ 准确率）
python interpret.py     # 可解释性分析（生成预测可视化图片）
```

## 环境要求

- Python 3.10+
- NumPy, Matplotlib
- (后续阶段) PyTorch 2.x, HuggingFace Transformers

## 参考资源

- 《动手学深度学习》—— 李沐 (d2l.ai)
- "Attention Is All You Need" — Vaswani et al., 2017
- Andrej Karpathy — nanoGPT
- HuggingFace NLP Course

## License

MIT
