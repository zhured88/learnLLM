# LLM 算法学习大纲（项目驱动 · 动手学深度学习风格）

> 参考李沐老师《动手学深度学习》，以“可运行的代码 > 公式推导”为原则，每个阶段都有一个可交付的 mini 项目。

---

## 阶段 0：环境搭建与基础热身（第 1 周）

### 目标
- 搭建 PyTorch + CUDA 开发环境
- 用 numpy 手写核心算子，建立对 Tensor 操作的肌肉记忆

### 项目
- [ ] **Mini 项目 0**：纯 numpy 实现一个 2 层 MLP，在 MNIST 上训练到 95%+ 准确率
  - 手写 Linear、ReLU、Softmax、CrossEntropyLoss
  - 手写 SGD 优化器 + 反向传播

### 核心算法
- 前向传播、反向传播（计算图）
- 梯度下降及其变体（SGD、Momentum、Adam）
- 链式法则的工程实现

---

## 阶段 1：深度学习基础（第 2-3 周）

### 目标
- 掌握 MLP、CNN、RNN 三大基础架构
- 理解过拟合/欠拟合、正则化、归一化

### 项目
- [ ] **Mini 项目 1**：用 PyTorch 复现 ResNet-18，在 CIFAR-10 上训练
  - 实现残差连接、BatchNorm、Data Augmentation
  - 对比有无残差连接的性能差异
- [ ] **Mini 项目 2**：用 LSTM/GRU 做文本生成（字符级）
  - 在莎士比亚文集或唐诗上训练
  - 对比 RNN / LSTM / GRU 的长序列建模能力

### 核心算法
- CNN：卷积、池化、感受野
- RNN：BPTT、梯度消失/爆炸
- LSTM/GRU：门控机制
- 正则化：Dropout、BatchNorm、LayerNorm

---

## 阶段 2：NLP 基础与注意力机制（第 4-5 周）

### 目标
- 掌握词向量、Seq2Seq、注意力机制
- 理解从 RNN 到 Transformer 的演进逻辑

### 项目
- [ ] **Mini 项目 3**：训练 Word2Vec（Skip-gram + 负采样）
  - 在中文维基百科语料上训练
  - 可视化词向量，验证“国王 - 男人 + 女人 ≈ 女王”
- [ ] **Mini 项目 4**：实现 Seq2Seq + Attention 的机器翻译
  - 英-中或英-法翻译
  - 对比 Bahdanau Attention vs Luong Attention
  - 可视化 Attention 权重对齐

### 核心算法
- Word2Vec、GloVe、FastText
- Seq2Seq、Teacher Forcing
- Bahdanau Attention、Luong Attention
- BLEU 评估指标

---

## 阶段 3：Transformer 架构精讲（第 6-7 周）⭐ 核心阶段

### 目标
- 彻底搞懂 Transformer 的每一个组件
- 从零手写一个可训练的 Transformer

### 项目
- [ ] **Mini 项目 5**：从零实现 "Attention Is All You Need" 完整 Transformer
  - Multi-Head Self-Attention、Positional Encoding
  - Feed-Forward Network、LayerNorm、Residual Connection
  - 在 WMT 翻译任务或 IWSLT 上验证
- [ ] **Mini 项目 6**：实现 GPT-2 级别的小型自回归语言模型
  - 仅 Decoder 架构
  - Causal Self-Attention（因果掩码）
  - 在 OpenWebText 子集上预训练，测试文本生成

### 核心算法
- Self-Attention 的 QKV 计算与复杂度分析
- Multi-Head Attention 的维度切分
- Positional Encoding（正弦 vs 可学习）
- Pre-Norm vs Post-Norm
- Causal Mask 的实现

---

## 阶段 4：LLM 预训练（第 8-10 周）🔥 关键阶段

### 目标
- 理解大规模预训练的完整流程
- 掌握分布式训练的核心概念

### 项目
- [ ] **Mini 项目 7**：在 1B tokens 级别语料上预训练一个 ~100M 参数的 GPT-like 模型
  - 数据预处理：BPE/WordPiece Tokenizer 训练
  - 训练框架：混合精度（FP16/BF16）、梯度累积、Flash Attention
  - 使用 DDP/FSDP 做多卡训练
  - 用 Chinchilla 缩放定律指导 token/param 配比
- [ ] **Mini 项目 8**：复现 LLaMA 架构关键改进
  - RoPE 旋转位置编码
  - RMSNorm
  - SwiGLU 激活函数
  - GQA（分组查询注意力）

### 核心算法
- Tokenization：BPE、WordPiece、SentencePiece
- 缩放定律：Kaplan vs Chinchilla
- 混合精度训练：Loss Scaling
- 分布式：Data Parallel、Model Parallel、Pipeline Parallel
- Flash Attention 原理
- RoPE、ALiBi 等位置编码改进
- GQA / MQA（多查询注意力）

---

## 阶段 5：指令微调与对齐（第 11-12 周）

### 目标
- 掌握 SFT、RLHF、DPO 的对齐技术栈

### 项目
- [ ] **Mini 项目 9**：对预训练模型做 SFT（监督微调）
  - 构建指令数据集（Alpaca 格式）
  - 用 LoRA/QLoRA 微调，对比全参微调
  - 评估：用 GPT-4 做 LLM-as-Judge
- [ ] **Mini 项目 10**：实现 DPO（Direct Preference Optimization）
  - 构建偏好对数据集
  - 对比 DPO vs RLHF 的效果差异
  - 理解 Bradley-Terry 模型到 DPO 损失的推导

### 核心算法
- SFT：Instruction Tuning、Prompt Template
- PEFT：LoRA、QLoRA、Adapter
- RLHF：Reward Model、PPO
- DPO：从 RLHF 到 DPO 的损失函数推导
- 对齐税（Alignment Tax）

---

## 阶段 6：推理优化与部署（第 13-14 周）

### 目标
- 掌握大模型推理的量化、KV Cache、投机采样

### 项目
- [ ] **Mini 项目 11**：实现 KV Cache 自回归推理
  - 手写 KV Cache 管理
  - 对比有无 KV Cache 的推理速度
- [ ] **Mini 项目 12**：模型量化实战
  - GPTQ / AWQ 量化
  - 用 llama.cpp / vLLM 部署模型
  - 对比 FP16 / INT8 / INT4 的推理质量与速度

### 核心算法
- KV Cache 原理与显存计算
- 量化：GPTQ、AWQ、GGUF
- Flash Decoding
- Continuous Batching
- 投机采样（Speculative Decoding）

---

## 阶段 7：前沿架构与多模态（第 15-16 周）

### 目标
- 了解 MoE、Mamba、多模态等前沿方向

### 项目
- [ ] **Mini 项目 13**：实现一个简化版 MoE（Mixture of Experts）
  - 稀疏门控路由（Top-k Gating）
  - 负载均衡损失（Load Balancing Loss）
  - 对比相同参数量下 Dense vs MoE 的性能
- [ ] **Mini 项目 14**：构建简易多模态模型
  - CLIP 风格的双塔对齐
  - LLaVA 风格的视觉-语言模型
  - 图文检索或视觉问答

### 核心算法
- MoE：Sparse Gating、Expert Capacity、Load Balancing
- Mamba / SSM（状态空间模型）
- CLIP：对比学习、双塔架构
- LLaVA：视觉 Encoder + LLM Projector
- 长上下文：RoPE 外推、YaRN、Ring Attention

---

## 阶段 8：Agent 与应用（第 17-18 周）

### 目标
- 掌握 LLM Agent、RAG、Function Calling

### 项目
- [ ] **Mini 项目 15**：构建 RAG 问答系统
  - 文档切片（Chunking）、Embedding 检索
  - 混合检索（BM25 + 向量检索）
  - Re-Rank 重排序
- [ ] **Mini 项目 16**：实现 ReAct Agent
  - Tool Use / Function Calling
  - 多步推理 + 工具调用循环
  - 简易 Code Interpreter

### 核心算法
- RAG：Chunking 策略、Embedding 模型、向量数据库
- Agent：ReAct、Plan-and-Execute
- Tool Use：Function Call Schema、指令解析
- 多 Agent 协作

---

## 🛠 推荐技术栈

| 领域 | 工具/库 |
|------|---------|
| 框架 | PyTorch 2.x |
| 训练加速 | Flash Attention 2, DeepSpeed, FSDP |
| 微调 | HuggingFace Transformers, PEFT, TRL |
| 推理 | vLLM, llama.cpp, TensorRT-LLM |
| 数据 | HuggingFace Datasets, Ray Data |
| 实验管理 | Weights & Biases, TensorBoard |
| 语料 | C4, The Pile, Wikipedia, 天工中文 |

---

## 📚 参考资源

1. **《动手学深度学习》** — 李沐 (d2l.ai)
2. **《Attention Is All You Need》** — Vaswani et al., 2017
3. **Andrej Karpathy** — "Let's build GPT from scratch" (nanoGPT)
4. **HuggingFace NLP Course** — huggingface.co/learn
5. **LLaMA 论文** — Touvron et al., 2023
6. **Chinchilla 缩放定律** — Hoffmann et al., 2022
7. **DPO 论文** — Rafailov et al., 2023

---

## 🎯 学习建议

1. **代码先行**：每看完一个算法，立刻手写实现，不要只看不写
2. **从小规模开始**：先在 1M tokens 上跑通，再放大到 1B tokens
3. **记录踩坑日志**：每个项目建一个 `NOTES.md`，记录 debug 过程
4. **逐层吃透 Transformer**：阶段 3 是整个学习的基石，务必吃透每一个维度变换
5. **善用 Colab / AutoDL**：预训练阶段需要 GPU，提前准备好算力资源
