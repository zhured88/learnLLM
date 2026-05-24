# 阶段 3 技术详解：Transformer 架构精讲

> 配套项目 `stage3-transformer/`，包含两个 Mini 项目：
> - `transformer-translation/` — 从零实现 "Attention Is All You Need" 完整 Transformer 翻译模型
> - `gpt2-lm/` — GPT-2 风格 Decoder-Only Transformer 预训练 + 文本生成

---

## 目录

1. [从 Attention 到 Self-Attention：范式的跃迁](#1-从-attention-到-self-attention范式的跃迁)
   - [1.1 RNN 的三个原罪](#11-rnn-的三个原罪)
   - [1.2 Self-Attention 的核心直觉](#12-self-attention-的核心直觉)
   - [1.3 Q/K/V：查询、键、值的物理解释](#13-qkv查询键值的物理解释)
2. [Scaled Dot-Product Attention：一个等式的逐项拆解](#2-scaled-dot-product-attention一个等式的逐项拆解)
   - [2.1 点积和 softmax](#21-点积和-softmax)
   - [2.2 为什么除以 √d_k](#22-为什么除以-d_k)
   - [2.3 因果掩码](#23-因果掩码)
3. [Multi-Head Attention：多重视角](#3-multi-head-attention多重视角)
4. [Positional Encoding：位置信息的注入](#4-positional-encoding位置信息的注入)
   - [4.1 为什么 Transformer 需要位置编码](#41-为什么-transformer-需要位置编码)
   - [4.2 Sinusoidal vs 可学习位置编码](#42-sinusoidal-vs-可学习位置编码)
5. [Transformer 完整架构](#5-transformer-完整架构)
   - [5.1 Encoder](#51-encoder)
   - [5.2 Decoder](#52-decoder)
   - [5.3 Pre-Norm vs Post-Norm](#53-pre-norm-vs-post-norm)
   - [5.4 Label Smoothing 和 Adam Warmup](#54-label-smoothing-和-adam-warmup)
6. [GPT-2：Decoder-Only 的崛起](#6-gpt-2decoder-only-的崛起)
   - [6.1 从 Transformer Decoder 到 GPT-2](#61-从-transformer-decoder-到-gpt-2)
   - [6.2 Causal Self-Attention](#62-causal-self-attention)
   - [6.3 GELU 激活函数](#63-gelu-激活函数)
   - [6.4 Weight Tying](#64-weight-tying)
7. [实验结果与对比](#7-实验结果与对比)
8. [动手练习](#8-动手练习)
9. [代码逐行讲解](#9-代码逐行讲解)

---

## 1. 从 Attention 到 Self-Attention：范式的跃迁

### 1.1 RNN 的三个原罪

阶段 1 和 2 中的 RNN/LSTM 有三个无法克服的限制，Transformer 论文正是冲着这三个问题来的：

| 原罪 | 描述 | Transformer 的解决方案 |
|------|------|----------------------|
| **无法并行** | 第 t 步依赖第 t-1 步的 hidden state，无法并行计算 | Self-Attention：所有位置两两直接连接，O(1) 的序列操作 |
| **长程梯度消失** | 即使 LSTM 有门控，100+ 步的信息传递仍然困难 | 任意两个位置直接建立连接，梯度路径长度 = 2（经由 attention 权重） |
| **信息瓶颈** | Encoder 最后一个 hidden state 要编码整个句子 | 没有瓶颈——Decoder 每一层都可以通过 Cross-Attention 直接读取 Encoder 所有位置 |

**核心数字**：RNN 处理序列长度 L 需要 O(L) 个串行步骤；Self-Attention 只需要 O(1) 个串行步骤。这就是为什么 Transformer 能扩展到千亿参数的根本原因——它可以充分利用 GPU 的并行算力。

### 1.2 Self-Attention 的核心直觉

在阶段 2 的注意力机制中，Decoder 关注 Encoder（Cross-Attention）。Self-Attention 更进一步：**序列中的每个位置关注同一序列中的所有其他位置（包括自己）。**

```
"I like the dog because it is cute"

没有 Self-Attention：模型读到 "it" 时，不知道它指代的是 "dog" 还是 "I" 还是 "the"
有 Self-Attention：  "it" 和 "dog" 之间的注意力权重最高 → 模型建立了指代消解
```

Self-Attention 本质上是一种**动态的路由机制**：每一层，每个 token 决定从其他 token 那里"获取多少信息"。

### 1.3 Q/K/V：查询、键、值的物理解释

Q/K/V 术语来自信息检索，但在 Self-Attention 语境下更直观的类比如下：

| 角色 | 含义 | 形状 |
|------|------|------|
| Query (Q) | "我想要什么信息？" —— 每个 token 根据自己的 Query 去"查找" | (B, S, d_k) |
| Key (K) | "我这里有什么信息？" —— 每个 token 提供一个"索引标签" | (B, S, d_k) |
| Value (V) | "我的实际内容是什么？" —— 每个 token 提供实际要传递的信息 | (B, S, d_k) |

**一个词扮演三种角色**：同一个词同时有 Q、K、V 三种表示——它们通过三个不同的线性投影得到。一个词作为"查询者"时关注什么，和它作为"被查询者"时提供什么信息，是可以不同的。

---

## 2. Scaled Dot-Product Attention：一个等式的逐项拆解

Transformer 论文中的注意力公式是整个深度学习领域最重要的单行公式之一：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### 2.1 点积和 softmax

$$\text{Scores} = Q \cdot K^T$$

对于位置 i：
- $Q_i$（第 i 个词的 query）和所有 K（所有词的 key）做点积
- 点积越大 → 两个词在当前层的表示空间中越"相关"
- 结果是一个 L×L 的矩阵：第 i 行是"位置 i 对所有其他位置的关注得分"

Softmax 沿每行归一化：$\sum_j \alpha_{i,j} = 1$。每一行是一个概率分布——第 i 个词把注意力分配给了谁。

### 2.2 为什么除以 √d_k

**如果不除以 √d_k 会怎样？**

点积 $Q_i \cdot K_j$ 是 d_k 个独立随机变量的和。按中心极限定理：
$$Q_i \cdot K_j \sim \mathcal{N}(0, d_k)$$

d_k=64 时，点积的标准差约 8。如果某个位置的点积恰好比较大（比如 15），另一个比较小（比如 -10），softmax 后的权重几乎是 [1, 0, 0, ...]，梯度趋近于 0。

**除以 √d_k 的作用**：把点积的方差从 d_k 变为 1，确保 softmax 处于"不饱和"区域（梯度较大）。

$$\frac{QK^T}{\sqrt{d_k}} \sim \mathcal{N}(0, 1)$$

**实验验证**：如果不除以 √d_k，随着 d_k 增大，模型训练迅速退化。

### 2.3 因果掩码

GPT/LLaMA 这类 Decoder-Only 模型的关键约束：**不能偷看未来的词**。

语言模型的任务是 $p(x_t | x_{<t})$——预测下一个 token 时只能看到已经生成的 token。实现方式是在 $QK^T$ 的上三角加上 `-inf`：

```python
# 因果掩码 = 下三角矩阵
mask = [[0, -inf, -inf, -inf],
        [0,   0,  -inf, -inf],
        [0,   0,    0,  -inf],
        [0,   0,    0,    0 ]]

scores = QK^T / sqrt(d_k) + mask  # -inf 位置 softmax 后 = 0
```

---

## 3. Multi-Head Attention：多重视角

单头注意力的一个局限：它只能关注一种关系模式。多头注意力用平行的多组 Q/K/V，每组关注不同的语义关系。

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**维度分配**：

$$d_{model} = h \times d_k$$

- d_model=512, h=8 → d_k=64
- 每个头处理 64 维的子空间
- 8 个头并行，各自学会不同的注意力模式
- 最后 concat 回 512 维

**多头在学什么？**（来自论文的可视化分析）

| 头 | 大约在学什么 |
|----|------------|
| 头 1-2 | 相邻词的局部语法关系 |
| 头 3-5 | 指代消解（代词→先行词） |
| 头 6-7 | 动词-宾语的长程语义依赖 |
| 头 8 | 和标点、句法结构相关的全局模式 |

---

## 4. Positional Encoding：位置信息的注入

### 4.1 为什么 Transformer 需要位置编码

Self-Attention 是对称的——交换两个位置的词，注意力权重只是换了行，网络不会自动知道"谁在左、谁在右"。

对比：
- RNN：位置信息天然在时间步顺序中——先处理第 1 个词，再第 2 个...
- CNN：卷积核的局部感受野自带位置感
- Transformer：Self-Attention 把序列当成一个**集合**——必须显式注入位置

### 4.2 Sinusoidal vs 可学习位置编码

| 特性 | Sinusoidal (原始 Transformer) | 可学习 (GPT/LLaMA) |
|------|---------------------------|-------------------|
| 参数 | 0 | L × d_model |
| 外推能力 | 可以（三角函数有规律） | 不能（训练外的位置是未定义的） |
| 表达能力 | 固定 | **更强**（数据驱动的） |
| 使用 | 翻译等 Encoder-Decoder | 语言模型（Decoder-Only） |

**为什么 GPT 选可学习？** 预训练数据量足够大（TB 级），可学习的位置编码能从数据中学到比固定三角函数更好的表示。而翻译任务数据量较小，sinusoidal 的归纳偏置反而有帮助。

**Sinusoidal 的数学直觉**：

$$\text{PE}(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$

$$\text{PE}(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

- i=0：波长 ≈ 2π ≈ 6 个 token → 编码"相邻词"
- i=d/2-1：波长 ≈ 10000 × 2π ≈ 62832 个 token → 编码"全局绝对位置"
- 每个维度对应一个不同频率的正弦波 → 位置 pos 的编码向量是"在这个位置采样不同频率正弦波得到的值"

---

## 5. Transformer 完整架构

### 5.1 Encoder

```
输入 (B, src_len)
  ↓
Token Embedding + Positional Encoding
  ↓
EncoderLayer × 6:
  ├── Multi-Head Self-Attention
  ├── Add & Norm        ← 残差连接 + LayerNorm
  ├── Feed-Forward      ← d_model → d_ff → d_model
  └── Add & Norm
  ↓
输出给 Decoder 做 Cross-Attention
```

### 5.2 Decoder

```
输入 (B, tgt_len)
  ↓
Token Embedding + Positional Encoding
  ↓
DecoderLayer × 6:
  ├── Masked Multi-Head Self-Attention  ← 因果掩码（不能看未来）
  ├── Add & Norm
  ├── Multi-Head Cross-Attention        ← Q from decoder, K&V from encoder
  ├── Add & Norm
  ├── Feed-Forward
  └── Add & Norm
  ↓
Linear + Softmax → 预测下一个 token
```

### 5.3 Pre-Norm vs Post-Norm

原始论文用 Post-Norm（子层之后做 LayerNorm），后来的 GPT/LLaMA 改用 Pre-Norm（子层之前做）：

| | Post-Norm (2017) | Pre-Norm (GPT-2+) |
|------|-----------------|-------------------|
| 结构 | x = LN(x + Sublayer(x)) | x = x + Sublayer(LN(x)) |
| 梯度流 | 梯度必须穿过 LN → 轻微衰减 | 梯度直接通过残差分支 → 无衰减 |
| 训练稳定性 | 需要 warmup | **更稳定**，warmup 可选 |
| 深层（100+ 层） | 容易发散 | **稳定** |

### 5.4 Label Smoothing 和 Adam Warmup

**Label Smoothing**：在 one-hot label 上添加均匀噪声。

- 原始 cross-entropy：loss = -log(p_correct)，模型被鼓励把正确 token 的概率推到 1.0
- Label smoothing：loss = (1-ε) × (-log(p_correct)) + ε × mean(-log(p))
- 效果：防止模型过度自信，提升泛化（特别是 BLEU 分数上 ~0.5-1 点提升）

**Adam Warmup**：前 N 步线性增加学习率，然后逐步衰减。

Transformer 训练初期，所有 token 的 embedding 是随机的，attention 权重几乎是均匀的，梯度不稳定。Warmup 让模型在"学会基本分布之前"不要走太大步。

---

## 6. GPT-2：Decoder-Only 的崛起

### 6.1 从 Transformer Decoder 到 GPT-2

GPT-2 的本质是**砍掉了 Transformer 的 Encoder**：

| 组件 | Transformer | GPT-2 |
|------|-----------|-------|
| Encoder | 双向 Self-Attention | —（删除） |
| Decoder Self-Attention | 因果 Self-Attention | ✅ 保留 |
| Cross-Attention | 关注 Encoder 输出 | —（删除，没有 Encoder） |
| FFN | ReLU | GELU |
| Norm | Post-Norm | Pre-Norm |
| Positional Encoding | Sinusoidal | 可学习 |

### 6.2 Causal Self-Attention

和 Transformer Decoder 的 Self-Attention 完全相同：

```python
def forward(self, x):
    qkv = self.c_attn(x)  # QKV 投影合并为一次矩阵乘法
    q, k, v = qkv.split(n_embd, dim=2)
    # ... 多头拆分 + 因果掩码 + softmax + 加权求和 ...
    return self.c_proj(y)
```

**GPT-2 的优化：QKV 投影合并**
原始 Transformer 用三个独立的 Linear(W_q, W_k, W_v)。GPT-2 把三个合并为一个 `c_attn: Linear(d, 3d)`，然后 split 为三个。这样做减少了 kernel launch 开销，实际速度更快。

### 6.3 GELU 激活函数

GPT-2 用 GELU（Gaussian Error Linear Unit）替代 ReLU：

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$

- ReLU：x < 0 时输出完全为 0（"硬"门控）
- GELU：x < 0 时输出接近 0 但不完全为 0（"软"门控）
- GELU 提供非零的梯度在整个实数域上，对深层网络的训练有帮助
- 实践中常用近似公式：$\text{GELU}(x) \approx 0.5x \cdot (1 + \tanh(\sqrt{2/\pi} \cdot (x + 0.044715x^3)))$

### 6.4 Weight Tying

GPT-2 把输入 embedding 矩阵和输出预测矩阵绑定为同一个：

```python
self.lm_head.weight = self.wte.weight  # 共享权重
```

**为什么？**
- 输入 embedding 把 token 映射到向量空间
- 输出 lm_head 把向量空间映射回 token 概率
- 这两个映射是互为逆运算——共享权重是合理的归纳偏置
- 节省约 (vocab_size × d_model) 个参数（通常占模型的 20-30%）

---

## 7. 实验结果与对比

### 7.1 Transformer 翻译

运行：

```bash
cd transformer-translation
python main.py --d_model 256 --num_heads 4 --num_layers 4 --epochs 15
```

| 配置 | 期望 Val PPL | 参数量 | 备注 |
|------|------------|--------|------|
| d=256, h=4, L=4 (Tiny) | ~3-5 | ~6M | 快速实验用 |
| d=512, h=8, L=6 (Base) | ~2-3 | ~60M | 原始论文配置 |
| 无 Label Smoothing | +0.5-1 PPL | — | 过度自信导致泛化差 |
| 无 Warmup | 可能不收敛 | — | 训练初期梯度不稳定 |

### 7.2 GPT-2 预训练

运行：

```bash
cd gpt2-lm
python main.py --n_layer 4 --n_embd 256 --epochs 5
```

| 配置 | 期望 Val PPL | 参数量 |
|------|------------|--------|
| L=4, D=256, H=4 (Tiny) | ~30-60 | ~8M |
| L=12, D=768, H=12 (Small) | ~20-30 | ~124M |

### 7.3 Encoder-Decoder vs Decoder-Only 对比

| 维度 | Transformer (Enc-Dec) | GPT-2 (Dec-Only) |
|------|---------------------|-------------------|
| 任务 | 序列映射（翻译、摘要） | 序列生成（续写、对话） |
| 上下文 | 双向 Encoder + 因果 Decoder | 仅因果 |
| 推理 | 需要 Encoder 和 Decoder 各跑一次 | 同一次前向即可 |
| 参数效率 | 两个独立参数集 | 单一参数集 |

---

## 8. 动手练习

| 序号 | 练习 | 难度 | 要改的文件 |
|------|------|------|-----------|
| 1 | 对比三种位置编码：Sinusoidal vs 可学习 vs NoPE（无位置编码） | ⭐⭐ | `embedding.py`, `gpt2.py` |
| 2 | 实现 Pre-Norm → Post-Norm 切换，对比训练曲线 | ⭐⭐ | `transformer.py`, `attention.py` |
| 3 | 可视化 Self-Attention 权重：分析 Encoder 中不同层/不同头的关注模式 | ⭐⭐ | 新建 `visualize_attn.py` |
| 4 | GPT-2 生成参数消融：对比 temperature / top-k / top-p 的生成效果 | ⭐ | `gpt2.py` |
| 5 | 手工实现 Flash Attention 的 tiling 思想（简化版，不依赖 CUDA） | ⭐⭐⭐ | 新建 `flash_attention.py` |
| 6 | 对比 d_k 缩放的效果：去掉 /√d_k，观察训练是否不稳定 | ⭐ | `attention.py` |

---

## 9. 代码逐行讲解

### 9.1 Multi-Head Self-Attention 的维度操作

这是 Transformer 最容易让人困惑的地方。让我们用一个具体的例子追踪每一步。

**场景**：B=32（batch），S=50（seq_len），d_model=512，num_heads=8

```python
def _split_heads(self, x):
    # x: (32, 50, 512)
    B, S, _ = x.shape
    x = x.view(B, S, 8, 64)     # (32, 50, 8, 64)
    return x.permute(0, 2, 1, 3)  # (32, 8, 50, 64)  ← "head" 成为第 2 维
```

**为什么要 permute？** 因为 batch matrix multiply `torch.matmul(Q, K.transpose(-2, -1))` 在最后两维上做矩阵乘法。我们需要 (B, h, S, d_k) × (B, h, d_k, S) → (B, h, S, S)，即每个头独立算注意力。

```python
def _merge_heads(self, x):
    # x: (32, 8, 50, 64)
    B, _, S, _ = x.shape
    x = x.permute(0, 2, 1, 3).contiguous()  # (32, 50, 8, 64)
    return x.view(B, S, 512)                  # (32, 50, 512)
```

**`.contiguous()` 的作用**：permute 后的 tensor 在内存中不连续，view 需要连续内存。`.contiguous()` 重新排列内存布局（复制一份），然后 view 才能正常工作。

### 9.2 GPT-2 的 QKV 合并投影

```python
# 原始 Transformer（三次独立的矩阵乘法）：
q = W_q(x)  # (B, T, 768)
k = W_k(x)
v = W_v(x)

# GPT-2 的优化（一次矩阵乘法 → split）：
qkv = c_attn(x)  # (B, T, 2304)  ← 2304 = 3 × 768
q, k, v = qkv.split(768, dim=2)  # 各 (B, T, 768)
```

**为什么更快？** 一次大矩阵乘法比三次小矩阵乘法快，因为 GPU 可以更充分地利用 Tensor Core。而且 `c_attn.weight` 形状为 (2304, 768)，一次内存读取就加载了所有 QKV 的权重。

### 9.3 Adam Warmup 的公式实现

```python
class AdamWarmup:
    def _get_lr(self):
        step = self.step_num
        arg1 = step ** (-0.5)         # 衰减线：1/√step
        arg2 = step * (warmup ** (-1.5))  # 上升线：step/warmup^1.5
        return d_model**(-0.5) * min(arg1, arg2)
```

**公式的可视化**：横轴是 step，纵轴是学习率

```
lr
^
|     /‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
|    /                            ← arg1 = 1/√step (衰减)
|   /
|  /                              ← arg2 = step/warmup^1.5 (上升)
| /
+──┴──────────────────────→ step
    warmup_steps
```

前 warmup_steps 步：`min(arg1, arg2) = arg2`（上升线更低）→ 线性增加
warmup 之后：`min(arg1, arg2) = arg1`（衰减线更低）→ 逐渐减小

### 9.4 Label Smoothing 的实现

```python
class LabelSmoothingLoss:
    def forward(self, logits, target):
        # nll: 标准的负对数似然（鼓励 p_correct → 1）
        nll = F.cross_entropy(logits, target, reduction="none")

        # smooth: 均匀分布的负对数似然（鼓励不选其他 token）
        log_probs = F.log_softmax(logits, dim=-1)
        smooth_loss = -log_probs.mean(dim=-1)

        # (1-ε) × 正确 + ε × 均匀 = 不让模型过度自信
        loss = (1 - smoothing) * nll + smoothing * smooth_loss
        return loss.mean()
```

**为什么不用简单的 `kl_div`？** 因为 target 是均匀分布，均匀分布的熵是 -log(V)，可以直接用 `-mean(log_probs)` 等价计算。省了创建 target 分布 tensor 的步骤。

### 9.5 从阶段 3 到阶段 4：GPT-2 → LLaMA

阶段 4 将介绍 LLaMA 架构——它在 GPT-2 基础上的四个关键改进：

| GPT-2 (阶段 3) | LLaMA (阶段 4) | 改进原因 |
|---------------|---------------|---------|
| LayerNorm | **RMSNorm** | 去掉均值的计算，更快 |
| GELU 激活 | **SwiGLU** | Gated Linear Unit + Swish → 更强表达能力 |
| 可学习 PE | **RoPE** | 旋转位置编码：通过旋转注入相对位置信息 |
| Multi-Head | **GQA (Grouped Query)** | 多个 Q 头共享一组 KV 头 → 推理时省 KV Cache |
