# Stage 4 · LLM 预训练技术详解

> 从 GPT-2 到 LLaMA：用四项架构改进将模型从 ~8M 参数规模化到 ~100M 参数

---

## 白话先行

在阶段 3，我们搭了一个 GPT-2 的"概念车"——能跑，但只有 800 万参数，字符级分词，LayerNorm 归一化，GELU 激活。到了阶段 4，我们要把它升级成"量产车"。每一项改进的直觉如下：

- **RMSNorm**：LayerNorm 每次要对输入算两个统计量（均值 + 方差），RMSNorm 只算一个（均方根）。少一个归约操作，在前向/反向传播中都能省。对 100 层的大模型，这个节省是显著的。

- **SwiGLU**：GELU 是非线性变换，把不重要的输入压到接近零；SwiGLU 更进一步——它用"门控"（gate）机制，让网络自己学习"这个输入该不该通过"。就像公司审批流程：GELU 是一个固定的审批标准，SwiGLU 是一群能根据上下文动态调整的审批员。

- **RoPE**：可学习位置编码是为每个绝对位置学一个向量——位置 5 的向量和位置 10 的向量完全独立。RoPE 不学绝对位置，而是通过复数旋转把相对位置编码进 Q 和 K 的点积里。好处是：训练时没见过长度 4096 的序列，推理时也能外推到更长。

- **GQA**：多 Query 头共享 K/V 头。我们在阶段 3 知道了 MHA 每个头都有独立的 Q、K、V。推理时，每生成一个 token 就要缓存所有 K/V——对 32 头 × 4096 长度的序列，KV Cache 是天文数字。GQA 让 4 个 Q 头共享一组 K/V，把 KV Cache 直接砍掉 4 倍。

---

## 1. 从 GPT-2 到 LLaMA：架构演进

| 组件 | GPT-2 (Stage 3) | LLaMA (Stage 4) | 改进动机 |
|------|-----------------|-----------------|----------|
| 归一化 | LayerNorm | **RMSNorm** | 少算均值，训练更快 |
| 激活函数 | GELU | **SwiGLU (SiLU)** | 门控 + Swish → 更强表达能力 |
| 位置编码 | 可学习 PE | **RoPE** | 相对位置、长度外推 |
| 注意力 | Multi-Head (MHA) | **Grouped-Query (GQA)** | 省 KV Cache，推理更快 |
| FFN 结构 | Linear→GELU→Linear | **gate·proj ⊙ up·proj → down·proj** | 门控线性单元 |

### 1.1 为什么需要这些改进？

GPT-2 是 2019 年的架构。LLaMA（2023）吸收了三年间的研究成果：

1. **训练效率**：RMSNorm 在 Transformer 论文发表前的 2019 年就被提出，但直到大规模预训练普及后才被广泛采用。
2. **表达力**：GLU Variants 论文（Shazeer, 2020）系统比较了各种激活函数，SwiGLU 在各尺寸模型上一致优于 ReLU/GELU。
3. **外推能力**：可学习 PE 的最大缺点是无法外推到更长序列。RoPE 通过相对位置编码解决了这个问题。
4. **推理成本**：LLM 部署的瓶颈是 KV Cache 占用。MQA（Multi-Query，1 组 KV）和 GQA（多组 KV）直接把 KV Cache 砍到 1/4~1/8。

---

## 2. RMSNorm — 均方根归一化

### 2.1 数学定义

LayerNorm:
$$y = \frac{x - \mu}{\sigma} \cdot \gamma + \beta$$

RMSNorm:
$$y = \frac{x}{\sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}} \cdot \gamma$$

### 2.2 关键洞察

Zhang & Sennrich (2019) 的论文发现：**LayerNorm 成功的关键是缩放（scaling），而非中心化（centering）**。

具体来说：
- 去掉均值 $\mu$：$\text{rms}(x) = \sqrt{\text{mean}(x^2)}$ 替代 $\sigma(x)$
- 去掉 bias $\beta$：只保留缩放参数 $\gamma$

**计算量对比：**

| 操作 | LayerNorm | RMSNorm |
|------|-----------|---------|
| sum(x) | ✓ | ✗ |
| mean(x²) | ✓ | ✓ |
| sqrt | ✓ | ✓ |
| 总归约次数 | 2 | 1 |

对张量 `(B, T, 4096)` 做 LayerNorm 需要两次全规约（计算 mean 和 var），RMSNorm 只需要一次（计算 mean(x²)）。

### 2.3 代码实现

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.gamma
```

---

## 3. SwiGLU — 门控线性单元

### 3.1 从 GELU 到 SwiGLU

**GELU** (GPT-2):
$$\text{GELU}(x) = x \cdot \Phi(x) \approx 0.5x(1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3)))$$
$$\text{GELU-MLP}(x) = W_{proj} \cdot \text{GELU}(W_{fc} \cdot x)$$

**SwiGLU** (LLaMA):
$$\text{SiLU}(x) = x \cdot \sigma(x) \quad (\sigma = \text{sigmoid})$$
$$\text{SwiGLU}(x) = \underbrace{\text{SiLU}(xW_{gate} + b_{gate})}_{\text{门控值}} \odot \underbrace{(xW_{up} + b_{up})}_{\text{输入值}} \cdot W_{down}$$

### 3.2 维度关系

传统 GELU-MLP 的参数量：$2 \times d \times 4d = 8d^2$

SwiGLU 有三个矩阵 ($W_{gate}$, $W_{up}$, $W_{down}$)，为保证参数量一致：
$$2 \times d \times d' + d' \times d = 3d \cdot d' = 8d^2$$
$$\Rightarrow d' = \frac{8}{3}d$$

LLaMA 具体实现中，为了硬件友好，将 $d'$ 向上取整到 256 的倍数。

### 3.3 为什么 SwiGLU 更好？

1. **门控机制**：SiLU(gate) 学习"哪些信息该通过"，up 提供实际信息。这比 GELU 的固定非线性灵活得多。
2. **梯度流动**：SiLU 处处可导，在正负区间都有梯度（sigmoid 部分保证即使是 x<0 也有非零梯度）。
3. **实证结果**：Shazeer (2020) 在 T5 上比较了所有 GLU 变体，SwiGLU 在所有下游任务上一致最优。

### 3.4 代码实现

```python
class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(2 * 4 * dim / 3)  # 8d/3
            hidden_dim = ((hidden_dim + 255) // 256) * 256
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))  # SiLU(xW_g)
        up = self.up_proj(x)              # xW_u
        return self.down_proj(gate * up)   # (gate ⊙ up)W_d
```

---

## 4. RoPE — 旋转位置编码

### 4.1 核心思想

传统的可学习位置编码 $p_m$ 和词向量 $x_m$ 直接相加：$q_m = W_q(x_m + p_m)$。问题是 $q_m^T k_n$ 同时依赖绝对位置 $m$ 和 $n$，无法自然地外推到训练时未见过的位置。

RoPE 不直接加位置向量，而是**用位置 $m$ 对向量 $q$ 做旋转**：
$$f_q(x_m, m) = R_m \cdot x_m$$

其中 $R_m$ 是旋转矩阵：
$$R_m = \begin{bmatrix}
\cos(m\theta_1) & -\sin(m\theta_1) & 0 & 0 & \cdots \\
\sin(m\theta_1) & \cos(m\theta_1) & 0 & 0 & \cdots \\
0 & 0 & \cos(m\theta_2) & -\sin(m\theta_2) & \cdots \\
0 & 0 & \sin(m\theta_2) & \cos(m\theta_2) & \cdots \\
\vdots & \vdots & \vdots & \vdots & \ddots
\end{bmatrix}$$

频率定义：
$$\theta_i = 10000^{-2i/d}, \quad i = 0, 1, ..., d/2-1$$

### 4.2 相对位置性质

两个旋转后的向量做内积：
$$(R_m \cdot q)^T (R_n \cdot k) = q^T R_m^T R_n k = q^T R_{n-m} k$$

**关键结论**：$q_m^T k_n$ 只依赖于相对位置 $(n-m)$，不依赖于绝对位置 $m$ 和 $n$。

比如 $q_0^T k_3 = q_5^T k_8$（两组距离都是 3）。

### 4.3 长序列外推

因为只依赖相对位置，RoPE 自然支持外推：
- 训练时用长度 2048
- 推理时用长度 4096 → 只需扩展 `cos_cached` / `sin_cached` 到 4096


### 4.4 不同维度的旋转频率

$$\theta_i = 10000^{-2i/d}$$

- i=0 (低频): $\theta_0 = 1$，旋转最慢，编码长距离依赖
- i=d/2-1 (高频): $\theta_{d/2-1} \approx 10^{-4}$，旋转最快，编码短距离依赖

这种多频率设计让 RoPE 像"傅里叶级数"——低频捕获全局结构，高频捕获局部模式。

---

## 5. GQA — 分组查询注意力

### 5.1 三种注意力机制

| 类型 | Q 头数 | K/V 头数 | KV Cache 大小 | 质量 |
|------|--------|----------|---------------|------|
| MHA | n | n | 2·n·d_k·L | 最优 |
| MQA | n | 1 | 2·1·d_k·L | 有损 |
| GQA | n | g (1<g<n) | 2·g·d_k·L | 接近 MHA |

### 5.2 直觉

- **MQA**：所有 Q 头共享同一组 K/V。极端节约 KV Cache（1/n），但信息损失太大。
- **GQA**：取折中。n 个 Q 头分成 g 组，每组内的 Q 头共享 K/V。
  - g = n → MHA
  - g = 1 → MQA
  - g = n/4 (LLaMA 3) → 节省 4× KV Cache

### 5.3 KV Cache 节省量计算

序列长度为 L 时，每层的 KV Cache 大小：
- MHA: $2 \cdot n_{head} \cdot d_k \cdot L$ 个 float
- GQA: $2 \cdot n_{kv\_head} \cdot d_k \cdot L$ 个 float

节约比例 = $\frac{n_{kv\_head}}{n_{head}}$

LLaMA 3 8B: n_head=32, n_kv_head=8 → 节约 75% 的 KV Cache。

### 5.4 GQA 的实现

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, dim, n_head, n_kv_head):
        assert n_head % n_kv_head == 0
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_rep = n_head // n_kv_head  # 每组 Q 头对应 1 个 KV 头
        self.d_k = dim // n_head

        self.q_proj = nn.Linear(dim, n_head * self.d_k, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_head * self.d_k, bias=False)  # 注意: n_kv_head
        self.v_proj = nn.Linear(dim, n_kv_head * self.d_k, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, mask=None, rope=None):
        # ... Q/K/V 投影 ...
        k = k.repeat_interleave(self.n_rep, dim=1)  # 复制 K/V 头
        v = v.repeat_interleave(self.n_rep, dim=1)
        # ... Scaled Dot-Product Attention ...
```

---

## 6. LLaMA 完整架构

### 6.1 模型结构

```
LLaMA(
  ├── Token Embedding (vocab_size × n_embd)      # 词嵌入（无可学习位置编码）
  ├── Transformer Blocks × n_layer
  │   ├── RMSNorm                               # Pre-Norm
  │   ├── GroupedQueryAttention
  │   │   ├── Q/K/V 投影 (q/k 含 RoPE)
  │   │   └── RoPE (仅 q 和 k)
  │   ├── Residual Connection
  │   ├── RMSNorm                               # Pre-Norm
  │   ├── SwiGLU
  │   │   ├── gate_proj → SiLU
  │   │   ├── up_proj
  │   │   └── gate ⊙ up → down_proj
  │   └── Residual Connection
  ├── RMSNorm (final)
  └── lm_head (n_embd × vocab_size, weight tied with embedding)
)
```

### 6.2 预置模型配置

| 配置 | n_embd | n_layer | n_head | n_kv_head | 参数量 |
|------|--------|---------|--------|-----------|--------|
| tiny | 384 | 8 | 6 | 2 | ~25M |
| small | 576 | 12 | 9 | 3 | ~55M |
| base | 768 | 12 | 12 | 4 | ~105M |

### 6.3 base 配置参数量计算

以 vocab_size=32000, n_embd=768, n_layer=12, n_head=12, n_kv_head=4 为例：

```
Embedding:
  Token Embedding: 32000 × 768 = 24,576,000

每层 LLaMABlock:
  Attention Norm (RMSNorm):     768 参数
  Q 投影:                        768 × 768 = 589,824
  K 投影:                        768 × 256 = 196,608 (4 KV heads × 64)
  V 投影:                        768 × 256 = 196,608
  O 投影:                        768 × 768 = 589,824
  FFN Norm (RMSNorm):           768
  SwiGLU gate_proj:             768 × 2048 = 1,572,864  (8/3 × 768 ≈ 2048)
  SwiGLU up_proj:               768 × 2048 = 1,572,864
  SwiGLU down_proj:             2048 × 768 = 1,572,864
  每层合计:                                  ≈ 6.3M

12层合计:                                   ≈ 75.5M

Final Norm (RMSNorm):                         768
LM Head:       (与 Embedding 共享参数，不计)    0

总参数量:      ≈ 100M (含 Embedding)
```

---

## 7. 预训练基础设施

### 7.1 混合精度训练 (AMP)

使用 `torch.cuda.amp` 的自动混合精度：
- `autocast`：前向传播自动将部分计算转为 FP16
- `GradScaler`：放大 loss 以防止 FP16 梯度下溢

```python
scaler = torch.amp.GradScaler("cuda")
with torch.amp.autocast("cuda"):
    logits, loss = model(x, targets=y)
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
# ... gradient clipping ...
scaler.step(optimizer)
scaler.update()
```

### 7.2 梯度累积

模拟更大的 batch size 而不用增加显存：

```
Effective Batch = per_step_batch × grad_accum_steps × seq_len
```

例如 `--batch_size 8 --grad_accum 8` → 模拟 `batch_size=64`。

### 7.3 学习率调度

```
Step < warmup_steps:  lr = peak_lr × (step / warmup_steps)
Step >= warmup_steps: lr = min_lr + 0.5 × (peak_lr - min_lr) × (1 + cos(π × progress))
```

前 warmup_steps 步线性增长防止冷启动不稳定，之后 cosine decay 平滑衰减到 min_lr。

### 7.4 数据流水线

```
原始文本 (.txt)
  → Tokenizer.encode() [BPE]
  → token IDs (numpy uint16 → .bin 缓存)
  → LMDataset (滑动窗口 seq_len=2048)
  → DataLoader (batch, shuffle, pin_memory)
  → 训练
```

使用 `.bin` 文件预 tokenize + 缓存，避免每 epoch 重新 encode。

### 7.5 检查点机制

```python
checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scaler_state_dict": scaler.state_dict(),
    "loss": val_loss,
    "model_config": { "vocab_size", "n_embd", "n_layer", "n_head", "n_kv_head" }
}
```

支持中断恢复：`python main.py --resume checkpoints/llama_tiny_epoch05.pt`

---

## 8. 实验与预期结果

### 8.1 WikiText-2 (tiny 模型, CPU)

| 指标 | 预期值 |
|------|--------|
| 参数量 | ~25M |
| 训练速度 (CPU) | ~3K tok/s |
| Train Loss (3 epochs) | ~4.0 → ~3.0 |
| Val PPL (3 epochs) | ~30 → ~20 |

### 8.2 WikiText-103 (base 模型, GPU)

| 指标 | 预期值 |
|------|--------|
| 参数量 | ~105M |
| 训练速度 (A100) | ~50K tok/s |
| Train Loss (3 epochs) | ~4.5 → ~2.5 |
| Val PPL (3 epochs) | ~35 → ~15 |

### 8.3 Scaling Law 观察

用 tiny/small/base 三种配置在同一数据集上训练相同步数，记录下来 loss 随参数量的变化趋势。预期：参数量 ↑ 时，相同步数下的 loss ↓。

---

## 9. 动手练习

| # | 练习 | 难度 |
|---|------|------|
| 1 | 对比 RMSNorm vs LayerNorm 的训练速度和收敛曲线 | ⭐ |
| 2 | 实现 SwiGLU 前向传播（不用 torch），验证数值 | ⭐⭐ |
| 3 | 对比 RoPE vs NoPE vs Learnable PE 的长度外推能力 | ⭐⭐ |
| 4 | 计算 GQA 在不同分组比例下的 KV Cache 节省量 | ⭐ |
| 5 | 编写梯度累积 + AMP 的正确性验证脚本 | ⭐⭐ |
| 6 | 训练 tiny/small/base 三个模型，观察 Scaling Law | ⭐⭐⭐ |

---

## 10. 从 Stage 4 到 Stage 5

Stage 4 完成了基础预训练。Stage 5 将进入**指令微调**阶段，涵盖：

| Stage 4 (预训练) | Stage 5 (指令微调) |
|-------------------|---------------------|
| 无监督 LM 目标 | 监督微调 (SFT) |
| 全参数训练 | LoRA 高效微调 |
| — | DPO / RLHF 偏好对齐 |
| 通用语料 | 指令数据 (Alpaca, ShareGPT) |

Stage 5 的核心挑战不再是架构设计，而是：
1. 如何构造高质量指令数据集
2. 如何用少量 GPU 微调大模型 (LoRA)
3. 如何把预训练语言模型变成"懂指令、会对话"的助手

---

## 参考资源

- Zhang & Sennrich (2019) — "Root Mean Square Layer Normalization"
- Shazeer (2020) — "GLU Variants Improve Transformer"
- Su et al. (2021) — "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- Ainslie et al. (2023) — "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
- Touvron et al. (2023) — "LLaMA: Open and Efficient Foundation Language Models"
- LLaMA 3 (2024) — Meta AI
