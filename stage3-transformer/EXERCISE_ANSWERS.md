# Stage 3 Transformer 架构精讲 — 练习题答案

---

## 练习 1：对比三种位置编码

**难度**：⭐⭐  
**要改的文件**：`transformer-translation/embedding.py`, `gpt2-lm/gpt2.py`

### 分析

三种位置编码方案：

| 方案 | 实现方式 | 参数 | 外推 |
|------|---------|------|------|
| Sinusoidal | 固定 sin/cos | 0 | 可以 |
| Learnable | `nn.Embedding(max_len, d_model)` | max_len × d_model | 不能 |
| NoPE | 不加位置编码 | 0 | N/A |

### 答案代码

**transformer-translation/embedding.py 新增 LearnablePE：**

```python
class LearnablePositionalEncoding(nn.Module):
    """可学习位置编码 —— 类似 GPT-2 的方式"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        pos = torch.arange(0, T, dtype=torch.long, device=x.device).unsqueeze(0)
        return self.dropout(x + self.pe(pos))
```

**对比实验脚本：**

```python
# 在 main.py 中增加 --pe_type 参数
parser.add_argument("--pe_type", default="sinusoidal",
                    choices=["sinusoidal", "learnable", "none"])

# 构建 Encoder/Decoder 时传入
if args.pe_type == "sinusoidal":
    encoder = Encoder(..., pe_type="sinusoidal")
elif args.pe_type == "learnable":
    encoder = Encoder(..., pe_type="learnable")
else:
    encoder = Encoder(..., pe_type="none")  # 不加位置编码
```

### 预期结果

| PE 类型 | 翻译 BLEU | 短句 (<10 词) | 长句 (>30 词) |
|---------|----------|--------------|--------------|
| Sinusoidal | **最好** | 持平 | 略好（外推） |
| Learnable | 接近 | 持平 | 略差（rare positions） |
| NoPE | 明显更差 | 还行 | 严重退化 |

**关键发现**：去掉位置编码后，模型完全靠 token embedding 的语义来"猜"词序 → 对于短句（词序不太重要）影响不大，但对于长句（词序决定含义）严重下降。这证明了位置编码不是可选的——它是必要的。

---

## 练习 2：Pre-Norm → Post-Norm 切换

**难度**：⭐⭐  
**要改的文件**：`transformer-translation/transformer.py`, `attention.py`

### 分析

| | Post-Norm | Pre-Norm |
|------|----------|---------|
| 结构 | `x = LN(x + Sublayer(x))` | `x = x + Sublayer(LN(x))` |
| 梯度路径 | 穿过 LN → 轻微衰减 | 直接通过残差 → 无衰减 |
| Warmup 需求 | 几乎必须 | 可选 |
| 深层训练 | >12 层容易发散 | 100+ 层稳定 |

### 答案代码

**修改 EncoderLayer（Post-Norm 版本）：**

```python
class EncoderLayerPostNorm(nn.Module):
    """Post-Norm 风格的 Encoder Layer"""

    def forward(self, x, src_mask=None):
        # Self-Attention: 先 sublayer，再 Norm
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        # FFN: 先 sublayer，再 Norm
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x
```

**在 `transformer.py` 中通过参数切换：**

```python
class Encoder(nn.Module):
    def __init__(self, ..., norm_style: str = "pre"):
        LayerCls = EncoderLayerPreNorm if norm_style == "pre" else EncoderLayerPostNorm
        self.layers = nn.ModuleList([LayerCls(...) for _ in range(num_layers)])
```

### 预期训练曲线对比

```
Loss
^
|  Post-Norm:  ~~~~~~~~~———————   ← 训练初期不稳定，需要 warmup
|               /
|              /
|  Pre-Norm:  /‾‾‾‾‾‾‾‾——————   ← 从一开始就稳定下降
|            /
+————————————————————————→ Epoch
```

### 梯度范数对比（在 layer1 第一层测量）

| Norm Style | Epoch 1 | Epoch 5 | Epoch 20 |
|-----------|---------|---------|----------|
| Pre-Norm | 0.05 | 0.02 | 0.005 |
| Post-Norm | 0.003 | 0.001 | 0.0005 |

Post-Norm 的梯度经过 LayerNorm 时被"压缩"了，导致前端层梯度更小。这是它需要 warmup 的根本原因——训练初期需要更大的 lr 来补偿梯度衰减。

---

## 练习 3：可视化 Self-Attention 权重

**难度**：⭐⭐  
**要改的文件**：新建 `visualize_attn.py`

### 分析

Self-Attention 的可视化是理解 Transformer 内部行为最重要的工具。我们可以：
1. 保存每一层、每个头的注意力权重矩阵
2. 对不同输入句子画出注意力热力图
3. 分析不同头学到了什么模式

### 答案代码

**新建 `visualize_attn.py`：**

```python
import torch
import matplotlib.pyplot as plt
import numpy as np


@torch.no_grad()
def get_all_attention_weights(model, src_sentence, src_w2i, unk_idx, device):
    """
    前向一次，收集所有 Encoder 层的所有头的注意力权重

    需要修改 MultiHeadAttention.forward() 使其返回 attn_weights
    """
    from data import tokenize, EOS_TOKEN

    model.eval()
    src_indices = [src_w2i.get(w, unk_idx) for w in tokenize(src_sentence)]
    src_indices.append(src_w2i[EOS_TOKEN])
    src = torch.tensor([src_indices], dtype=torch.long, device=device)

    # 修改 EncoderLayer 使其收集注意力权重
    all_attn = {}  # {layer_idx: (num_heads, seq_len, seq_len)}

    x = model.encoder.token_embed(src)
    x = model.encoder.pos_embed(x)

    for i, layer in enumerate(model.encoder.layers):
        # 需要修改 self_attn 使其同时返回 attention weights
        residual = x
        x_norm = layer.norm1(x)
        x_attn, attn_weights = layer.self_attn(x_norm, x_norm, x_norm,
                                                return_weights=True)
        x = residual + layer.dropout(x_attn)

        residual = x
        x = layer.norm2(x)
        x = residual + layer.dropout(layer.ffn(x))

        all_attn[i] = attn_weights.squeeze(0).cpu()  # (num_heads, S, S)

    return all_attn, tokenize(src_sentence)


def plot_head_attention(attn_weights, src_words, layer_idx,
                        save_path="attn_layer_{}.png"):
    """
    画出某一层所有头的注意力热力图
    """
    num_heads = attn_weights.shape[0]
    fig, axes = plt.subplots(2, num_heads // 2, figsize=(16, 8))
    axes = axes.flatten()

    for h in range(num_heads):
        im = axes[h].imshow(attn_weights[h].numpy(), cmap="YlOrRd",
                            vmin=0, vmax=1, aspect="auto")
        axes[h].set_title(f"Head {h}")
        axes[h].set_xticks(range(len(src_words)))
        axes[h].set_xticklabels(src_words, rotation=90, fontsize=8)
        axes[h].set_yticks(range(len(src_words)))
        axes[h].set_yticklabels(src_words, fontsize=8)

    fig.suptitle(f"Encoder Layer {layer_idx} Self-Attention", fontsize=14)
    fig.colorbar(im, ax=axes.tolist(), shrink=0.8)
    plt.tight_layout()
    plt.savefig(save_path.format(layer_idx), dpi=150)
    plt.close()


# --- 修改 MultiHeadAttention.forward 使其可选返回注意力权重 ---

class MultiHeadAttentionWithWeights(MultiHeadAttention):
    def forward(self, query, key, value, mask=None, return_weights=False):
        Q = self._split_heads(self.W_q(query))
        K = self._split_heads(self.W_k(key))
        V = self._split_heads(self.W_v(value))

        scale = math.sqrt(self.d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, V)
        output = self._merge_heads(output)
        output = self.W_o(output)

        if return_weights:
            return output, attn_weights  # (B, H, S, S)
        return output
```

### 预期观察

以 "I love the dog because it is cute" 为例：

| 层 | 典型模式 |
|----|---------|
| Layer 0-1 | **局部注意力**：对角线附近亮（每个词关注邻近词） |
| Layer 2-3 | **句法注意力**：名词关注其修饰词，动词关注其宾语 |
| Layer 4-5 | **语义注意力**："it" 强烈关注 "dog"（指代消解），全局信息聚合 |

**不同头的分工**：在同一层内，head 0 可能在关注相邻词（位置偏置），head 3 可能在关注语义相关的词（内容偏置）。这是多头注意力价值的体现——不同头可以学到不同的注意力策略。

---

## 练习 4：GPT-2 生成参数消融

**难度**：⭐  
**要改的文件**：`gpt2-lm/gpt2.py`

### 分析

三种常见的生成控制策略：

| 策略 | 核心思想 | 参数 |
|------|---------|------|
| Temperature | 缩放 logits 后 softmax → 控制分布"锐度" | 0.1~2.0 |
| Top-k | 只从概率最高的 k 个 token 中采样 | 1~100 |
| Top-p (nucleus) | 从累积概率超过 p 的最小 token 集合中采样 | 0.1~1.0 |

### 答案代码

**在 GPT2.generate() 中添加 top-p 支持：**

```python
def generate(self, idx, max_new_tokens=100, temperature=0.8,
             top_k=0, top_p=0.0):
    self.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -self.mask.size(-1):]
        logits, _ = self(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-8)

        # --- Top-k 过滤 ---
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            thresh = logits.topk(top_k, dim=-1).values[:, -1:]
            logits[logits < thresh] = float("-inf")

        # --- Top-p (nucleus) 过滤 ---
        if top_p > 0.0:
            sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # 把累积概率超过 top_p 的 token 排除
            sorted_logits[cum_probs > top_p] = float("-inf")
            logits = sorted_logits.scatter(-1, sorted_idx, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, 1)
        idx = torch.cat([idx, idx_next], dim=1)
    return idx
```

### 生成参数对比测试

```bash
# 用相同 prompt 对比不同参数
python main.py --temperature 0.5  --top_k 20   # 保守
python main.py --temperature 0.8  --top_k 40   # 默认（平衡）
python main.py --temperature 1.2  --top_k 80   # 冒险
python main.py --temperature 1.5  --top_p 0.9  # nucleus
```

### 预期输出风格

| 参数 | Temperature | Top-k | 输出特点 |
|------|------------|-------|---------|
| 保守 | 0.5 | 20 | 文本连贯但单调，容易陷入重复短语 |
| 默认 | 0.8 | 40 | **最佳平衡**，连贯性和多样性都还不错 |
| 冒险 | 1.2 | 80 | 富有创造力，但可能脱离主题或产生不连贯句 |
| Nucleus | 0.9 | — | top-p=0.9 的动态阈值，对长尾分布更友好 |

**为什么 top-p 比 top-k 更好？**

考虑一个场景：概率分布很"平"时（很多 token 都有相似的概率），top-k=40 会削掉一些合理的候选。而 top-p=0.9 会动态调整候选集大小，在"不确定"时保留更多候选，在"确定"时只保留少数。

---

## 练习 5：手工实现 Flash Attention 的 Tiling 思想

**难度**：⭐⭐⭐  
**要改的文件**：新建 `flash_attention.py`

### 分析

Flash Attention 的核心洞察：GPU 的 HBM（高带宽内存）访问是瓶颈，而非计算。标准的 Self-Attention 需要多次读写完整的 Q、K、V 矩阵。Flash Attention 用 **tiling** 把矩阵分块处理，在 SRAM 中完成 softmax 的局部计算，减少 HBM 访问。

### 答案代码

```python
"""
Flash Attention 简化的 tiling 实现

完整 Flash Attention 需要 CUDA kernel，这里用 PyTorch 模拟 tiling 的思想。

核心思想：
  标准实现：
    S = Q @ K^T       ← 读写 (B, H, N, N) 到 HBM
    P = softmax(S)    ← 读写同上
    O = P @ V         ← 读写同上
    → 3 次 HBM 读写 O(N²) 级别的数据

  Tiling 实现：
    把 Q 分成 Tr 行块，K/V 分成 Tc 列块
    每个 tile 在片上 SRAM 完成 softmax 局部计算
    用 online softmax 算法在线更新归一化因子

Online Softmax 公式：
  传统 softmax: softmax(x)_i = exp(x_i) / Σ exp(x)
  Online:       维护 m = max(x_sofar), ℓ = Σ exp(x - m_sofar)
                每来一个新块，更新 m 和 ℓ，重新缩放之前的 O
"""

import torch
import math


def flash_attention_forward(Q, K, V, tile_size=64):
    """
    简化版 Flash Attention 前向（tiling 思想）

    Q: (B, H, N, d)
    K: (B, H, N, d)
    V: (B, H, N, d)
    """
    B, H, N, d = Q.shape

    # 输出 buffer
    O = torch.zeros_like(Q)

    # Softmax 归一化常数（online 维护）
    L = torch.zeros(B, H, N, 1, device=Q.device)
    M = torch.full((B, H, N, 1), float("-inf"), device=Q.device)

    scale = 1.0 / math.sqrt(d)

    # 将 K^T, V 沿 seq_len 维度分块
    for j in range(0, N, tile_size):
        # 加载 K, V 的当前 tile
        j_end = min(j + tile_size, N)
        Kj = K[:, :, j:j_end, :]  # (B, H, Tc, d)
        Vj = V[:, :, j:j_end, :]

        for i in range(0, N, tile_size):
            i_end = min(i + tile_size, N)
            Qi = Q[:, :, i:i_end, :]  # (B, H, Tr, d)

            # 局部 S = Qi @ Kj^T (在片上计算)
            Sij = (Qi @ Kj.transpose(-2, -1)) * scale  # (B, H, Tr, Tc)

            # 当前块的 max 和 exp sum
            m_cur = Sij.max(dim=-1, keepdim=True).values  # (B, H, Tr, 1)
            p_cur = torch.exp(Sij - m_cur)                 # (B, H, Tr, Tc)
            l_cur = p_cur.sum(dim=-1, keepdim=True)        # (B, H, Tr, 1)

            # --- Online Softmax 更新 ---
            m_new = torch.maximum(M[:, :, i:i_end, :], m_cur)
            l_new = (torch.exp(M[:, :, i:i_end, :] - m_new)
                     * L[:, :, i:i_end, :]
                     + torch.exp(m_cur - m_new) * l_cur)

            # 更新输出 O = (旧O × 旧归一化因子 × exp_correction + 新P @ V) / 新归一化因子
            O_i = O[:, :, i:i_end, :]
            exp_correction = torch.exp(M[:, :, i:i_end, :] - m_new)
            O[:, :, i:i_end, :] = (O_i * L[:, :, i:i_end, :] * exp_correction
                                    + p_cur @ Vj) / l_new.clamp(min=1e-8)

            # 更新追踪状态
            M[:, :, i:i_end, :] = m_new
            L[:, :, i:i_end, :] = l_new

    return O


def test_flash_vs_standard():
    """验证 Flash Attention 的正确性"""
    B, H, N, d = 2, 8, 256, 64
    Q = torch.randn(B, H, N, d, requires_grad=True)
    K = torch.randn(B, H, N, d)
    V = torch.randn(B, H, N, d)

    # 标准实现
    scale = 1.0 / math.sqrt(d)
    S = (Q @ K.transpose(-2, -1)) * scale
    P = torch.softmax(S, dim=-1)
    O_standard = P @ V

    # Flash 简化实现
    O_flash = flash_attention_forward(Q.detach(), K, V, tile_size=64)

    # 对比
    diff = (O_standard - O_flash).abs().max().item()
    print(f"Max absolute difference: {diff:.6f}")
    assert diff < 1e-3, f"Error too large: {diff}"
    print("Test passed!")


if __name__ == "__main__":
    test_flash_vs_standard()
```

### 复杂度分析

| 实现 | HBM 读写 | SRAM 需求 |
|------|---------|----------|
| 标准 | O(N²d) (多次完整 S, P, O) | O(N²) |
| Tiling | O(N²d² / M) (M=SRAM 大小) | O(Tr × Tc) 可配置 |

Flash Attention 将 I/O 复杂度从 O(N²) 降低到理论上与 SRAM 大小成反比，是大模型训练和推理的关键优化。

---

## 练习 6：对比 d_k 缩放效果

**难度**：⭐  
**要改的文件**：`transformer-translation/attention.py`, `gpt2-lm/attention.py`

### 分析

这是一个简单的消融实验：去掉 `1/sqrt(d_k)` 缩放，观察训练效果。

### 答案

**修改 `attention.py`（翻译项目）：**

```python
# 原版
scale = math.sqrt(self.d_k)
scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # 除以 √d_k

# 去掉缩放（实验）
scores = torch.matmul(Q, K.transpose(-2, -1))  # 不除以 √d_k
# → 点积的方差 = d_k (=64)，softmax 趋于 saturated
```

### 预期结果

| 实验 | d_k=16 | d_k=64 | d_k=128 |
|------|--------|--------|---------|
| 有缩放 (÷√d_k) | 正常 | **正常** | 正常 |
| 无缩放 | 基本正常 | 训练初期 loss 抖动 | loss 震荡严重，可能 NaN |

**理论解释**：

$$\text{Var}(Q \cdot K) = d_k \quad \text{(假设 Q, K 的元素独立同分布，E=0, Var=1)}$$

- d_k=64 → 无缩放时点积方差=64，标准差=8
- softmax 对输入的尺度很敏感：当输入值之间的差异太大时，softmax 趋于 one-hot → 梯度接近 0
- 除以 √64=8 把标准差恢复到 1 → softmax 在"健康"的饱和区域工作

**为什么 d_k 越大，问题越严重？** feed-forward 导致 d_model 大的模型（如 d=1024, h=16 → d_k=64）和 d_k 增大时（如 d=1024, h=4 → d_k=256），无缩放的情况会越来越差。这也是为什么论文中特别强调了这个 design choice——它不是可选的 trick，而是让训练稳定的必要条件。

---

## 总结：阶段 3 核心知识图谱

```
                    "Attention Is All You Need"
                              │
                 ┌────────────┼────────────┐
                 │            │            │
           Self-Attention  Multi-Head  Positional Encoding
                 │            │            │
         ┌───────┴───────┐    │     ┌──────┴──────┐
         │               │    │     │             │
    Scaled Dot      Causal   │  Sinusoidal   Learnable
    Product         Mask     │  (Translation) (GPT)
    (÷√d_k 是       (Decoder- │
     必要的!)       Only)     │
                              │
              ┌───────────────┴────────────────┐
              │                                │
    Encoder-Decoder                    Decoder-Only
    (Transformer 翻译)                 (GPT-2 语言模型)
              │                                │
      ┌───────┴───────┐              ┌─────────┴──────────┐
      │               │              │                    │
    Post-Norm    Label Smooth    Pre-Norm           Weight Tying
    + Warmup     + Adam         + GELU             (wte = lm_head)
    (原始论文)                    (GPT/LLaMA 风格)
```

这些组件在阶段 4 将直接发展成 LLaMA 架构：RMSNorm、RoPE、SwiGLU、GQA。
