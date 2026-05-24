"""
Causal Self-Attention —— GPT-2 的核心计算单元

和 Encoder Self-Attention 的唯一区别：**因果掩码 (causal mask)**

因果掩码是一个下三角矩阵，确保位置 i 只能"看到"位置 0, 1, ..., i
这保证了语言模型的自回归性质——"不能偷看未来的词"

例子（seq_len=4）：
  [[ True, False, False, False ],   ← pos 0 只看自己
   [ True,  True, False, False ],   ← pos 1 看 0 和 1
   [ True,  True,  True, False ],   ← pos 2 看 0, 1, 2
   [ True,  True,  True,  True ]]   ← pos 3 看所有前面的词

和原始 Transformer Decoder 的区别：
  - GPT 没有 Cross-Attention（因为没有 Encoder）
  - GPT 使用 GELU 激活而非 ReLU
  - GPT-2 使用 Pre-Norm（LayerNorm 在子层之前）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CausalSelfAttention(nn.Module):
    """
    因果自注意力：和 MultiHeadAttention 相同，但 mask 固定为下三角

    维度追踪（B=32, T=256, C=768, H=12）：
      Q/K/V 投影:  (B, T, 768) → (B, 12, T, 64)
      注意力得分:   (B, 12, T, 64) × (B, 12, 64, T) → (B, 12, T, T)
      加因果掩码:   (B, 12, T, T) → 上三角设为 -inf
      Softmax:      (B, 12, T, T)  注意力权重（下三角 + 行归一化）
      加权求和:     (B, 12, T, T) × (B, 12, T, 64) → (B, 12, T, 64)
      concat:       (B, 12, T, 64) → (B, T, 768)
    """

    def __init__(self, n_embd: int = 768, n_head: int = 12,
                 dropout: float = 0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.d_k = n_embd // n_head

        # QKV 合并为一次投影（GPT-2 的标准优化）
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=True)
        # 输出投影
        self.c_proj = nn.Linear(n_embd, n_embd, bias=True)

        self.dropout = nn.Dropout(dropout)

        # 注册因果掩码
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(1024, 1024)).view(1, 1, 1024, 1024),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # ① QKV 投影：一次性算，然后切片
        qkv = self.c_attn(x)  # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # ② 拆分为多头
        q = q.view(B, T, self.n_head, self.d_k).permute(0, 2, 1, 3)  # (B, H, T, d_k)
        k = k.view(B, T, self.n_head, self.d_k).permute(0, 2, 1, 3)
        v = v.view(B, T, self.n_head, self.d_k).permute(0, 2, 1, 3)

        # ③ Scaled Dot-Product Attention + 因果掩码
        scale = math.sqrt(self.d_k)
        att = (q @ k.transpose(-2, -1)) / scale  # (B, H, T, T)

        # 因果掩码：上三角设为 -inf
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        # ④ 加权求和 + 合并多头 + 输出投影
        y = att @ v  # (B, H, T, d_k)
        y = y.permute(0, 2, 1, 3).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    """GPT-2 FeedForward：Linear → GELU → Linear → Dropout"""

    def __init__(self, n_embd: int = 768, dropout: float = 0.1):
        super().__init__()
        # GPT-2: 4x 膨胀
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GELU: Gaussian Error Linear Unit，比 ReLU 平滑
        # GELU(x) = x * Φ(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
        return self.c_proj(self.dropout(F.gelu(self.c_fc(x))))


class GPT2Block(nn.Module):
    """
    一个 GPT-2 Transformer Block（Pre-Norm 风格）

    结构：LN → CausalAttn → + → LN → MLP → +
    即：x = x + Attn(LN(x))  然后  x = x + MLP(LN(x))
    """

    def __init__(self, n_embd: int = 768, n_head: int = 12,
                 dropout: float = 0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
