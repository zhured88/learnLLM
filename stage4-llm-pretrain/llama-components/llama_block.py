"""
LLaMA Transformer Block — 组装 RMSNorm + GQA + RoPE + SwiGLU

与 GPT-2 Block 的逐项对比：

╔══════════════╦══════════════════╦══════════════════╗
║   组件        ║    GPT-2         ║    LLaMA         ║
╠══════════════╬══════════════════╬══════════════════╣
║ 归一化        ║ LayerNorm       ║ RMSNorm          ║
║ 激活函数      ║ GELU            ║ SiLU (SwiGLU)    ║
║ 位置编码      ║ 可学习 Embedding║ RoPE              ║
║ 注意力        ║ Multi-Head      ║ Grouped-Query     ║
║ FFN 结构      ║ Linear->GELU->  ║ SwiGLU(gate+up)  ║
║              ║ Linear          ║ -> down_proj      ║
║ Norm 位置     ║ Pre-Norm        ║ Pre-Norm          ║
╚══════════════╩══════════════════╩══════════════════╝
"""

import math
import torch
import torch.nn as nn
from rms_norm import RMSNorm
from gqa import GroupedQueryAttention
from swiglu import SwiGLU
from rope import RotaryPositionEmbedding


class LLaMABlock(nn.Module):
    """LLaMA 的一个 Decoder Block。

    结构 (Pre-Norm):
      x = x + GQA(RoPE(RMSNorm(x)))    # Self-Attention with RoPE
      x = x + SwiGLU(RMSNorm(x))       # FFN with SwiGLU
    """

    def __init__(self, dim: int, n_head: int, n_kv_head: int,
                 max_seq_len: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.n_head = n_head

        # Pre-Norm: 两个 RMSNorm
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

        # Self-Attention with GQA
        self.attention = GroupedQueryAttention(dim, n_head, n_kv_head, dropout)

        # RoPE
        d_k = dim // n_head
        self.rope = RotaryPositionEmbedding(d_k, max_seq_len)

        # FFN with SwiGLU
        self.ffn = SwiGLU(dim, dropout=dropout)

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor = None,
                offset: int = 0) -> torch.Tensor:
        """前向传播。

        Args:
            x: (B, T, dim)
            mask: (1, 1, T, T) 因果掩码
            offset: KV cache 偏移量
        """
        # Self-Attention 分支
        residual = x
        x_norm = self.attention_norm(x)
        x_attn = self.attention(x_norm, mask=mask, rope=self.rope, offset=offset)
        x = residual + x_attn

        # FFN 分支
        residual = x
        x_norm = self.ffn_norm(x)
        x_ffn = self.ffn(x_norm)
        x = residual + x_ffn

        return x


# ── 与 GPT-2 Block 对比 ──────────────────────────────────────

class GPT2Block(nn.Module):
    """从 Stage 3 复制的 GPT-2 Block，用于并列对比。"""

    def __init__(self, dim: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        from gqa import MultiHeadAttention
        self.ln_1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, n_head, dropout)
        self.ln_2 = nn.LayerNorm(dim)
        # GELU MLP
        self.fc = nn.Linear(dim, 4 * dim)
        self.proj = nn.Linear(4 * dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask=None):
        x = x + self.attn(self.ln_1(x), mask)
        x = x + self.dropout(self.proj(torch.nn.functional.gelu(self.fc(self.ln_2(x)))))
        return x


def compare_blocks():
    """并排对比 LLaMA Block 和 GPT-2 Block。"""
    dim, n_head, n_kv_head = 512, 8, 4

    llama_block = LLaMABlock(dim, n_head, n_kv_head)
    gpt2_block = GPT2Block(dim, n_head)

    llama_params = sum(p.numel() for p in llama_block.parameters())
    gpt2_params = sum(p.numel() for p in gpt2_block.parameters())

    print(f"dim={dim}, n_head={n_head}, n_kv_head={n_kv_head}")
    print(f"  LLaMA Block params:  {llama_params:,}")
    print(f"  GPT-2 Block params:  {gpt2_params:,}")

    # 验证前向传播
    x = torch.randn(2, 64, dim)
    mask = torch.tril(torch.ones(1, 1, 64, 64))
    y_llama = llama_block(x, mask)
    y_gpt2 = gpt2_block(x, mask)

    print(f"  LLaMA output: {y_llama.shape}, mean={y_llama.mean():.4f}, std={y_llama.std():.4f}")
    print(f"  GPT-2 output: {y_gpt2.shape}, mean={y_gpt2.mean():.4f}, std={y_gpt2.std():.4f}")


if __name__ == "__main__":
    print("=" * 50)
    print("LLaMA Block vs GPT-2 Block")
    print("=" * 50)
    compare_blocks()
