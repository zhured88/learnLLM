"""
GQA: Grouped Query Attention 分组查询注意力

LLaMA 用 GQA 替代标准 Multi-Head Attention：
  - MHA (Multi-Head):   n_query_heads = n_kv_heads（Q/K/V 头数相同）
  - MQA (Multi-Query):  n_kv_heads = 1（所有 Q 头共享一组 K/V）
  - GQA (Grouped-Query): 1 < n_kv_heads < n_query_heads（分组共享）

  GQA 的好处：推理时 KV Cache 减少 n_query_heads/n_kv_heads 倍，
  同时保持比 MQA 更好的质量。

用法示例:
  # LLaMA 3 8B: n_head=32, n_kv_head=8 (ratio=4)
  attn = GroupedQueryAttention(dim=4096, n_head=32, n_kv_head=8)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedQueryAttention(nn.Module):
    """GQA: 多组 Q 头共享 K/V 头。

    Args:
        dim: 模型维度 d_model
        n_head: Q 头数
        n_kv_head: K/V 头数（必须整除 n_head）
        dropout: attention dropout
    """

    def __init__(self, dim: int, n_head: int, n_kv_head: int,
                 dropout: float = 0.0):
        super().__init__()
        assert n_head % n_kv_head == 0, f"n_head({n_head}) 必须整除 n_kv_head({n_kv_head})"
        self.dim = dim
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_rep = n_head // n_kv_head       # 每组 KV 头对应几个 Q 头
        self.d_k = dim // n_head                 # 每个头的维度

        # Q 投影：n_head * d_k
        self.q_proj = nn.Linear(dim, n_head * self.d_k, bias=False)
        # K/V 投影：n_kv_head * d_k（只输出 n_kv_head 个头）
        self.k_proj = nn.Linear(dim, n_kv_head * self.d_k, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_head * self.d_k, bias=False)
        # 输出投影
        self.o_proj = nn.Linear(n_head * self.d_k, dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor = None,
                rope: callable = None,
                offset: int = 0) -> torch.Tensor:
        """
        Args:
            x: (B, T, dim)
            mask: (1, 1, T, T) 因果掩码
            rope: RoPE 函数 (q, k, T, offset) -> q_rot, k_rot
            offset: KV cache 偏移
        """
        B, T, C = x.shape

        # Q/K/V 投影
        q = self.q_proj(x).view(B, T, self.n_head, self.d_k).transpose(1, 2)    # (B, n_head, T, d_k)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.d_k).transpose(1, 2) # (B, n_kv_head, T, d_k)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.d_k).transpose(1, 2)

        # RoPE（只对 Q 和 K）
        if rope is not None:
            q, k = rope(q, k, seq_len=T, offset=offset)

        # 复制 K/V 头以匹配 Q 头数: (B, n_kv_head, T, d_k) -> (B, n_head, T, d_k)
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        # Scaled Dot-Product Attention
        scale = 1.0 / math.sqrt(self.d_k)
        attn_scores = (q @ k.transpose(-2, -1)) * scale  # (B, n_head, T, T)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask[:, :, :T, :T] == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = attn_weights @ v  # (B, n_head, T, d_k)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, n_head*d_k)
        return self.o_proj(out)


# ── 对比分析 ──────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    """标准 Multi-Head Attention（GQA 在 n_kv_head=n_head 时的特例）。"""

    def __init__(self, dim: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        self.n_head = n_head
        self.d_k = dim // n_head
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask=None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.d_k).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.d_k)
        scores = (q @ k.transpose(-2, -1)) * scale
        if mask is not None:
            scores = scores.masked_fill(mask[:, :, :T, :T] == 0, float('-inf'))
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


def compare_gqa_mha():
    """对比 GQA 和 MHA 的参数量和 KV Cache 大小。"""
    dim = 768
    n_head = 12
    n_kv_head = 4  # GQA ratio = 3

    gqa = GroupedQueryAttention(dim, n_head, n_kv_head)
    mha = MultiHeadAttention(dim, n_head)

    gqa_params = sum(p.numel() for p in gqa.parameters())
    mha_params = sum(p.numel() for p in mha.parameters())

    # KV Cache 计算（每层、每个 token）
    kv_cache_mha = 2 * n_head * (dim // n_head)  # 2 * n_head * d_k
    kv_cache_gqa = 2 * n_kv_head * (dim // n_head)

    print(f"dim={dim}, n_head={n_head}, n_kv_head={n_kv_head}")
    print(f"  GQA 参数量:   {gqa_params:,}")
    print(f"  MHA 参数量:   {mha_params:,}")
    print(f"  参数节省:     {(1 - gqa_params/mha_params)*100:.1f}%")
    print(f"  KV Cache/token (MHA): {kv_cache_mha} floats = {kv_cache_mha * 2} bytes (fp16)")
    print(f"  KV Cache/token (GQA): {kv_cache_gqa} floats = {kv_cache_gqa * 2} bytes (fp16)")
    print(f"  KV Cache 节省: {(1 - kv_cache_gqa/kv_cache_mha)*100:.0f}%")

    # 验证输出 shape 一致
    x = torch.randn(2, 16, dim)
    mask = torch.tril(torch.ones(1, 1, 16, 16))
    y_gqa = gqa(x, mask=mask)
    y_mha = mha(x, mask=mask)
    print(f"  GQA output shape: {y_gqa.shape}")
    print(f"  MHA output shape: {y_mha.shape}")


if __name__ == "__main__":
    print("=" * 50)
    print("GQA vs MHA 对比")
    print("=" * 50)
    compare_gqa_mha()
