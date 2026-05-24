"""
Multi-Head Self-Attention —— Transformer 的核心引擎

核心概念：
  - Self-Attention: 序列中的每个位置"关注"所有其他位置，
    计算加权和作为当前输出的表示
  - Multi-Head: 多组平行的 Q/K/V 投影，每组关注不同的语义关系
  - Scaled Dot-Product: 除以 sqrt(d_k) 防止点积过大导致 softmax 梯度消失

Q/K/V 的直觉理解：
  假设你在图书馆找书（Self-Attention 的一次查询）：
    Query  (Q): "我想要一本关于深度学习的书"  ← 你的查询意图
    Key    (K): 每本书的书名和简介              ← 所有书的索引
    Value  (V): 每本书的完整内容                ← 最终返回给你的信息

  计算过程：
    ① 拿你的 Q 和每本书的 K 算相似度（点积）→ 得出一组"相关性分数"
    ② 相关性分数 ÷ sqrt(d_k) → softmax → "注意力权重"（哪本书最值得看）
    ③ 注意力权重 × V → 加权求和 → 你要的综合信息

维度追踪（以 d_model=512, num_heads=8 为例）：
  Q, K, V 投影:  (B, S, 512) → (B, S, 512) → split → (B, 8, S, 64)
  注意力得分:     (B, 8, S, 64) × (B, 8, 64, S) → (B, 8, S, S)
  注意力输出:     (B, 8, S, S) × (B, 8, S, 64) → (B, 8, S, 64) → concat → (B, S, 512)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    多头注意力 —— 同时支持 Self-Attention 和 Cross-Attention

    三种使用模式：
      1. Encoder Self-Attention:  Q=K=V=encoder 输入（编码器内部自注意）
      2. Decoder Self-Attention:  Q=K=V=decoder 输入（带因果掩码）
      3. Decoder Cross-Attention: Q=decoder 输入, K=V=encoder 输出（跨注意力）
    """

    def __init__(self, d_model: int = 512, num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) 必须能被 num_heads ({num_heads}) 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度（512/8 = 64）

        # Q, K, V 的线性投影
        # 原始论文：d_model → d_model，然后 split 成 num_heads 个 d_k
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # 输出投影：把多头拼接后的结果映射回 d_model
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        把 (B, S, d_model) 拆成 (B, num_heads, S, d_k)

        为什么这样做？
          - 原始的 Q, K, V 是 d_model 维的，我们希望多个"头"关注不同方面
          - 比如 head0 关注语法关系，head1 关注语义相似度，head2 关注位置相邻性
          - "拆分"操作让不同头有独立的小维度空间来学习不同的注意力模式
        """
        B, S, _ = x.shape
        x = x.view(B, S, self.num_heads, self.d_k)  # (B, S, 8, 64)
        return x.permute(0, 2, 1, 3)                  # (B, 8, S, 64)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """把 (B, num_heads, S, d_k) 合并回 (B, S, d_model)"""
        B, _, S, _ = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, S, 8, 64)
        return x.view(B, S, self.d_model)        # (B, S, 512)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        参数：
          query: (B, S_q, d_model)  查询
          key:   (B, S_k, d_model)  键（S_k = S_q for self-attention）
          value: (B, S_k, d_model)  值
          mask:  (B, S_q, S_k) 或 (B, 1, S_k)
                 True=允许注意, False=屏蔽（设为 -inf）

        返回：
          output: (B, S_q, d_model)
        """
        B = query.size(0)

        # --- ① 线性投影 + 拆分为多头 ---
        Q = self._split_heads(self.W_q(query))  # (B, num_heads, S_q, d_k)
        K = self._split_heads(self.W_k(key))    # (B, num_heads, S_k, d_k)
        V = self._split_heads(self.W_v(value))  # (B, num_heads, S_k, d_k)

        # --- ② Scaled Dot-Product Attention ---
        # 注意力得分 = Q · K^T / sqrt(d_k)
        # Q: (B, h, S_q, d_k), K^T: (B, h, d_k, S_k) → 得分: (B, h, S_q, S_k)
        scale = math.sqrt(self.d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

        # 应用掩码（屏蔽不应关注的位置）
        if mask is not None:
            # mask 形状可能是 (B, S_q, S_k) 或 (B, 1, S_k)，需要适配
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (B, 1, S_q, S_k)
            scores = scores.masked_fill(~mask, float("-inf"))

        # Softmax → 注意力权重
        attn_weights = F.softmax(scores, dim=-1)  # (B, h, S_q, S_k)
        attn_weights = self.dropout(attn_weights)

        # 加权求和：attn_weights × V
        # attn: (B, h, S_q, S_k), V: (B, h, S_k, d_k) → (B, h, S_q, d_k)
        attn_output = torch.matmul(attn_weights, V)

        # --- ③ 合并多头 + 输出投影 ---
        output = self._merge_heads(attn_output)  # (B, S_q, d_model)
        return self.W_o(output)


# ============================================================
# 掩码生成工具
# ============================================================

def make_padding_mask(x: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    生成 padding 掩码

    返回: (B, 1, S)  True=有效位置, False=<PAD>
    用法: 广播到 (B, 1, S_q, S_k)，屏蔽 K 维度上的 <PAD>
    """
    return (x != pad_idx).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)


def make_causal_mask(sz: int, device: torch.device) -> torch.Tensor:
    """
    生成因果掩码（下三角矩阵）

    返回: (1, 1, sz, sz)
         [[True,  False, False],
          [True,  True,  False],
          [True,  True,  True ]]

    用法: 用于 Decoder Self-Attention，防止"看到未来的词"
    """
    return torch.tril(torch.ones(sz, sz, dtype=torch.bool, device=device)
                      ).unsqueeze(0).unsqueeze(0)


def make_combined_mask(src: torch.Tensor, tgt: torch.Tensor,
                       src_pad_idx: int, tgt_pad_idx: int) -> tuple:
    """
    生成 Transformer 需要的所有掩码:
      src_mask:    (B, 1, 1, src_len) — padding mask for encoder
      tgt_mask:    (1, 1, tgt_len, tgt_len) — causal mask for decoder self-attn
      mem_mask:    (B, 1, 1, src_len) — padding mask for decoder cross-attn
    """
    src_mask = make_padding_mask(src, src_pad_idx)       # (B, 1, 1, src_len)
    tgt_mask = make_causal_mask(tgt.size(1), tgt.device)  # (1, 1, tgt_len, tgt_len)
    mem_mask = make_padding_mask(src, src_pad_idx)       # (B, 1, 1, src_len)
    return src_mask, tgt_mask, mem_mask
