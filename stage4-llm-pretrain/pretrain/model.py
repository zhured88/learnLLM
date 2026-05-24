"""
LLaMA 完整模型 — 集成 RMSNorm + SwiGLU + RoPE + GQA

从 Stage 3 GPT-2 演进到 Stage 4 LLaMA:

  GPT-2                    LLaMA
  ─────                    ─────
  LayerNorm          →     RMSNorm          (更快，省均值计算)
  GELU MLP           →     SwiGLU           (更强表达能力)
  可学习位置编码       →     RoPE             (相对位置、外推更好)
  Multi-Head Attn    →     GQA              (省 KV Cache)

模型规模预设:
  tiny:   25M params  (n_embd=384, n_layer=8,  n_head=6,  n_kv_head=2)
  small:  55M params  (n_embd=576, n_layer=12, n_head=9,  n_kv_head=3)
  base:  105M params  (n_embd=768, n_layer=12, n_head=12, n_kv_head=4)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 从 llama-components 导入
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "llama-components"))

from rms_norm import RMSNorm
from gqa import GroupedQueryAttention
from swiglu import SwiGLU
from rope import RotaryPositionEmbedding


class LLaMABlock(nn.Module):
    """LLaMA Decoder Block (Pre-Norm 架构)。

    x = x + GQA(RoPE(RMSNorm(x)))
    x = x + SwiGLU(RMSNorm(x))
    """

    def __init__(self, dim: int, n_head: int, n_kv_head: int,
                 max_seq_len: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.attention_norm = RMSNorm(dim)
        self.attention = GroupedQueryAttention(dim, n_head, n_kv_head, dropout)
        d_k = dim // n_head
        self.rope = RotaryPositionEmbedding(d_k, max_seq_len)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None,
                offset: int = 0) -> torch.Tensor:
        # Attention + RoPE
        residual = x
        x_norm = self.attention_norm(x)
        x_attn = self.attention(x_norm, mask=mask, rope=self.rope, offset=offset)
        x = residual + x_attn
        # FFN (SwiGLU)
        residual = x
        x_ffn = self.ffn(self.ffn_norm(x))
        return residual + x_ffn


class LLaMA(nn.Module):
    """LLaMA 完整语言模型。

    结构:
      Embedding -> [LLaMABlock x n_layer] -> RMSNorm -> lm_head
    """

    def __init__(self, vocab_size: int,
                 n_embd: int = 768,
                 n_layer: int = 12,
                 n_head: int = 12,
                 n_kv_head: int = 4,
                 max_seq_len: int = 2048,
                 dropout: float = 0.0,
                 pad_idx: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.pad_idx = pad_idx

        # Token Embedding (无位置 embedding，由 RoPE 处理位置)
        self.tok_embeddings = nn.Embedding(vocab_size, n_embd, padding_idx=pad_idx)

        # Transformer Blocks
        self.layers = nn.ModuleList([
            LLaMABlock(n_embd, n_head, n_kv_head, max_seq_len, dropout)
            for _ in range(n_layer)
        ])

        # Final Norm + LM Head
        self.norm = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # 因果掩码 (下三角)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        self.register_buffer("mask", mask)

        # Weight tying: lm_head 权重与 token embedding 共享
        self.lm_head.weight = self.tok_embeddings.weight

        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """前向传播。

        Args:
            idx: (B, T) token id 序列
            targets: (B, T) 目标 token id（训练时）

        Returns:
            logits: (B, T, vocab_size)
            loss: scalar (如果提供了 targets)
        """
        B, T = idx.shape

        # Token Embedding (仅 token，无 position)
        x = self.tok_embeddings(idx)  # (B, T, C)

        # 通过所有 Transformer Block
        for layer in self.layers:
            x = layer(x, mask=self.mask[:, :, :T, :T])

        # Final Norm + LM Head
        x = self.norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=self.pad_idx
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 100,
                 temperature: float = 0.8, top_k: int = 50,
                 top_p: float = 0.9, repetition_penalty: float = 1.0):
        """自回归文本生成。

        Args:
            idx: (B, T) 初始 token 序列
            max_new_tokens: 最多生成的 token 数
            temperature: 温度（<1 更确定性，>1 更随机）
            top_k: top-k 采样（0 表示不限制）
            top_p: nucleus sampling（0 表示不限制）
            repetition_penalty: 重复惩罚（>1 惩罚重复）

        Returns:
            (B, T + max_new_tokens) 完整序列
        """
        for _ in range(max_new_tokens):
            # 截断到 max_seq_len
            idx_cond = idx if idx.shape[1] <= self.mask.shape[-1] else idx[:, -self.mask.shape[-1]:]
            T_cond = idx_cond.shape[1]

            # 前向传播，取最后位置
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :]  # (B, vocab_size)

            # Repetition penalty
            if repetition_penalty != 1.0:
                for b in range(idx.shape[0]):
                    for token_id in set(idx[b].tolist()):
                        if logits[b, token_id] > 0:
                            logits[b, token_id] /= repetition_penalty
                        else:
                            logits[b, token_id] *= repetition_penalty

            # Temperature
            logits = logits / max(temperature, 1e-8)

            # Top-k
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float('-inf')

            # Top-p (nucleus)
            if top_p > 0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                for b in range(logits.shape[0]):
                    indices_to_remove = sorted_indices[b][sorted_indices_to_remove[b]]
                    logits[b, indices_to_remove] = float('-inf')

            # Softmax + 采样
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # 追加
            idx = torch.cat([idx, idx_next], dim=1)

        return idx

    def count_parameters(self) -> dict:
        """统计模型参数量。"""
        total = sum(p.numel() for p in self.parameters())
        embed = self.tok_embeddings.weight.numel()
        non_embed = total - embed
        return {
            "total": total,
            "embedding": embed,
            "non_embedding": non_embed
        }


# ── 预置模型配置 ────────────────────────────────────────────────

MODEL_CONFIGS = {
    "tiny": {
        "n_embd": 384, "n_layer": 8, "n_head": 6, "n_kv_head": 2
    },
    "small": {
        "n_embd": 576, "n_layer": 12, "n_head": 9, "n_kv_head": 3
    },
    "base": {
        "n_embd": 768, "n_layer": 12, "n_head": 12, "n_kv_head": 4
    },
}


def create_model(model_size: str = "tiny", vocab_size: int = 32000,
                 max_seq_len: int = 2048, dropout: float = 0.0) -> LLaMA:
    """根据预置配置创建模型。"""
    assert model_size in MODEL_CONFIGS, f"未知配置: {model_size}，可选: {list(MODEL_CONFIGS.keys())}"
    cfg = MODEL_CONFIGS[model_size]
    return LLaMA(
        vocab_size=vocab_size,
        n_embd=cfg["n_embd"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_kv_head=cfg["n_kv_head"],
        max_seq_len=max_seq_len,
        dropout=dropout,
    )
