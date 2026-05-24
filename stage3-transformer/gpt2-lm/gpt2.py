"""
GPT-2 语言模型 —— Decoder-Only Transformer

架构对比：

  Transformer (Encoder-Decoder):        GPT-2 (Decoder-Only):
    Encoder (双向 Self-Attention)          ↓ 删掉
    Decoder (因果 Self-Attention)          GPT2Block × N
    Decoder Cross-Attention                ↓ 删掉（没有 Encoder 做 Cross-Attention）
    Output Linear                          Output Linear

GPT-2 的核心简化：
  1. 去掉 Encoder（不需要双向编码，因为不是翻译任务）
  2. 去掉 Cross-Attention（没有 Encoder 输出可以关注）
  3. Pre-Norm（LayerNorm 在子层之前，而非之后）
  4. GELU 激活（而非 ReLU）
  5. 位置编码改为可学习的（而非 sinusoidal）

模型规模（GPT-2 系列）：
  GPT-2 Small:  12 layers, 768 dim, 12 heads,  124M params
  GPT-2 Medium: 24 layers, 1024 dim, 16 heads, 355M params
  GPT-2 Large:  36 layers, 1280 dim, 20 heads, 774M params
  GPT-2 XL:     48 layers, 1600 dim, 25 heads, 1.5B params
"""

import torch
import torch.nn as nn
from attention import GPT2Block


class GPT2(nn.Module):
    """GPT-2 风格的语言模型"""

    def __init__(
        self,
        vocab_size: int,
        n_embd: int = 768,         # 隐藏维度
        n_layer: int = 12,         # Transformer 块数
        n_head: int = 12,          # 注意力头数
        max_seq_len: int = 1024,   # 最大序列长度
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()

        # Token Embedding
        self.wte = nn.Embedding(vocab_size, n_embd, padding_idx=pad_idx)

        # Position Embedding（可学习，而非 sinusoidal）
        self.wpe = nn.Embedding(max_seq_len, n_embd)

        self.drop = nn.Dropout(dropout)

        # Stack of GPT2Blocks
        self.blocks = nn.ModuleList([
            GPT2Block(n_embd, n_head, dropout)
            for _ in range(n_layer)
        ])

        # 最终 LayerNorm
        self.ln_f = nn.LayerNorm(n_embd)

        # 输出头（预测下一个 token）
        # 原始 GPT-2 使用 weight tying：wte.weight 和 lm_head.weight 共享
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # 初始化
        self.apply(self._init_weights)

        # Weight tying
        self.lm_head.weight = self.wte.weight

        # 因果掩码
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len))
            .view(1, 1, max_seq_len, max_seq_len),
        )

        self.n_embd = n_embd
        self.pad_idx = pad_idx

    def _init_weights(self, module):
        """GPT-2 风格的初始化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,         # (B, T) token 索引
        targets: torch.Tensor = None,  # (B, T) 训练时传入的标签
    ) -> tuple:
        """
        返回：
          logits: (B, T, V)
          loss: 标量（仅当 target 不为 None）
        """
        B, T = idx.shape
        assert T <= self.mask.size(-1), f"序列长度 {T} 超过最大 {self.mask.size(-1)}"

        # --- Token + Position Embedding ---
        # pos: [0, 1, 2, ..., T-1]
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        tok_emb = self.wte(idx)  # (B, T, C)
        pos_emb = self.wpe(pos)  # (1, T, C)
        x = self.drop(tok_emb + pos_emb)

        # --- 通过所有 Transformer Block ---
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        # --- 输出 ---
        logits = self.lm_head(x)  # (B, T, V)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.pad_idx,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,            # (B, T0) 起始 token 序列
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        自回归文本生成

        参数：
          idx:              (B, T0) 起始序列（如 <SOS> 或一段 prompt）
          max_new_tokens:   最多生成多少新 token
          temperature:      温度（<1 保守, >1 冒险）
          top_k:            top-k 采样（只从概率最高的 k 个 token 中抽）

        返回：
          (B, T0 + max_new_tokens) 完整序列
        """
        self.eval()

        for _ in range(max_new_tokens):
            # 如果序列太长，截取最后 max_seq_len 个 token（context window）
            idx_cond = idx[:, -self.mask.size(-1):]

            logits, _ = self(idx_cond)
            # 取最后一个位置的 logits
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            # Top-k 过滤：把非 top-k 的 logits 设为 -inf
            if top_k > 0:
                top_k = min(top_k, logits.size(-1))
                thresh = logits.topk(top_k, dim=-1).values[:, -1:]
                logits[logits < thresh] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)  # (B, 1)

            idx = torch.cat([idx, next_token], dim=1)

        return idx
