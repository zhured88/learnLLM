"""
Transformer 编码器 + 解码器 + 完整模型

架构（原始 "Attention Is All You Need" 论文）：

  Encoder:                           Decoder:
    输入 Embedding + PE                 输出 Embedding + PE
      ↓                                   ↓
    EncoderLayer × N                    DecoderLayer × N
      │  ├── Self-Attention               │  ├── Self-Attention (causal mask)
      │  ├── Add & Norm                   │  ├── Add & Norm
      │  ├── FeedForward                  │  ├── Cross-Attention (← encoder output)
      │  └── Add & Norm                   │  ├── Add & Norm
      ↓                                   │  ├── FeedForward
    输出给 Decoder 做 Cross-Attention     │  └── Add & Norm
                                          ↓
                                        Linear → Softmax → 预测 token

Pre-Norm vs Post-Norm:
  - Post-Norm (原始论文): x + Sublayer(Norm(x)) — Norm 在残差之后
  - Pre-Norm  (GPT/LLaMA): x + Norm(Sublayer(x)) — Norm 在残差之前
  本项目使用 Pre-Norm，因为它训练更稳定，梯度流更顺畅
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import MultiHeadAttention


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network

    两层全连接 + GELU 激活（原始论文用 ReLU）

    结构: Linear(d_model → d_ff) → ReLU → Linear(d_ff → d_model)

    d_ff 通常是 d_model 的 4 倍（512 → 2048），
    这个"先膨胀再压缩"的设计让 FFN 有足够的容量做非线性变换
    """

    def __init__(self, d_model: int = 512, d_ff: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # d_model → d_ff → ReLU → Dropout → d_model
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class EncoderLayer(nn.Module):
    """一个 Transformer Encoder 层 = Self-Attention + FFN + 两个残差连接"""

    def __init__(self, d_model: int = 512, num_heads: int = 8,
                 d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()

        # Self-Attention 子层
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)

        # Feed-Forward 子层
        self.ffn = FeedForward(d_model, d_ff, dropout)

        # LayerNorm（Pre-Norm 风格）
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None
                ) -> torch.Tensor:
        """
        x: (B, src_len, d_model)
        src_mask: (B, 1, 1, src_len) padding mask
        """
        # --- Self-Attention 子层（Pre-Norm）---
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, src_mask)  # Q=K=V=x
        x = self.dropout(x)
        x = residual + x  # 残差连接

        # --- FFN 子层（Pre-Norm）---
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = residual + x

        return x


class DecoderLayer(nn.Module):
    """
    一个 Transformer Decoder 层 =
      Self-Attention (因果) + Cross-Attention + FFN + 三个残差连接
    """

    def __init__(self, d_model: int = 512, num_heads: int = 8,
                 d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()

        # Self-Attention（带因果掩码）
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)

        # Cross-Attention（Q=decoder, K&V=encoder）
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)

        # Feed-Forward
        self.ffn = FeedForward(d_model, d_ff, dropout)

        # LayerNorm × 3
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                    # (B, tgt_len, d_model)
        enc_output: torch.Tensor,           # (B, src_len, d_model)
        tgt_mask: torch.Tensor = None,      # causal mask
        mem_mask: torch.Tensor = None,      # encoder padding mask
    ) -> torch.Tensor:
        # --- Self-Attention（causal）---
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, tgt_mask)
        x = self.dropout(x)
        x = residual + x

        # --- Cross-Attention（Q=dec, K&V=enc）---
        residual = x
        x = self.norm2(x)
        x = self.cross_attn(x, enc_output, enc_output, mem_mask)
        x = self.dropout(x)
        x = residual + x

        # --- FFN ---
        residual = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = residual + x

        return x


class Encoder(nn.Module):
    """完整 Encoder = Embedding + PositionalEncoding + N × EncoderLayer"""

    def __init__(self, vocab_size: int, d_model: int = 512,
                 num_heads: int = 8, num_layers: int = 6,
                 d_ff: int = 2048, max_len: int = 5000,
                 dropout: float = 0.1, pad_idx: int = 0):
        super().__init__()
        from embedding import TokenEmbedding, PositionalEncoding

        self.token_embed = TokenEmbedding(vocab_size, d_model, pad_idx)
        self.pos_embed = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)  # 最终 LayerNorm
        self.pad_idx = pad_idx

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None
                ) -> torch.Tensor:
        x = self.pos_embed(self.token_embed(x))
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """完整 Decoder = Embedding + PositionalEncoding + N × DecoderLayer"""

    def __init__(self, vocab_size: int, d_model: int = 512,
                 num_heads: int = 8, num_layers: int = 6,
                 d_ff: int = 2048, max_len: int = 5000,
                 dropout: float = 0.1, pad_idx: int = 0):
        super().__init__()
        from embedding import TokenEmbedding, PositionalEncoding

        self.token_embed = TokenEmbedding(vocab_size, d_model, pad_idx)
        self.pos_embed = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.pad_idx = pad_idx

    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        mem_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        返回: (B, tgt_len, vocab_size) logits
        """
        x = self.pos_embed(self.token_embed(x))
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, mem_mask)
        x = self.norm(x)
        return self.fc_out(x)


class Transformer(nn.Module):
    """
    完整 Transformer 模型

    用法：
      model = Transformer(src_vocab, tgt_vocab)
      logits = model(src, tgt_input, src_mask, tgt_mask, mem_mask)
    """

    def __init__(self, src_vocab: int, tgt_vocab: int,
                 d_model: int = 512, num_heads: int = 8,
                 num_layers: int = 6, d_ff: int = 2048,
                 max_len: int = 5000, dropout: float = 0.1,
                 pad_idx: int = 0):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, num_heads, num_layers,
                               d_ff, max_len, dropout, pad_idx)
        self.decoder = Decoder(tgt_vocab, d_model, num_heads, num_layers,
                               d_ff, max_len, dropout, pad_idx)
        self.pad_idx = pad_idx

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
        mem_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        enc_output = self.encoder(src, src_mask)
        return self.decoder(tgt, enc_output, tgt_mask, mem_mask)
