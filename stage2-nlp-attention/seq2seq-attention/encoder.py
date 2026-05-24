"""
Encoder：双向 GRU 编码器

核心概念：
  - 双向 RNN：两个方向各跑一个 RNN，把两个方向的隐藏状态拼接
    → 每个时间步的表示包含"上文信息"和"下文信息"
  - 对翻译任务，知道一个词前后的上下文对理解其含义至关重要
    "bank" 在 "river bank" 和 "money bank" 中含义不同，
    双向编码让模型能看到名词前后的词来判断语义

输出：
  enc_outputs: (B, src_len, 2 * hidden_dim)  ← 所有时间步的双向拼接隐藏状态
  hidden:      (num_layers * 2, B, hidden_dim) ← 最终隐藏状态（单向的 2 倍）
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    双向 GRU 编码器

    结构：
      Embedding → Dropout → Bidirectional GRU → (outputs, hidden)

    维度变化：
      输入:  (B, src_len)           单词索引
      embed: (B, src_len, embed_dim)  词向量
      GRU:   (B, src_len, 2*hidden_dim) 双向输出
      hidden: (2*num_layers, B, hidden_dim)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pad_idx = pad_idx

        # Embedding：把离散词索引映射到连续空间
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        # padding_idx=0 表示 <PAD> token 的 embedding 始终为 0，不会被训练更新

        self.dropout = nn.Dropout(dropout)

        # 双向 GRU
        # bidirectional=True → 输出维度翻倍（正向 + 反向 concat）
        self.gru = nn.GRU(
            embed_dim, hidden_dim, num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

    def forward(
        self,
        src: torch.Tensor,          # (B, src_len)
        src_lengths: torch.Tensor,  # (B,) 每个句子的实际长度
    ) -> tuple:
        """
        返回：
          enc_outputs: (B, src_len, 2 * hidden_dim)  双向 GRU 输出
          hidden:      (2 * num_layers, B, hidden_dim) GRU 最终隐状态
        """
        # Embedding
        embed = self.dropout(self.embedding(src))  # (B, S) → (B, S, E)

        # pack_padded_sequence：跳过 <PAD> 位置，让 GRU 只在有效时间步上计算
        # 这是加速 + 防止 <PAD> 干扰隐藏状态的关键操作
        packed = nn.utils.rnn.pack_padded_sequence(
            embed, src_lengths.cpu(), batch_first=True, enforce_sorted=False,
        )

        outputs, hidden = self.gru(packed)

        # unpack：恢复成正常的 padded tensor
        # outputs: (B, src_len, 2 * hidden_dim)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        return outputs, hidden
