"""
Transformer Embedding 层：Token Embedding + Sinusoidal Positional Encoding

核心概念：
  - Transformer 没有 RNN 的循环结构，必须显式注入位置信息
  - Sinusoidal Positional Encoding: 用 sin/cos 函数编码位置，
    允许模型外推到比训练时更长的序列

公式（原始论文）：
  PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

  pos: 位置索引 (0, 1, 2, ..., max_len-1)
  i:   维度索引 (0, 1, 2, ..., d_model/2-1)
  10000: 温度参数（频率缩放因子）

直观理解：
  - 低维度 (i 小) → 波长长 → 编码"全局位置感"（第 0 个 vs 第 50 个词）
  - 高维度 (i 大) → 波长短 → 编码"局部位置感"（第 5 个 vs 第 6 个词）
  - 不同频率的正弦波叠加 → 每个位置有唯一的编码向量
"""

import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    """词嵌入：把 token 索引映射到 d_model 维空间，并乘以 sqrt(d_model)"""

    def __init__(self, vocab_size: int, d_model: int, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, seq_len) token 索引
        返回: (B, seq_len, d_model)
        """
        # 乘以 sqrt(d_model) 是原始 Transformer 的做法：
        # embedding 权重初始化为 N(0, 1)，乘以 sqrt(d_model) 后和 positional encoding 尺度匹配
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding

    特点：
      - 固定编码（非可学习参数），训练和推断时都相同
      - 因为用三角函数，模型可以外推：训练时 max_len=100，推断时 max_len=150 也能工作
      - 和可学习位置编码的区别：GPT/LLaMA 改用可学习 PE，因为数据量大时可以学到更好的表示
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # --- 创建位置编码矩阵 (max_len, d_model) ---
        # pos: (max_len, 1)
        # i:   (d_model/2,)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        # div_term = 1 / (10000^(2i/d_model)) 的等价高效写法

        pe[:, 0::2] = torch.sin(pos * div_term)   # 偶数维度 = sin
        pe[:, 1::2] = torch.cos(pos * div_term)   # 奇数维度 = cos

        # 注册为 buffer（不是参数，但会随模型保存/加载）
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, seq_len, d_model) token embedding 输出
        返回: (B, seq_len, d_model) 加上位置信息后的表示
        """
        x = x + self.pe[:, :x.size(1), :]  # 截取实际序列长度，加到 embedding 上
        return self.dropout(x)
