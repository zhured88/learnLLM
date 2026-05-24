"""
注意力机制 —— Bahdanau (Additive) & Luong (Dot / General / Concat)

核心概念：
  - 注意力是一个"软查找"操作：Decoder 在每个时间步，自动找到 Encoder 输出中
    和当前翻译最相关的部分
  - 注意力得分 → softmax → 注意力权重 → 加权求和 → 上下文向量

三种经典注意力：

  1. Bahdanau (Additive / MLP) —— 论文最早提出的注意力
     score(h_j, s_{t-1}) = v^T · tanh(W_h h_j + W_s s_{t-1})
     特点：一层全连接后投影到标量，对齐效果最直观

  2. Luong Dot —— 最简单的注意力
     score(h_j, s_t) = h_j^T · s_t
     特点：无参数，要求 h_j 和 s_t 维度相同

  3. Luong General —— 带可学习变换的点积
     score(h_j, s_t) = h_j^T · W · s_t
     特点：W 矩阵可以适应 encoder/decoder 维度不同的情况

  4. Luong Concat —— 类似 Bahdanau
     score(h_j, s_t) = v^T · tanh(W · [h_j; s_t])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    """
    Bahdanau (Additive) Attention

    论文：Bahdanau et al., "Neural Machine Translation by Jointly Learning
          to Align and Translate", ICLR 2015

    计算流程：
      ① energy_j = v^T · tanh(W_h h_j + W_s s_{t-1})
         其中 h_j 是 encoder 在位置 j 的隐藏状态
             s_{t-1} 是 decoder 上一步的隐藏状态
      ② alpha = softmax(energy)     ← 注意力权重（和为 1）
      ③ context = Σ alpha_j · h_j   ← 加权求和 = 上下文向量

    参数维度：
      W_h: (hidden_dim, hidden_dim)  作用在 encoder hidden 上
      W_s: (hidden_dim, hidden_dim)  作用在 decoder hidden 上
      v:   (hidden_dim, 1)          投影到标量
    """

    def __init__(self, enc_hidden_dim: int, dec_hidden_dim: int):
        super().__init__()
        self.W_h = nn.Linear(enc_hidden_dim, dec_hidden_dim, bias=False)
        self.W_s = nn.Linear(dec_hidden_dim, dec_hidden_dim, bias=False)
        self.v = nn.Linear(dec_hidden_dim, 1, bias=False)

    def forward(
        self,
        enc_outputs: torch.Tensor,   # (B, src_len, enc_hidden_dim)
        dec_hidden: torch.Tensor,     # (B, dec_hidden_dim)
        mask: torch.Tensor = None,    # (B, src_len) True=有效位置
    ) -> tuple:
        """
        返回：
          context: (B, dec_hidden_dim) 上下文向量
          attn_weights: (B, src_len) 注意力权重（可视化用）
        """
        # ① 计算每个 encoder 位置的得分
        # enc_outputs:  (B, S, H_enc), dec_hidden: (B, H_dec)
        # W_h(h_j):     (B, S, H_dec)
        energy_h = self.W_h(enc_outputs)    # (B, S, H_dec)

        # W_s(s_{t-1}): (B, H_dec) → (B, 1, H_dec) 广播到每个时间步
        energy_s = self.W_s(dec_hidden).unsqueeze(1)  # (B, 1, H_dec)

        # tanh 相加 → 投影到标量
        energy = self.v(torch.tanh(energy_h + energy_s)).squeeze(-1)  # (B, S)

        # ② 遮罩：把 <PAD> 位置的得分设为 -inf，softmax 后权重 = 0
        if mask is not None:
            energy = energy.masked_fill(~mask, float("-inf"))

        # ③ softmax → 注意力权重
        attn_weights = F.softmax(energy, dim=1)  # (B, S)

        # ④ 加权求和 → 上下文向量
        context = torch.bmm(attn_weights.unsqueeze(1), enc_outputs)  # (B, 1, H_enc)
        context = context.squeeze(1)  # (B, H_enc)

        return context, attn_weights


class LuongAttention(nn.Module):
    """
    Luong Attention —— 三种得分函数

    论文：Luong et al., "Effective Approaches to Attention-based NMT", EMNLP 2015

    和 Bahdanau 的区别：
      - Luong 用 s_t（当前 decoder hidden）而非 s_{t-1}（上一步）
      - 即先算出 s_t，再用它去对齐 encoder 的 h_j
      - Luong 提出多种得分函数（dot / general / concat）

    score 函数：
      dot:     score = h_j^T · s_t
      general: score = h_j^T · W · s_t
      concat:  score = v^T · tanh(W · [h_j; s_t])
    """

    def __init__(self, enc_hidden_dim: int, dec_hidden_dim: int,
                 method: str = "general"):
        super().__init__()
        self.method = method.lower()

        if self.method == "general":
            # W: (enc_hidden_dim, dec_hidden_dim) 可学习的双线性变换
            self.W = nn.Linear(dec_hidden_dim, enc_hidden_dim, bias=False)
        elif self.method == "concat":
            self.W = nn.Linear(enc_hidden_dim + dec_hidden_dim,
                               enc_hidden_dim, bias=False)
            self.v = nn.Linear(enc_hidden_dim, 1, bias=False)
        # dot: 不需要参数

    def _score(self, enc_outputs: torch.Tensor,
               dec_hidden: torch.Tensor) -> torch.Tensor:
        """
        计算注意力得分

        参数：
          enc_outputs: (B, S, H_enc)
          dec_hidden:  (B, H_dec)
        返回：
          scores: (B, S)
        """
        if self.method == "dot":
            # (B, S, H) × (B, H) → (B, S)
            return torch.bmm(enc_outputs, dec_hidden.unsqueeze(2)).squeeze(2)

        elif self.method == "general":
            # h_j^T · W · s_t = h_j^T · (W · s_t)
            # W·s_t: (B, H_dec) → (B, H_enc)
            dec_transformed = self.W(dec_hidden)  # (B, H_enc)
            # (B, S, H_enc) × (B, H_enc) → (B, S)
            return torch.bmm(enc_outputs, dec_transformed.unsqueeze(2)).squeeze(2)

        elif self.method == "concat":
            # [h_j; s_t]: (B, S, H_enc + H_dec)
            S = enc_outputs.size(1)
            dec_expanded = dec_hidden.unsqueeze(1).expand(-1, S, -1)  # (B, S, H_dec)
            concat = torch.cat([enc_outputs, dec_expanded], dim=2)     # (B, S, 2H)
            energy = torch.tanh(self.W(concat))  # (B, S, H)
            return self.v(energy).squeeze(2)     # (B, S)

        else:
            raise ValueError(f"Unknown attention method: {self.method}")

    def forward(
        self,
        enc_outputs: torch.Tensor,
        dec_hidden: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> tuple:
        """返回 context, attn_weights"""
        scores = self._score(enc_outputs, dec_hidden)  # (B, S)

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=1)  # (B, S)
        context = torch.bmm(attn_weights.unsqueeze(1), enc_outputs)  # (B, 1, H_enc)
        context = context.squeeze(1)  # (B, H_enc)

        return context, attn_weights
