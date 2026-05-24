"""
RoPE: Rotary Position Embedding 旋转位置编码

LLaMA 用 RoPE 替代可学习位置编码。核心思想：
  - 对 Q/K 的每一对维度 (2i, 2i+1) 施加一个 2D 旋转
  - 旋转角度 θ_i = 10000^(-2i/d)，位置 m 旋转 m·θ_i
  - 效果: q_m^T k_n 只依赖于相对位置 (m-n)

数学:
  f_q(x_m, m) = R_m * x_m    (把位置 m 的第 i 对维度旋转 m·θ_i 弧度)
  f_k(x_n, n) = R_n * x_n
  则 (R_m·q)^T (R_n·k) = q^T R_{n-m} k    —— 只依赖相对位置

Paper: "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
"""

import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class RotaryPositionEmbedding(nn.Module):
    """RoPE: 对 Q/K 施加旋转位置编码。

    用法:
      rope = RotaryPositionEmbedding(d_k=64, max_seq_len=2048, theta=10000.0)
      q_rotated, k_rotated = rope(q, k, seq_len=T)
    """

    def __init__(self, d_k: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.theta = theta

        # 预计算频率: θ_i = theta^(-2i/d)  for i = 0..d_k/2-1
        # shape: (d_k/2,)
        i = torch.arange(0, d_k, 2).float()
        freqs = 1.0 / (theta ** (i / d_k))  # (d_k/2,)

        # 预计算 cos(m·θ), sin(m·θ) for all positions
        # shape: (max_seq_len, d_k/2)
        m = torch.arange(max_seq_len).float()
        freqs = torch.outer(m, freqs)  # (max_seq_len, d_k/2)

        # 注册为 buffer（不参与训练，但随模型移动设备）
        self.register_buffer("cos_cached", freqs.cos())  # (max_seq_len, d_k/2)
        self.register_buffer("sin_cached", freqs.sin())  # (max_seq_len, d_k/2)

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                seq_len: int = None, offset: int = 0):
        """对 Q 和 K 施加旋转位置编码。

        Args:
            q: (B, n_head, T, d_k)
            k: (B, n_head, T, d_k)
            seq_len: 当前序列长度
            offset: KV cache 的偏移量（推理时使用）

        Returns:
            q_rotated, k_rotated: 与输入同形状
        """
        if seq_len is None:
            seq_len = q.shape[2]

        cos = self.cos_cached[offset:offset + seq_len, :]  # (T, d_k/2)
        sin = self.sin_cached[offset:offset + seq_len, :]  # (T, d_k/2)
        # 广播维度: (1, 1, T, d_k/2)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        q_rotated = self._apply_rotary(q, cos, sin)
        k_rotated = self._apply_rotary(k, cos, sin)
        return q_rotated, k_rotated

    def _apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """对 x 的每一对维度 (2i, 2i+1) 施加 2D 旋转。

        [x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]
        """
        # 把 x 分成偶数维和奇数维
        x_even = x[..., 0::2]  # (..., d_k/2)
        x_odd = x[..., 1::2]   # (..., d_k/2)

        # 旋转
        rot_even = x_even * cos - x_odd * sin
        rot_odd = x_even * sin + x_odd * cos

        # 交错拼回
        result = torch.empty_like(x)
        result[..., 0::2] = rot_even
        result[..., 1::2] = rot_odd
        return result


# ── 验证 RoPE 的相对位置性质 ──────────────────────────────────

def verify_relative_property():
    """验证 q_m^T k_n 只依赖于 (m-n)，而非绝对位置。"""
    d_k, n_head = 64, 1
    rope = RotaryPositionEmbedding(d_k, max_seq_len=128)

    # 两个相同内容的向量，放在不同位置
    q_content = torch.randn(1, n_head, 1, d_k)  # 单个 query
    k_content = torch.randn(1, n_head, 1, d_k)   # 单个 key

    # 构建分别在位置 0 和位置 5 的 Q
    q0 = q_content.repeat(1, 1, 1, 1)
    q5 = q_content.repeat(1, 1, 1, 1)

    # 构建分别在位置 3 和位置 8 的 K（保持相对位置差 3）
    k3 = k_content.repeat(1, 1, 1, 1)
    k8 = k_content.repeat(1, 1, 1, 1)

    # 手动构造包含所有 4 个位置的序列
    # 位置 0..9，在位置 0 和 5 放 q，在位置 3 和 8 放 k
    T = 10
    q_seq = torch.randn(1, n_head, T, d_k)
    k_seq = torch.randn(1, n_head, T, d_k)
    q_seq[:, :, 0, :] = q_content[:, :, 0, :]
    q_seq[:, :, 5, :] = q_content[:, :, 0, :]
    k_seq[:, :, 3, :] = k_content[:, :, 0, :]
    k_seq[:, :, 8, :] = k_content[:, :, 0, :]

    q_rot, k_rot = rope(q_seq, k_seq, seq_len=T)

    # q0·k3 和 q5·k8 应该相同（相对位置差都是 3）
    score_0_3 = (q_rot[0, 0, 0] * k_rot[0, 0, 3]).sum()
    score_5_8 = (q_rot[0, 0, 5] * k_rot[0, 0, 8]).sum()

    print("RoPE 相对位置性质验证:")
    print(f"  q0·k3 = {score_0_3:.6f}")
    print(f"  q5·k8 = {score_5_8:.6f}")
    print(f"  差值 = {abs(score_0_3 - score_5_8):.8f}")


# ── 可视化旋转频率 ────────────────────────────────────────────

def visualize_frequencies():
    """可视化 RoPE 不同维度对的旋转频率。"""
    d_k = 64
    rope = RotaryPositionEmbedding(d_k, max_seq_len=256, theta=10000.0)

    cos = rope.cos_cached[:, :4].cpu().numpy()  # 取前 4 对维度
    positions = range(256)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, ax in enumerate(axes.flat):
        ax.plot(positions, cos[:, i])
        ax.set_title(f"维度对 {i} (θ = 10000^(-2×{i}/{d_k}))")
        ax.set_xlabel("位置 m")
        ax.set_ylabel("cos(m·θ)")
        ax.grid(True, alpha=0.3)

    plt.suptitle("RoPE: 不同维度对的旋转频率", fontsize=14)
    plt.tight_layout()
    plt.savefig("rope_frequencies.png", dpi=150)
    print("可视化已保存到 rope_frequencies.png")


if __name__ == "__main__":
    print("=" * 50)
    print("RoPE 验证")
    print("=" * 50)
    verify_relative_property()

    print("\n" + "=" * 50)
    print("RoPE 频率可视化")
    print("=" * 50)
    visualize_frequencies()
