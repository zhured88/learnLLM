"""
SwiGLU: Gated Linear Unit with Swish Activation

LLaMA 用 SwiGLU 替代 GPT-2 的 GELU FFN：

  GPT-2 MLP:  x -> Linear(d, 4d) -> GELU -> Linear(4d, d)
  SwiGLU:      x -> Linear(d, 2d') [拆成 gate, up] -> SiLU(gate) * up -> Linear(d', d)

  其中 d' = 2/3 * 4d = 8d/3，保持总参数量一致。

Paper: "GLU Variants Improve Transformer" (Shazeer, 2020)
LLaMA 选择 β=1 的 Swish (即 SiLU): SiLU(x) = x * sigmoid(x)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiLU(nn.Module):
    """SiLU (Sigmoid Linear Unit) = Swish with β=1.

    SiLU(x) = x * sigmoid(x)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """SwiGLU FFN — LLaMA 使用的激活前馈层。

    SwiGLU(x) = (SiLU(x @ W_gate)) * (x @ W_up) @ W_down

    dim -> hidden_dim 膨胀比例: 2/3 * 4 = 8/3
    使 SwiGLU 与标准 4x GELU-MLP 的参数量一致。
    """

    def __init__(self, dim: int, hidden_dim: int = None, dropout: float = 0.0):
        super().__init__()
        if hidden_dim is None:
            # 标准 LLaMA 比例：让总参数量 = 2 * dim * hidden (gate+up) + hidden * dim (down)
            # 对比 4x GELU: 2 * dim * 4d = 8d^2
            # SwiGLU: 2 * dim * d' + d' * dim = 3 * dim * d'
            # 令 3 * dim * d' = 8 * dim * dim => d' = 8d/3
            hidden_dim = int(2 * 4 * dim / 3)  # 8d/3
            # 取最近的 256 倍数（硬件友好）
            hidden_dim = ((hidden_dim + 255) // 256) * 256

        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))  # SiLU(gate)
        up = self.up_proj(x)               # linear projection
        return self.dropout(self.down_proj(gate * up))


# ── 对比测试 ──────────────────────────────────────────────────

class GELU_MLP(nn.Module):
    """GPT-2 风格的 GELU-MLP（来自阶段 3）。"""
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc = nn.Linear(dim, 4 * dim)
        self.proj = nn.Linear(4 * dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.proj(F.gelu(self.fc(x))))


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def compare_ffn(dim: int = 768):
    """对比 SwiGLU 和 GELU-MLP 的参数量和输出 shape。"""
    swiglu = SwiGLU(dim)
    gelu_mlp = GELU_MLP(dim)

    swiglu_params = count_params(swiglu)
    gelu_params = count_params(gelu_mlp)

    print(f"dim = {dim}")
    print(f"  SwiGLU hidden_dim: {swiglu.gate_proj.out_features}")
    print(f"  SwiGLU params:      {swiglu_params:,}")
    print(f"  GELU-MLP params:    {gelu_params:,}")
    print(f"  参数量比:           {swiglu_params / gelu_params:.2f}")

    x = torch.randn(2, 16, dim)
    y_swiglu = swiglu(x)
    y_gelu = gelu_mlp(x)

    print(f"  SwiGLU output: {y_swiglu.shape}")
    print(f"  GELU-MLP output: {y_gelu.shape}")


if __name__ == "__main__":
    print("=" * 50)
    print("SwiGLU vs GELU-MLP 对比")
    print("=" * 50)
    compare_ffn(768)
    print()

    # 验证 SiLU
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    print("SiLU 验证:")
    print(f"  x:      {x}")
    print(f"  SiLU(x): {F.silu(x)}")
    print(f"  手算:    {x * torch.sigmoid(x)}")
