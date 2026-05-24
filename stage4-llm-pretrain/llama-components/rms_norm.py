"""
RMSNorm: Root Mean Square Layer Normalization

LLaMA 用 RMSNorm 替代 LayerNorm，核心区别：
  LayerNorm: y = (x - mean) / std * gamma + beta
  RMSNorm:  y = x / rms(x) * gamma

去掉均值中心化，省一次归约操作，训练/推理更快。
Paper: "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)
"""

import torch
import torch.nn as nn
import time


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    y = x / sqrt(mean(x^2) + eps) * gamma

    与 LayerNorm 的区别:
      - 不做均值中心化 (zero-mean)
      - 不用 beta (bias) 参数
      - 只缩放，不平移
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (*, dim)，在最后一维做归一化
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.gamma


# ── 对比测试 ──────────────────────────────────────────────────

def compare_speed():
    """对比 RMSNorm 和 LayerNorm 的前向/反向速度"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, T, C = 32, 512, 768
    x = torch.randn(B, T, C, device=device)

    rms_norm = RMSNorm(C).to(device)
    ln = nn.LayerNorm(C).to(device)

    # Warmup
    for _ in range(20):
        rms_norm(x).sum().backward()
        ln(x).sum().backward()

    torch.cuda.synchronize() if device == "cuda" else None

    # RMSNorm timing
    t0 = time.time()
    for _ in range(200):
        y = rms_norm(x)
        y.sum().backward()
    torch.cuda.synchronize() if device == "cuda" else None
    t_rms = time.time() - t0

    # LayerNorm timing
    t0 = time.time()
    for _ in range(200):
        y = ln(x)
        y.sum().backward()
    torch.cuda.synchronize() if device == "cuda" else None
    t_ln = time.time() - t0

    print(f"RMSNorm:   {t_rms:.3f}s")
    print(f"LayerNorm: {t_ln:.3f}s")
    print(f"Speedup:   {t_ln / t_rms:.2f}x")


def compare_output():
    """展示 RMSNorm 和 LayerNorm 输出的数值差异"""
    x = torch.randn(2, 3, 4)
    rms = RMSNorm(4)
    ln = nn.LayerNorm(4, elementwise_affine=False)

    with torch.no_grad():
        y_rms = rms(x)
        y_ln = ln(x)

    print("输入 x:", x)
    print("\nRMSNorm 输出:", y_rms)
    print("\nLayerNorm 输出:", y_ln)
    print("\nRMSNorm mean:", y_rms.mean(dim=-1))  # 不一定是零
    print("LayerNorm mean:", y_ln.mean(dim=-1))   # 接近零
    print("\n两者都让 std ≈ 1")


if __name__ == "__main__":
    print("=" * 50)
    print("RMSNorm vs LayerNorm 速度对比")
    print("=" * 50)
    compare_speed()

    print("\n" + "=" * 50)
    print("RMSNorm vs LayerNorm 输出对比")
    print("=" * 50)
    compare_output()
