"""
Transformer 训练模块 —— Label Smoothing + Adam Warmup + 训练循环

核心技巧：
  1. Label Smoothing: 把 one-hot label 软化（如 1.0 → 0.9, 其他 → 0.1/N）
     防止模型过度自信，提升泛化能力（Transformer 论文的核心 trick）
  2. Adam + Warmup: 前 warmup_steps 步线性增加 LR，之后按 1/sqrt(step) 衰减
     原始论文公式: lr = d_model^{-0.5} × min(step^{-0.5}, step × warmup^{-1.5})
  3. 梯度裁剪 + 混合精度（可选）
"""

import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import Transformer
from attention import make_combined_mask


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing CrossEntropy

    原始 CrossEntropy:  loss = -log(p_correct)
    Label Smoothing:     loss = (1-ε) × (-log(p_correct)) + ε × mean(-log(p))
                         即把"100% 确信"变成"90% 确信 + 10% 均匀"

    ε (smoothing): 原始论文用 0.1
    """

    def __init__(self, smoothing: float = 0.1, pad_idx: int = 0):
        super().__init__()
        self.smoothing = smoothing
        self.pad_idx = pad_idx

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, T, V) raw logits
        target: (B, T)  token 索引
        """
        V = logits.size(-1)
        confidence = 1.0 - self.smoothing

        # 标准 NLL loss
        nll = F.cross_entropy(
            logits.view(-1, V), target.view(-1),
            ignore_index=self.pad_idx, reduction="none",
        )

        # 平滑项：对每个位置均匀分布的 KL 散度
        # log(p) 在均匀分布下的期望 = -log(V) 的负熵近似
        log_probs = F.log_softmax(logits, dim=-1)
        smooth_loss = -log_probs.mean(dim=-1)  # 负的均匀分布交叉熵

        # 组合：nll 占 (1-ε)，smooth 占 ε
        mask = (target != self.pad_idx).float()
        loss = confidence * nll + self.smoothing * smooth_loss.view(-1)

        return (loss * mask.view(-1)).sum() / mask.sum().clamp(min=1)


class AdamWarmup:
    """
    Transformer 论文的 Adam Warmup 学习率调度器

    公式（"Attention Is All You Need" 第 5.3 节）：
      lr = d_model^{-0.5} × min(step_num^{-0.5}, step_num × warmup_steps^{-1.5})

    直觉：
      - 前 warmup 步：线性增加（"预热"阶段，不要一开始就跑太快）
      - 之后：逐步衰减（接近收敛时需要小步幅精细搜索）
    """

    def __init__(self, optimizer: torch.optim.Optimizer,
                 d_model: int, warmup_steps: int = 4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def _get_lr(self):
        step = self.step_num
        arg1 = step ** (-0.5)
        arg2 = step * (self.warmup_steps ** (-1.5))
        return (self.d_model ** (-0.5)) * min(arg1, arg2)


def train_epoch(
    model: Transformer,
    loader: torch.utils.data.DataLoader,
    criterion: LabelSmoothingLoss,
    optimizer: AdamWarmup,
    device: torch.device,
    clip: float = 1.0,
) -> float:
    """训练一轮"""
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)

        # decoder 输入/输出拆分
        tgt_input = tgt[:, :-1]   # <SOS> ... token_{n-1}
        tgt_output = tgt[:, 1:]   # token_1 ... <EOS>

        # 生成掩码
        src_mask, tgt_mask, mem_mask = make_combined_mask(
            src, tgt_input, model.pad_idx, model.pad_idx,
        )

        optimizer.zero_grad()

        logits = model(src, tgt_input, src_mask, tgt_mask, mem_mask)
        # logits: (B, tgt_len-1, V), tgt_output: (B, tgt_len-1)

        loss = criterion(logits, tgt_output)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        n_tokens = (tgt_output != model.pad_idx).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def evaluate(
    model: Transformer,
    loader: torch.utils.data.DataLoader,
    criterion: LabelSmoothingLoss,
    device: torch.device,
) -> float:
    """验证"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_mask, tgt_mask, mem_mask = make_combined_mask(
            src, tgt_input, model.pad_idx, model.pad_idx,
        )

        logits = model(src, tgt_input, src_mask, tgt_mask, mem_mask)
        loss = criterion(logits, tgt_output)

        n_tokens = (tgt_output != model.pad_idx).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    return total_loss / max(total_tokens, 1)


def train(
    model: Transformer,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int = 20,
    d_model: int = 512,
    warmup_steps: int = 4000,
    smoothing: float = 0.1,
    clip: float = 1.0,
    device: torch.device = None,
):
    """总控训练"""
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    model = model.to(device)
    criterion = LabelSmoothingLoss(smoothing, model.pad_idx)
    base_optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98),
                                      eps=1e-9)
    optimizer = AdamWarmup(base_optimizer, d_model, warmup_steps)

    print(f"\n{'='*60}")
    print(f"  Transformer · Neural Machine Translation")
    print(f"  d_model={d_model}, warmup={warmup_steps}, label_smoothing={smoothing}")
    print(f"  Device: {device}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*60}")
    print(f"{'Epoch':>5} {'Train Loss':>12} {'Val Loss':>12} "
          f"{'Val PPL':>10} {'Time':>8}")
    print("-" * 54)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer,
                                 device, clip)
        val_loss = evaluate(model, val_loader, criterion, device)
        val_ppl = math.exp(min(val_loss, 10))
        elapsed = time.time() - t0

        print(f"{epoch:>5} {train_loss:>12.4f} {val_loss:>12.4f} "
              f"{val_ppl:>9.2f} {elapsed:>7.1f}s")

    return model
