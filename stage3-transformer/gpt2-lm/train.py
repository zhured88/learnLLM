"""
GPT-2 预训练循环

和翻译训练的关键区别：
  1. 输出直接和输入比较（语言模型的自监督任务）
  2. 输入 = 前 T-1 个 token, 输出 = 预测第 1..T 个 token

GPT-2 论文的训练配置（Small）：
  - Batch size: 64
  - Sequence length: 512
  - Learning rate: 2.5e-4 (warmup + cosine decay)
  - Optimizer: Adam (β1=0.9, β2=0.95, ε=1e-8)
  - Weight decay: 0.1
  - Gradient clipping: 1.0
"""

import time
import math
import torch
import torch.nn as nn
from gpt2 import GPT2


def train_epoch(
    model: GPT2,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    clip: float = 1.0,
    grad_accum_steps: int = 1,
) -> float:
    """
    训练一轮

    grad_accum_steps: 梯度累积步数（显存不够时增大此值，模拟更大的 batch）
    """
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        _, loss = model(x, y)

        # 梯度累积：除以累积步数
        loss = loss / grad_accum_steps
        loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps * x.size(0)
        total_tokens += (y != model.pad_idx).sum().item()

    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def evaluate(
    model: GPT2,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """验证"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item() * (y != model.pad_idx).sum().item()
        total_tokens += (y != model.pad_idx).sum().item()

    return total_loss / max(total_tokens, 1)


def train(
    model: GPT2,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int = 5,
    lr: float = 2.5e-4,
    weight_decay: float = 0.1,
    device: torch.device = None,
    warmup_epochs: int = 1,
    grad_accum_steps: int = 1,
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

    # GPT-2 论文使用的 Adam 参数
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95),
        eps=1e-8, weight_decay=weight_decay,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"  GPT-2 Language Model Pre-training")
    print(f"  Layers: {len(model.blocks)}, Dim: {model.n_embd}")
    print(f"  Device: {device}")
    print(f"  Params: {total_params:,}")
    print(f"  LR: {lr}, Weight Decay: {weight_decay}")
    print(f"{'='*60}")
    print(f"{'Epoch':>5} {'Train Loss':>12} {'Val Loss':>12} "
          f"{'Val PPL':>10} {'LR':>10} {'Time':>8}")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # 线性 warmup + cosine decay（简化版）
        if epoch <= warmup_epochs:
            current_lr = lr * epoch / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
            current_lr = lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        train_loss = train_epoch(model, train_loader, optimizer, device,
                                  grad_accum_steps=grad_accum_steps)
        val_loss = evaluate(model, val_loader, device)
        val_ppl = math.exp(min(val_loss, 10))
        elapsed = time.time() - t0

        print(f"{epoch:>5} {train_loss:>12.4f} {val_loss:>12.4f} "
              f"{val_ppl:>9.2f} {current_lr:>8.2e} {elapsed:>7.1f}s")

        # 每轮生成示例文本
        sample = generate_sample(model, device)
        print(f"\n  --- Sample ---")
        print(f"  {sample[:200]}")
        print()

    return model


@torch.no_grad()
def generate_sample(model: GPT2, device: torch.device,
                    prompt: str = "The", max_new: int = 100) -> str:
    """生成一段示例文本"""
    model.eval()
    # 用最简单的 prompt 生成
    idx = torch.tensor([[0]], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new,
                         temperature=0.8, top_k=40)
    # 转回文本（简化）
    return f"[generated {max_new} tokens]"


def get_lr_schedule(lr: float, warmup_epochs: int, total_epochs: int,
                    epoch: int) -> float:
    """GPT-2 风格的学习率调度：warmup + cosine decay"""
    if epoch <= warmup_epochs:
        return lr * epoch / warmup_epochs
    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    return lr * 0.5 * (1.0 + math.cos(math.pi * progress))
