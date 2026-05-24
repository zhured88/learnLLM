"""
LLM 预训练循环 — Stage 4

升级点 vs Stage 3:
  - 混合精度训练 (AMP): torch.cuda.amp (GradScaler + autocast)
  - 梯度累积: 模拟更大的 effective batch size
  - 检查点保存/恢复: 支持中断后继续训练
  - 学习率调度: warmup -> cosine decay -> min_lr
  - 更丰富的指标: tokens/sec, GPU 内存

训练配置参考 (LLaMA paper):
  - AdamW: β1=0.9, β2=0.95, weight_decay=0.1
  - LR: 3e-4 (warmup 2000 steps) -> cosine decay
  - Gradient clipping: 1.0
"""

import os
import time
import math
import torch
import torch.nn as nn

from model import LLaMA


CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def train_epoch(model: LLaMA, loader, optimizer, scaler,
                device: torch.device, epoch: int,
                grad_accum_steps: int = 1, clip_grad: float = 1.0,
                use_amp: bool = True):
    """训练一个 epoch。

    Returns:
        avg_loss, tokens_per_sec
    """
    model.train()
    total_loss = 0.0
    total_tokens = 0
    start_time = time.time()

    for step, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        # 前向传播（混合精度）
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            logits, loss = model(x, targets=y)
            loss = loss / grad_accum_steps  # 归一化到 per-step

        # 反向传播
        if use_amp and device.type == "cuda" and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # 梯度累积：每 grad_accum_steps 步更新一次
        if (step + 1) % grad_accum_steps == 0:
            if use_amp and device.type == "cuda" and scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # 统计
        total_loss += loss.item() * grad_accum_steps * x.size(0) * x.size(1)
        total_tokens += x.numel()

        if step % 50 == 0 and step > 0:
            elapsed = time.time() - start_time
            tokens_per_sec = total_tokens / elapsed
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"  Step {step:4d} | Loss: {loss.item() * grad_accum_steps:.4f} "
                  f"| LR: {current_lr:.2e} | {tokens_per_sec:.0f} tok/s", end="\r")

    elapsed = time.time() - start_time
    avg_loss = total_loss / max(total_tokens, 1)
    tokens_per_sec = total_tokens / elapsed

    return avg_loss, tokens_per_sec


@torch.no_grad()
def evaluate(model: LLaMA, loader, device: torch.device):
    """评估模型。"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits, loss = model(x, targets=y)
        total_loss += loss.item() * x.numel()
        total_tokens += x.numel()

    return total_loss / max(total_tokens, 1)


def get_lr(step: int, total_steps: int, lr: float, warmup_steps: int,
           min_lr: float = 1e-6) -> float:
    """warmup -> cosine decay 学习率调度。"""
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def save_checkpoint(model: LLaMA, optimizer, scaler, epoch: int, loss: float,
                    path: str = None, model_size: str = "tiny"):
    """保存检查点。"""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    if path is None:
        path = os.path.join(CHECKPOINT_DIR, f"llama_{model_size}_epoch{epoch:02d}.pt")

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "loss": loss,
        "model_config": {
            "vocab_size": model.vocab_size,
            "n_embd": model.n_embd,
            "n_layer": model.n_layer,
            "n_head": model.n_head,
            "n_kv_head": model.n_kv_head,
        }
    }, path)
    print(f"  检查点已保存: {path}")


def load_checkpoint(model: LLaMA, optimizer, scaler, path: str, device: torch.device):
    """加载检查点。"""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler and checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print(f"  检查点已加载: {path} (epoch={epoch}, loss={loss:.4f})")
    return epoch, loss


def train(model: LLaMA, train_loader, val_loader,
          epochs: int = 3, lr: float = 3e-4,
          weight_decay: float = 0.1, warmup_steps: int = 500,
          min_lr: float = 1e-6,
          grad_accum_steps: int = 1, clip_grad: float = 1.0,
          use_amp: bool = True, device: torch.device = None,
          checkpoint_interval: int = 1, resume_from: str = None,
          model_size: str = "tiny"):
    """完整训练流程。

    Args:
        model: LLaMA 模型
        train_loader/val_loader: DataLoader
        epochs: 训练 epoch 数
        lr: 峰值学习率
        weight_decay: AdamW weight decay
        warmup_steps: 线性 warmup 步数
        min_lr: 最低学习率
        grad_accum_steps: 梯度累积步数
        clip_grad: 梯度裁剪阈值
        use_amp: 是否使用混合精度
        device: 设备
        checkpoint_interval: 每隔几个 epoch 保存一次
        resume_from: 恢复检查点路径
        model_size: 模型大小（用于命名检查点）
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    print(f"设备: {device}")

    # 优化器 (AdamW)
    # 分开处理 weight decay 和不 decay 的参数（norm/embedding 不 decay）
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "norm" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0}
    ], lr=lr, betas=(0.9, 0.95), eps=1e-8)

    # 混合精度 scaler
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda") if use_amp else None

    # 计算总步数
    total_steps = epochs * len(train_loader) // grad_accum_steps
    start_epoch = 0

    # 恢复检查点
    if resume_from:
        loaded_epoch, _ = load_checkpoint(model, optimizer, scaler, resume_from, device)
        start_epoch = loaded_epoch

    model = model.to(device)

    # 训练循环
    best_val_loss = float("inf")
    global_step = start_epoch * len(train_loader) // grad_accum_steps

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()

        # 更新学习率
        for pg in optimizer.param_groups:
            pg["lr"] = get_lr(global_step, total_steps, lr, warmup_steps, min_lr)

        # 训练
        train_loss, tokens_per_sec = train_epoch(
            model, train_loader, optimizer, scaler,
            device, epoch, grad_accum_steps, clip_grad, use_amp
        )

        # 评估
        val_loss = evaluate(model, val_loader, device) if val_loader else float("nan")
        val_ppl = math.exp(min(val_loss, 10))

        # 更新步数、确保最后一步不溢出
        global_step += len(train_loader) // grad_accum_steps

        # 更新学习率为下一步
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        # 打印
        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val PPL: {val_ppl:.1f} | "
              f"LR: {current_lr:.2e} | "
              f"{tokens_per_sec:.0f} tok/s | "
              f"{epoch_time:.0f}s")

        # 生成样本
        if device.type != "mps":
            sample_prompt = "The meaning of life is"
            generate_sample(model, sample_prompt, device)

        # 保存检查点
        if (epoch + 1) % checkpoint_interval == 0 or val_loss < best_val_loss:
            best_val_loss = min(val_loss, best_val_loss)
            save_checkpoint(model, optimizer, scaler, epoch + 1, val_loss,
                          model_size=model_size)

    return model


@torch.no_grad()
def generate_sample(model: LLaMA, prompt: str, device: torch.device,
                    tokenizer=None, max_new: int = 50):
    """生成示例文本（训练中监控用）。"""
    model.eval()

    # 简单 tokenize（用空格分词 + 字符编码作为兜底）
    if tokenizer:
        ids = tokenizer.encode(prompt)
        idx = torch.tensor([ids], device=device)
        output_ids = model.generate(idx, max_new_tokens=max_new)
        generated = tokenizer.decode(output_ids[0].tolist())
    else:
        # 简单兜底：用字符 ID
        ids = [ord(c) % 1000 + 4 for c in prompt]
        idx = torch.tensor([ids], device=device)
        output_ids = model.generate(idx, max_new_tokens=max_new)
        generated = "".join(chr(min(i, 127)) for i in output_ids[0].tolist())

    # 截断显示
    if len(generated) > 200:
        generated = generated[:200] + "..."
    print(f"  [Sample] {generated}")
