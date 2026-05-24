"""
训练循环 —— 字符级 RNN 文本生成

和 ResNet 训练的关键区别：
  1. BPTT（沿时间反向传播）：RNN 的梯度要沿着 100 个时间步往回传
  2. hidden.detach()：截断计算图，防止显存爆炸
  3. 梯度裁剪：防止 RNN 梯度爆炸（偶尔某个罕见字符序列会导致梯度飙升）
  4. 每 5 轮生成一段样本文本，直观感受模型进步
"""

import time
import torch
import torch.nn as nn
from rnn_gen import CharRNN


def train_epoch(
    model: CharRNN,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    clip: float = 1.0,  # 梯度裁剪阈值：梯度范数超过此值则等比例缩回
) -> float:
    """
    训练一轮

    和 CNN 训练的关键区别：
      - 需要初始化隐藏状态（每个 batch 独立）
      - hidden.detach() 截断计算图（Truncated BPTT）
      - 梯度裁剪（clip_grad_norm_）
    """
    model.train()
    total_loss = 0.0

    for x, y in loader:
        # --- 搬数据到 GPU ---
        x, y = x.to(device), y.to(device)        # x: (B, 100), y: (B, 100)
        batch_size = x.size(0)

        # --- ① 初始化隐藏状态 ---
        # 每个 batch 都是独立的文本片段 → 隐藏状态从头开始（全零）
        hidden = model.init_hidden(batch_size, device)

        # --- ② 截断计算图 ★ BPTT 的关键操作 ---
        # 如果不 detach()，计算图会从第 1 个 batch 延伸到第 N 个 batch → 显存爆炸
        # detach() 告诉 PyTorch："从这里断开，前面的梯度别往回传了"
        # 但当前 batch 的 100 个时间步内，梯度仍然是连通的（这就是 BPTT 的 T）
        if isinstance(hidden, tuple):
            hidden = tuple(h.detach() for h in hidden)  # LSTM：detach (h, c)
        else:
            hidden = hidden.detach()                     # RNN/GRU：detach h

        # --- ③ 前向传播 ---
        optimizer.zero_grad()
        logits, hidden = model(x, hidden)

        # --- ④ 计算 loss ---
        # logits: (B, 100, 65), y: (B, 100)
        # .view(-1, 65)：展平成 (B×100, 65)，每个时间步的每个字符都独立监督
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

        # --- ⑤ 反向传播 ---
        loss.backward()

        # --- ⑥ 梯度裁剪 ★ 防止 RNN 梯度爆炸 ---
        # 计算所有参数梯度的总范数，如果超过 clip，等比例缩回 clip
        # 不带裁剪的 RNN 训练时，偶尔某个 batch 梯度会飙升到几万 → NaN
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # --- ⑦ 参数更新 ---
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)  # 返回平均每个 batch 的 loss


@torch.no_grad()
def evaluate(
    model: CharRNN,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    验证一轮：只前向传播，不反向、不更新

    不需要 detach hidden，因为 @torch.no_grad() 已经关闭了梯度追踪
    """
    model.eval()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        batch_size = x.size(0)

        hidden = model.init_hidden(batch_size, device)
        logits, _ = model(x, hidden)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item()

    return total_loss / len(loader)


def train(
    model: CharRNN,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    char_to_idx: dict,
    idx_to_char: dict,
    epochs: int = 20,
    lr: float = 0.001,        # Adam 的默认学习率，比 SGD 小 100 倍
    device: torch.device = None,
    prompt: str = "ROMEO:",    # 每 5 轮生成文本的起始提示
):
    """
    总控训练函数

    特色：
      - 使用 Adam 优化器（RNN 训练基本都用 Adam）
      - 输出困惑度（Perplexity = exp(loss)，模型"犹豫"的候选数）
      - 每 5 轮用当前模型生成一段文本，直观感受进步
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")           # NVIDIA GPU
        elif torch.backends.mps.is_available():
            device = torch.device("mps")            # Apple Silicon GPU（M1/M2/M3/M4）
        else:
            device = torch.device("cpu")            # 纯 CPU 兜底

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\n{'='*60}")
    print(f"  Char-RNN ({model.rnn_type.upper()}) · Shakespeare Text Generation")
    print(f"  Device: {device}, Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*60}")
    print(f"{'Epoch':>5} {'Train Loss':>12} {'Val Loss':>12} {'Perplexity':>12} {'Time':>8}")
    print("-" * 56)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        # 困惑度 = exp(交叉熵 loss)
        # 含义：模型在猜下一个字符时，平均从多少个候选中选
        # PPL=2  → 只在 2 个候选中犹豫（非常确定）
        # PPL=65 → 在全部 65 个字符中随机猜（完全没学到东西）
        perplexity = torch.exp(torch.tensor(val_loss)).item()
        elapsed = time.time() - t0

        print(f"{epoch:>5} {train_loss:>12.4f} {val_loss:>12.4f} "
              f"{perplexity:>11.2f} {elapsed:>7.1f}s")

        # 每 5 轮生成一段样本文本，可视化模型进步
        if epoch % 5 == 0:
            sample = model.generate(
                prompt, char_to_idx, idx_to_char,
                length=150, temperature=0.8, device=device,
            )
            print(f"\n  --- Generation Sample (epoch {epoch}) ---")
            print(f"  {sample[:200]}")
            print()

    return model
