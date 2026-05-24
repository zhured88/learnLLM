"""训练循环 — 字符级 RNN"""

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
    clip: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        batch_size = x.size(0)

        hidden = model.init_hidden(batch_size, device)
        if isinstance(hidden, tuple):
            hidden = tuple(h.detach() for h in hidden)
        else:
            hidden = hidden.detach()

        optimizer.zero_grad()
        logits, hidden = model(x, hidden)

        # logits: (B, T, V), y: (B, T)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()

        # 梯度裁剪：防止 RNN 梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: CharRNN,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
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
    lr: float = 0.001,
    device: torch.device = None,
    prompt: str = "ROMEO:",
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        perplexity = torch.exp(torch.tensor(val_loss)).item()
        elapsed = time.time() - t0

        print(f"{epoch:>5} {train_loss:>12.4f} {val_loss:>12.4f} "
              f"{perplexity:>11.2f} {elapsed:>7.1f}s")

        # 每 5 轮生成一段样本文本
        if epoch % 5 == 0:
            sample = model.generate(
                prompt, char_to_idx, idx_to_char,
                length=150, temperature=0.8, device=device,
            )
            print(f"\n  --- Generation Sample (epoch {epoch}) ---")
            print(f"  {sample[:200]}")
            print()

    return model
