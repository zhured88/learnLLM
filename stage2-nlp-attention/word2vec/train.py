"""
Word2Vec Skip-gram 训练循环

关键要点：
  - 使用 Adam 优化器（比 SGD 更适合稀疏梯度的场景）
  - 不需要像 RNN 那样的 BPTT（每个中心词独立训练）
  - 每个 epoch 后评估词类比准确率
"""

import time
import torch
import torch.nn as nn
from word2vec import SkipGramNeg, find_analogy, find_similar


def train_epoch(
    model: SkipGramNeg,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    clip: float = 1.0,
) -> float:
    """
    训练一轮

    流程：取 (center, context, neg_samples) → 前向 → 反向 → 梯度裁剪 → 更新
    """
    model.train()
    total_loss = 0.0

    for center, context, neg_samples in loader:
        center = center.to(device)
        context = context.to(device)
        neg_samples = neg_samples.to(device)

        optimizer.zero_grad()
        loss = model(center, context, neg_samples)
        loss.backward()

        # 梯度裁剪——防止偶然的大梯度破坏训练
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_analogies(
    model: SkipGramNeg,
    word_to_idx: dict,
    idx_to_word: dict,
    analogies: list,
    top_k: int = 1,
):
    """
    评估词类比准确率

    analogies 格式：[("king", "man", "woman", "queen"), ...]
    返回 top-1 准确率
    """
    vectors = model.get_vectors()
    correct = 0
    total = 0

    for a, b, c, expected in analogies:
        if any(w not in word_to_idx for w in [a, b, c, expected]):
            continue
        results = find_analogy(a, b, c, word_to_idx, idx_to_word,
                               vectors, top_k=top_k)
        predicted_words = [w for w, _ in results[:top_k]]
        if expected in predicted_words:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def train(
    model: SkipGramNeg,
    train_loader: torch.utils.data.DataLoader,
    word_to_idx: dict,
    idx_to_word: dict,
    test_words: list,          # 用于打印"相似词"的测试词
    analogies: list,            # 类比测试用例
    epochs: int = 10,
    lr: float = 0.001,
    device: torch.device = None,
):
    """
    Word2Vec 训练总控

    Word2Vec 通常不需要很多 epoch（5-10 轮即可），
    因为数据量已经足够覆盖每个词的多种上下文
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\n{'='*60}")
    print(f"  Word2Vec Skip-gram + Negative Sampling")
    print(f"  Vocab: {model.vocab_size:,}, Embed Dim: {model.embed_dim}")
    print(f"  Device: {device}, Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Epochs: {epochs}, LR: {lr}")
    print(f"{'='*60}")
    print(f"{'Epoch':>5} {'Loss':>10} {'Analogy Acc':>12} {'Time':>8}")
    print("-" * 42)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        loss = train_epoch(model, train_loader, optimizer, device)
        elapsed = time.time() - t0

        # 评估类比准确率（每 2 个 epoch 或最后一个 epoch）
        acc = 0.0
        if epoch % 2 == 0 or epoch == epochs:
            acc = evaluate_analogies(model, word_to_idx, idx_to_word, analogies)

        print(f"{epoch:>5} {loss:>10.4f} {acc:>11.2%} {elapsed:>7.1f}s")

    # --- 训练结束后，展示词相似度和类比 ---
    print(f"\n{'='*60}")
    print("  词相似度测试")
    print(f"{'='*60}")
    vectors = model.get_vectors()
    for w in test_words:
        similar = find_similar(w, word_to_idx, idx_to_word, vectors, top_k=5)
        print(f"  {w:12s} → {[s[0] for s in similar]}")

    print(f"\n{'='*60}")
    print("  词类比测试")
    print(f"{'='*60}")
    for a, b, c, expected in analogies[:10]:
        results = find_analogy(a, b, c, word_to_idx, idx_to_word, vectors, top_k=3)
        print(f"  {a} - {b} + {c} ≈ {[r[0] for r in results]}  (期望: {expected})")

    return model
