#!/usr/bin/env python3
"""
模型可解释性分析 + 逐轮预测可视化

每轮训练后，对 10 张固定测试样本生成预测截图（含 saliency map），
输出 PNG 到 viz/ 目录。

运行：python interpret.py
"""

import sys
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")  # 无头模式，不弹窗
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from src.data import load_mnist
from src.model import MLP
from src.losses import CrossEntropyLoss
from src.optim import SGD
from src.data import get_batches
from src.train import compute_accuracy

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "viz")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 中文字体设置
plt.rcParams["font.family"] = "Arial Unicode MS"
plt.rcParams["axes.unicode_minus"] = False


def saliency_map(model: MLP, x: np.ndarray, target_class: int) -> np.ndarray:
    """计算输入 x 对 target_class 的 saliency map (28, 28)"""
    h1 = model.fc1.forward(x)
    h1_relu = model.relu.forward(h1)
    logits = model.fc2.forward(h1_relu)

    dlogits = np.zeros_like(logits)
    dlogits[0, target_class] = 1.0

    d_h1_relu = model.fc2.backward(dlogits)
    d_h1 = model.relu.backward(d_h1_relu)
    dx = model.fc1.backward(d_h1)

    return np.abs(dx).reshape(28, 28)


def plot_epoch_samples(
    model: MLP,
    samples_X: np.ndarray,
    samples_y: np.ndarray,
    epoch: int,
):
    """
    对 10 个固定样本，生成一张 2 行 x 5 列的预测图：
      - 每个格子显示原始图像 + saliency map 叠加
      - 标题显示 [真实/预测] + 置信度
      - 正确=绿色边框，错误=红色边框
    """
    logits = model.forward(samples_X)
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    preds = logits.argmax(axis=1)

    fig, axes = plt.subplots(4, 5, figsize=(22, 18))
    fig.suptitle(
        f"Epoch {epoch:02d} — 10 个测试样本预测结果", fontsize=20, fontweight="bold", y=0.98
    )

    for i in range(10):
        row_img = 0 if i < 5 else 2
        col = i % 5

        ax_img = axes[row_img, col]
        ax_sal = axes[row_img + 1, col]

        true_label = samples_y[i]
        pred_label = preds[i]
        conf = probs[i, pred_label]
        is_correct = true_label == pred_label

        # --- 原始图片 ---
        img = samples_X[i].reshape(28, 28)
        ax_img.imshow(img, cmap="gray_r")
        border_color = "#2ecc71" if is_correct else "#e74c3c"
        for spine in ax_img.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(4)
        title = f"[{true_label}->{pred_label}] conf={conf:.3f}"
        ax_img.set_title(title, fontsize=14, color=border_color, fontweight="bold")
        ax_img.set_xticks([])
        ax_img.set_yticks([])

        # --- Saliency map ---
        smap = saliency_map(model, samples_X[i : i + 1], pred_label)
        ax_sal.imshow(img, cmap="gray_r", alpha=0.4)
        im = ax_sal.imshow(smap, cmap="hot", alpha=0.6)
        ax_sal.set_title(f"Saliency (预测={pred_label})", fontsize=12)
        ax_sal.set_xticks([])
        ax_sal.set_yticks([])

    # 隐藏多余的轴（共 10 个样本，4x5=20 格子，只用了 20 个）
    for ax in axes.flat:
        if not ax.has_data():
            ax.set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUTPUT_DIR, f"epoch_{epoch:02d}.png")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_final_report(
    model: MLP,
    samples_X: np.ndarray,
    samples_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
):
    """最终报告：大图汇总 10 个样本 + 首层权重"""
    logits = model.forward(test_X)
    acc = compute_accuracy(logits, test_y)

    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    preds = logits.argmax(axis=1)

    fig = plt.figure(figsize=(26, 20))
    fig.suptitle(
        f"最终测试准确率: {acc:.2%}  |  10 个样本详细分析",
        fontsize=22, fontweight="bold", y=0.98,
    )

    # --- 上半部分：10 个样本（2 行 × 5 列）---
    for i in range(10):
        ax_img = fig.add_subplot(3, 10, i + 1)
        ax_sal = fig.add_subplot(3, 10, i + 11)

        true_label = samples_y[i]
        pred_label = preds[i]
        conf = probs[i, pred_label]
        is_correct = true_label == pred_label

        img = samples_X[i].reshape(28, 28)
        ax_img.imshow(img, cmap="gray_r")
        color = "#2ecc71" if is_correct else "#e74c3c"
        for spine in ax_img.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
        ax_img.set_title(f"[{true_label}->{pred_label}]\n{conf:.3f}", fontsize=10, color=color)
        ax_img.set_xticks([])
        ax_img.set_yticks([])

        smap = saliency_map(model, samples_X[i : i + 1], pred_label)
        ax_sal.imshow(img, cmap="gray_r", alpha=0.35)
        ax_sal.imshow(smap, cmap="hot", alpha=0.65)
        ax_sal.set_xticks([])
        ax_sal.set_yticks([])
        if i == 0:
            ax_img.set_ylabel("原始图片", fontsize=13, fontweight="bold")
            ax_sal.set_ylabel("Saliency", fontsize=13, fontweight="bold")

    # --- 中间：置信度分布柱状图 ---
    ax_conf = fig.add_subplot(3, 2, 3)
    correct_mask = preds == test_y
    correct_conf = probs[correct_mask].max(axis=1)
    wrong_conf = probs[~correct_mask].max(axis=1)

    bins = np.linspace(0, 1, 21)
    ax_conf.hist(correct_conf, bins=bins, alpha=0.7, label=f"正确 ({correct_mask.sum()})", color="#2ecc71", edgecolor="white")
    ax_conf.hist(wrong_conf, bins=bins, alpha=0.7, label=f"错误 ({(~correct_mask).sum()})", color="#e74c3c", edgecolor="white")
    ax_conf.set_xlabel("置信度", fontsize=13)
    ax_conf.set_ylabel("样本数", fontsize=13)
    ax_conf.set_title("置信度分布：正确 vs 错误预测", fontsize=14, fontweight="bold")
    ax_conf.legend(fontsize=11)
    ax_conf.axvline(x=0.7, color="orange", linestyle="--", linewidth=1.5, label="0.7 阈值")
    ax_conf.set_yscale("log")

    # --- 混淆矩阵热力图 ---
    ax_cm = fig.add_subplot(3, 2, 4)
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(test_y, preds):
        cm[t, p] += 1
    # 只显示非对角线（错误）
    cm_err = cm.copy()
    np.fill_diagonal(cm_err, 0)
    im = ax_cm.imshow(cm_err, cmap="YlOrRd", aspect="auto")
    ax_cm.set_xticks(range(10))
    ax_cm.set_yticks(range(10))
    ax_cm.set_xlabel("预测", fontsize=13)
    ax_cm.set_ylabel("真实", fontsize=13)
    ax_cm.set_title("混淆矩阵（仅显示错误）", fontsize=14, fontweight="bold")
    for i in range(10):
        for j in range(10):
            if cm_err[i, j] > 0:
                ax_cm.text(j, i, str(cm_err[i, j]), ha="center", va="center",
                           fontsize=8, fontweight="bold",
                           color="white" if cm_err[i, j] > 3 else "black")
    plt.colorbar(im, ax=ax_cm, shrink=0.8)

    # --- 底部：fc1 权重可视化 ---
    ax_w = fig.add_subplot(3, 1, 3)
    n_neurons = 20
    w_grid = []
    for n in range(n_neurons):
        w = model.fc1.W[:, n].reshape(28, 28)
        w = (w - w.min()) / (w.max() - w.min() + 1e-8)
        w_grid.append(w)
    big_w = np.zeros((28 * 2, 28 * 10))
    for idx, w in enumerate(w_grid):
        r, c = divmod(idx, 10)
        big_w[r * 28 : (r + 1) * 28, c * 28 : (c + 1) * 28] = w
    ax_w.imshow(big_w, cmap="RdBu_r", aspect="auto")
    ax_w.set_title("fc1 前 20 个隐藏神经元的权重模板（红=正权, 蓝=负权）", fontsize=14, fontweight="bold")
    ax_w.set_xticks([])
    ax_w.set_yticks([])
    for n in range(n_neurons):
        r, c = divmod(n, 10)
        ax_w.text(c * 28 + 14, r * 28 + 14, str(n), ha="center", va="center",
                  fontsize=8, color="black", fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUTPUT_DIR, "final_report.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def train_with_visualization(
    model: MLP,
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 0.1,
    momentum: float = 0.9,
    lr_decay: float = 0.95,
):
    """逐轮训练，每轮对 10 个固定测试样本截图"""
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.linear_layers, lr=lr, momentum=momentum)

    # 固定 10 个测试样本（每个数字各取 1 个，尽量找"典型但有点难度"的）
    np.random.seed(42)
    fixed_indices = []
    for digit in range(10):
        digit_idx = np.where(test_y == digit)[0]
        # 每个数字随机选 1 个
        chosen = np.random.choice(digit_idx, 1)
        fixed_indices.append(chosen[0])
    fixed_indices = np.array(fixed_indices)
    samples_X = test_X[fixed_indices]
    samples_y = test_y[fixed_indices]

    print(f"\n{'Epoch':>5} {'Train Loss':>12} {'Train Acc':>10} {'Test Acc':>10} {'Time':>8}")
    print("-" * 52)

    for epoch in range(1, epochs + 1):
        import time
        t0 = time.time()
        total_loss = 0.0
        total_acc = 0.0
        n_batch = 0

        for X_batch, y_batch in get_batches(train_X, train_y, batch_size):
            logits = model.forward(X_batch)
            loss = criterion.forward(logits, y_batch)
            dout = criterion.backward()
            model.backward(dout)
            optimizer.step()

            total_loss += loss
            total_acc += compute_accuracy(logits, y_batch)
            n_batch += 1

        optimizer.lr *= lr_decay

        test_logits = model.forward(test_X)
        test_acc = compute_accuracy(test_logits, test_y)

        avg_loss = total_loss / n_batch
        avg_acc = total_acc / n_batch
        elapsed = time.time() - t0

        print(
            f"{epoch:>5} {avg_loss:>12.4f} {avg_acc:>9.2%} {test_acc:>9.2%} {elapsed:>7.1f}s"
        )

        # 每轮生成可视化
        path = plot_epoch_samples(model, samples_X, samples_y, epoch)
        print(f"  -> 保存: {path}")

    return samples_X, samples_y


def main():
    print("=" * 60)
    print("  模型可解释性分析 + 逐轮预测可视化")
    print("=" * 60)

    # 加载数据
    print("\n[1/3] 加载 MNIST 数据...")
    train_X, train_y, test_X, test_y = load_mnist()

    # 构建模型
    print("[2/3] 构建模型...")
    model = MLP(input_dim=784, hidden_dim=256, num_classes=10)
    print(f"  参数量: {model.total_params:,}")

    # 逐轮训练 + 可视化
    print(f"[3/3] 训练 {10} 轮，每轮生成可视化...\n")
    samples_X, samples_y = train_with_visualization(
        model, train_X, train_y, test_X, test_y, epochs=10
    )

    # 最终报告
    print("\n生成最终报告...")
    report_path = plot_final_report(model, samples_X, samples_y, test_X, test_y)
    print(f"  -> 保存: {report_path}")

    logits = model.forward(test_X)
    acc = compute_accuracy(logits, test_y)
    print(f"\n测试准确率: {acc:.2%}")
    print(f"所有图片保存在: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
