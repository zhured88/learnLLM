#!/usr/bin/env python3
"""
ResNet-18 CIFAR-10 训练结果可视化

功能：
  1. 训练曲线（loss + accuracy 双轴图）
  2. 测试图片预测展示（正确/错误 + 置信度）
  3. 第一层卷积核可视化（学到的边缘检测器）
  4. 特征图可视化（逐层展示网络"看到"什么）
  5. 混淆矩阵热力图
  6. 单张图片推理应用

运行：
  python visualize.py                          # 训练 10 轮 + 可视化（快速体验）
  python visualize.py --epochs 30              # 训练 30 轮
  python visualize.py --load checkpoints/resnet18.pth  # 加载已训练模型
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from data import get_cifar10
from resnet import ResNet18, PlainNet18

# 中文字体
plt.rcParams["font.family"] = "Arial Unicode MS"
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "viz")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CIFAR10_CLASSES = [
    "飞机", "汽车", "鸟", "猫", "鹿",
    "狗", "青蛙", "马", "船", "卡车",
]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================
# 图 1：训练曲线
# ============================================================
def plot_training_curves(history: dict, save_path: str):
    """训练/测试 loss 和 accuracy 双轴曲线"""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("ResNet-18 CIFAR-10 训练曲线", fontsize=16, fontweight="bold")

    # --- Loss 曲线 ---
    ax1.plot(epochs, history["train_loss"], "b-", label="训练 Loss", linewidth=1.5)
    ax1.plot(epochs, history["test_loss"], "r-", label="测试 Loss", linewidth=1.5)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Loss 曲线", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    # 标注最优 loss
    best_idx = np.argmin(history["test_loss"])
    ax1.annotate(f'Best: {history["test_loss"][best_idx]:.4f}',
                 xy=(best_idx + 1, history["test_loss"][best_idx]),
                 xytext=(best_idx + 3, history["test_loss"][best_idx] + 0.1),
                 arrowprops=dict(arrowstyle="->", color="red"), fontsize=9, color="red")

    # --- Accuracy 曲线 ---
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "b-", label="训练 Acc", linewidth=1.5)
    ax2.plot(epochs, [a * 100 for a in history["test_acc"]], "r-", label="测试 Acc", linewidth=1.5)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_title("准确率曲线", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    best_idx = np.argmax(history["test_acc"])
    ax2.annotate(f'Best: {history["test_acc"][best_idx]*100:.1f}%',
                 xy=(best_idx + 1, history["test_acc"][best_idx] * 100),
                 xytext=(best_idx + 3, history["test_acc"][best_idx] * 100 - 2),
                 arrowprops=dict(arrowstyle="->", color="red"), fontsize=9, color="red")

    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [1/5] 训练曲线 → {save_path}")


# ============================================================
# 图 2：测试图片预测展示
# ============================================================
def plot_predictions(model, test_loader, device, save_path: str, num: int = 20):
    """展示 20 张测试图片的预测结果：绿色边框=正确，红色=错误"""
    model.eval()
    classes = CIFAR10_CLASSES

    # 获取一批数据和预测
    images, labels = next(iter(test_loader))
    images, labels = images[:num], labels[:num]
    with torch.no_grad():
        outputs = model(images.to(device))
        probs = F.softmax(outputs, dim=1).cpu()
        preds = outputs.argmax(1).cpu()

    # 反归一化（用于显示）
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    images = images * std + mean
    images = images.clamp(0, 1)

    rows = 4
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(16, 13))
    fig.suptitle("测试图片预测结果（绿=正确，红=错误）", fontsize=16, fontweight="bold")

    for i, ax in enumerate(axes.flat):
        img = images[i].permute(1, 2, 0).numpy()
        true_label = labels[i].item()
        pred_label = preds[i].item()
        conf = probs[i, pred_label].item()
        is_correct = true_label == pred_label

        ax.imshow(img)
        color = "#2ecc71" if is_correct else "#e74c3c"
        title = f"真实:{classes[true_label]}\n预测:{classes[pred_label]} ({conf:.2f})"
        ax.set_title(title, fontsize=10, color=color, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(4 if not is_correct else 2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [2/5] 预测展示 → {save_path}")


# ============================================================
# 图 3：第一层卷积核可视化
# ============================================================
def plot_conv_filters(model, save_path: str):
    """可视化 conv1 的 64 个 3×3 卷积核"""
    # 提取第一层卷积权重
    w = model.conv1.weight.data.cpu().numpy()  # (64, 3, 3, 3)
    w = (w - w.min()) / (w.max() - w.min() + 1e-8)

    fig, axes = plt.subplots(8, 8, figsize=(12, 12))
    fig.suptitle("conv1 的 64 个卷积核（每个核 3×3，RGB 三通道）\n"
                 "亮色=正权（兴奋），暗色=负权（抑制）",
                 fontsize=13, fontweight="bold")

    for i, ax in enumerate(axes.flat):
        # 将 3 通道的 3×3 核显示为 RGB 图（转置为 H×W×C）
        kernel = w[i].transpose(1, 2, 0)
        ax.imshow(kernel, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [3/5] 卷积核 → {save_path}")


# ============================================================
# 图 4：特征图可视化
# ============================================================
def plot_feature_maps(model, test_loader, device, save_path: str):
    """对一张测试图片，展示各层输出的特征图"""
    model.eval()

    # 取一张图片
    images, _ = next(iter(test_loader))
    img = images[0:1].to(device)

    # 手动逐层前向，记录每层输出
    feature_maps = {}

    x = img
    x = model.conv1(x)
    x = model.bn1(x)
    x = F.relu(x)
    feature_maps["conv1 + BN + ReLU"] = x

    x = model.layer1(x)
    feature_maps["layer1 (32×32, 64ch)"] = x

    x = model.layer2(x)
    feature_maps["layer2 (16×16, 128ch)"] = x

    x = model.layer3(x)
    feature_maps["layer3 (8×8, 256ch)"] = x

    x = model.layer4(x)
    feature_maps["layer4 (4×4, 512ch)"] = x

    fig, axes = plt.subplots(len(feature_maps), 8, figsize=(18, 2.5 * len(feature_maps)))
    fig.suptitle("ResNet-18 逐层特征图可视化\n"
                 "浅层：边缘/纹理检测 → 深层：语义概念（物体局部/轮廓）",
                 fontsize=13, fontweight="bold")

    for row_idx, (name, fm) in enumerate(feature_maps.items()):
        fm_np = fm[0].cpu().detach().numpy()  # (C, H, W)
        n_channels = fm_np.shape[0]

        for col in range(8):
            ax = axes[row_idx, col]
            ch = col * max(1, n_channels // 8)
            if ch < n_channels:
                ax.imshow(fm_np[ch], cmap="viridis")
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(name, fontsize=10, fontweight="bold")

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [4/5] 特征图 → {save_path}")


# ============================================================
# 图 5：混淆矩阵
# ============================================================
def plot_confusion_matrix(model, test_loader, device, save_path: str):
    """混淆矩阵热力图，标注最容易混淆的类别对"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            preds = outputs.argmax(1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_preds)
    classes = CIFAR10_CLASSES

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(classes, fontsize=10, rotation=45)
    ax.set_yticklabels(classes, fontsize=10)
    ax.set_xlabel("预测类别", fontsize=12)
    ax.set_ylabel("真实类别", fontsize=12)
    ax.set_title(f"混淆矩阵（总样本: {len(test_loader.dataset):,}）", fontsize=14, fontweight="bold")

    # 在格子里显示数字
    for i in range(10):
        for j in range(10):
            text = str(cm[i, j]) if cm[i, j] > 0 else ""
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, shrink=0.8, label="样本数")
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [5/5] 混淆矩阵 → {save_path}")

    return cm


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="ResNet 训练结果可视化")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--load", type=str, default=None, help="加载已训练模型路径")
    parser.add_argument("--plain", action="store_true", help="使用 PlainNet-18")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # --- 加载数据 ---
    print("加载 CIFAR-10 数据...")
    train_loader, test_loader = get_cifar10(batch_size=args.batch_size, augment=True)

    # --- 构建或加载模型 ---
    if args.plain:
        model = PlainNet18()
        model_name = "PlainNet-18 (无残差连接)"
    else:
        model = ResNet18()
        model_name = "ResNet-18"

    if args.load:
        print(f"加载预训练模型: {args.load}")
        model.load_state_dict(torch.load(args.load, map_location=device))
        model.to(device)
        # 快速评估
        from train import evaluate
        import torch.nn as nn
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"  测试 Loss: {test_loss:.4f}, 测试 Acc: {test_acc:.2%}")
        # 构造模拟 history（只有最终值）
        history = {"train_loss": [], "train_acc": [], "test_loss": [test_loss], "test_acc": [test_acc], "lr": []}
    else:
        print(f"训练 {model_name}，{args.epochs} 轮...")
        from train import train
        _, history = train(
            model, train_loader, test_loader,
            epochs=args.epochs, lr=0.1,
            device=device, label=model_name,
        )
        # 保存模型
        os.makedirs("./checkpoints", exist_ok=True)
        save_name = "plainnet18" if args.plain else "resnet18"
        torch.save(model.state_dict(), f"./checkpoints/{save_name}.pth")

    # --- 生成可视化图 ---
    print("\n生成可视化...")

    if len(history["train_loss"]) > 0:
        plot_training_curves(history, os.path.join(OUTPUT_DIR, "01_training_curves.png"))
    else:
        print("  [1/5] 训练曲线 → 跳过（加载的是已训练模型）")

    if not args.load or len(history["train_loss"]) > 0:
        plot_predictions(model, test_loader, device,
                         os.path.join(OUTPUT_DIR, "02_predictions.png"))

    plot_conv_filters(model, os.path.join(OUTPUT_DIR, "03_conv_filters.png"))
    plot_feature_maps(model, test_loader, device,
                      os.path.join(OUTPUT_DIR, "04_feature_maps.png"))

    cm = plot_confusion_matrix(model, test_loader, device,
                               os.path.join(OUTPUT_DIR, "05_confusion_matrix.png"))

    # --- 打印关键统计 ---
    if not args.load or len(history["train_loss"]) > 0:
        print(f"\n{'='*50}")
        print(f"  训练结果摘要")
        print(f"{'='*50}")
        best_idx = int(np.argmax(history["test_acc"]))
        best_acc = history["test_acc"][best_idx] * 100
        print(f"  最佳测试准确率: {best_acc:.2f}% (第 {best_idx+1} 轮)")
        print(f"  总参数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  所有图片保存在: {OUTPUT_DIR}/")

    # 最易混淆的类别对
    cm_off = cm.copy()
    np.fill_diagonal(cm_off, 0)
    worst = np.unravel_index(np.argmax(cm_off), cm_off.shape)
    print(f"  最易混淆: {CIFAR10_CLASSES[worst[0]]} → {CIFAR10_CLASSES[worst[1]]} ({cm_off[worst]} 次)")


if __name__ == "__main__":
    main()
