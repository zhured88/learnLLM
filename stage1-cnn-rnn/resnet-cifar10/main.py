#!/usr/bin/env python3
"""
阶段 1 · Mini 项目 1：PyTorch 复现 ResNet-18 · CIFAR-10 分类

核心实验（三个对比维度）：
  1. 残差连接的效果：   python main.py         vs  python main.py --plain
  2. 数据增强的效果：   python main.py         vs  python main.py --no-aug
  3. 残差 + 增强组合：  python main.py          ← 预期最佳

运行示例：
  python main.py                      # 标准 ResNet-18，50 轮，有数据增强
  python main.py --plain              # PlainNet-18（无残差连接）
  python main.py --no-aug             # 关闭数据增强
  python main.py --epochs 100         # 训练 100 轮
  python main.py --batch_size 64      # 减小 batch size（显存不够时）
"""

import argparse
import torch
from data import get_cifar10
from resnet import ResNet18, PlainNet18
from train import train


def main():
    # --- 命令行参数 ---
    parser = argparse.ArgumentParser(description="ResNet-18 CIFAR-10 训练")
    parser.add_argument("--plain", action="store_true",
                        help="使用无残差连接的 PlainNet-18（对比实验）")
    parser.add_argument("--no-aug", action="store_true",
                        help="关闭数据增强（对比实验）")
    parser.add_argument("--epochs", type=int, default=50,
                        help="训练轮数（默认 50）")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch 大小（默认 128）")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="初始学习率（默认 0.1）")
    args = parser.parse_args()

    # --- 自动选择设备（GPU 优先：CUDA > MPS > CPU）---
    if torch.cuda.is_available():
        device = torch.device("cuda")           # NVIDIA GPU
    elif torch.backends.mps.is_available():
        device = torch.device("mps")            # Apple Silicon GPU（M1/M2/M3/M4）
    else:
        device = torch.device("cpu")            # 纯 CPU 兜底
    print(f"Device: {device}")

    # --- 1. 加载数据 ---
    # augment=True → 训练集用数据增强（RandomCrop + Flip）
    # augment=False → 仅归一化
    train_loader, test_loader = get_cifar10(
        batch_size=args.batch_size,
        augment=not args.no_aug,
    )
    print(f"CIFAR-10: {len(train_loader.dataset):,} train, "
          f"{len(test_loader.dataset):,} test")

    # --- 2. 构建模型 ---
    # --plain → PlainNet-18（无残差），用于证明残差连接的价值
    if args.plain:
        model = PlainNet18()
        label = "PlainNet-18 (无残差连接)"
    else:
        model = ResNet18()
        label = "ResNet-18"
    print(f"Model: {label}")

    # --- 3. 训练 ---
    best_acc, history = train(
        model, train_loader, test_loader,
        epochs=args.epochs, lr=args.lr,
        device=device, label=label,
    )

    # --- 4. 保存模型和训练历史 ---
    save_dir = "./checkpoints"
    import os
    os.makedirs(save_dir, exist_ok=True)
    model_name = "plainnet18" if args.plain else "resnet18"
    torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}.pth"))
    print(f"模型已保存: {save_dir}/{model_name}.pth")


if __name__ == "__main__":
    main()
