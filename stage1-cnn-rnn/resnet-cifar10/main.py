#!/usr/bin/env python3
"""
阶段 1 · Mini 项目 1：PyTorch 复现 ResNet-18 · CIFAR-10 分类

核心目标：
  1. 理解残差连接为什么能让深层网络训得动
  2. 掌握 BatchNorm 在 CNN 中的用法
  3. 理解数据增强对泛化能力的影响

运行：
  python main.py           # 训练 ResNet-18（标准）
  python main.py --plain   # 训练无残差的 PlainNet-18
  python main.py --no-aug  # 关闭数据增强
"""

import argparse
import torch
from data import get_cifar10
from resnet import ResNet18, PlainNet18
from train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plain", action="store_true",
                        help="使用无残差连接的 PlainNet-18")
    parser.add_argument("--no-aug", action="store_true",
                        help="关闭数据增强")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 数据
    train_loader, test_loader = get_cifar10(
        batch_size=args.batch_size,
        augment=not args.no_aug,
    )
    print(f"CIFAR-10: {len(train_loader.dataset)} train, "
          f"{len(test_loader.dataset)} test")

    # 模型
    if args.plain:
        model = PlainNet18()
        label = "PlainNet-18 (无残差连接)"
    else:
        model = ResNet18()
        label = "ResNet-18"

    print(f"Model: {label}")

    train(
        model, train_loader, test_loader,
        epochs=args.epochs, lr=args.lr,
        device=device, label=label,
    )


if __name__ == "__main__":
    main()
