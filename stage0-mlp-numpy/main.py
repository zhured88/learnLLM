#!/usr/bin/env python3
"""
阶段 0 · Mini 项目：纯 numpy 实现 2 层 MLP，在 MNIST 上训练到 95%+ 准确率

核心目标：
  1. 理解前向传播的计算过程
  2. 手写反向传播（链式法则的工程实现）
  3. 理解 SGD + Momentum 的更新逻辑
  4. 建立对 "Tensor 操作" 的肌肉记忆

运行：
  python main.py
"""

import sys
import os

# 将 src 加入路径
sys.path.insert(0, os.path.dirname(__file__))

from src.data import load_mnist
from src.model import MLP
from src.train import train


def main():
    print("=" * 52)
    print("  阶段 0：纯 NumPy 实现 2 层 MLP · MNIST 分类")
    print("=" * 52)

    # 超参数
    HIDDEN_DIM = 256
    EPOCHS = 20
    BATCH_SIZE = 64
    LR = 0.1
    MOMENTUM = 0.9

    print(f"\n超参数: hidden={HIDDEN_DIM}, epochs={EPOCHS}, "
          f"batch={BATCH_SIZE}, lr={LR}, momentum={MOMENTUM}\n")

    # 1. 加载数据
    print("[1/3] 加载 MNIST 数据...")
    train_X, train_y, test_X, test_y = load_mnist()
    print(f"  训练集: {train_X.shape}, 测试集: {test_X.shape}")

    # 2. 构建模型
    print("[2/3] 构建 2 层 MLP...")
    model = MLP(input_dim=784, hidden_dim=HIDDEN_DIM, num_classes=10)
    total_params = model.total_params
    print(f"  参数量: {total_params:,}")

    # 3. 训练
    print("[3/3] 开始训练...\n")
    train(
        model, train_X, train_y, test_X, test_y,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        lr=LR, momentum=MOMENTUM,
    )

    # 最终测试
    from src.train import compute_accuracy
    final_logits = model.forward(test_X)
    final_acc = compute_accuracy(final_logits, test_y)
    print(f"\n最终测试准确率: {final_acc:.2%}")

    if final_acc >= 0.95:
        print("✓ 达成目标：测试准确率 >= 95%")
    else:
        print(f"✗ 未达标 (差 {0.95 - final_acc:.2%})，尝试调大 hidden_dim 或 epochs")


if __name__ == "__main__":
    main()
