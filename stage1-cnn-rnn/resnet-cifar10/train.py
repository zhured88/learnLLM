"""
训练循环 —— ResNet CIFAR-10

包含三个函数：
  train_epoch(): 训练一轮（前向 + 反向 + 参数更新）
  evaluate():    测试一轮（只前向，不更新参数，也不记录梯度）
  train():       总控函数（组装优化器、调度器、循环调用上面两个函数）
"""

import time
import torch
import torch.nn as nn


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple:
    """
    训练一轮 = 遍历一遍训练集的所有 batch，每个 batch 做：
      ① 清空梯度
      ② 前向传播（算预测 + loss）
      ③ 反向传播（算梯度）
      ④ 更新参数

    返回：(平均 loss, 准确率)
    """
    model.train()  # ★ 关键！切换到训练模式（BatchNorm 用当前 batch 统计量，Dropout 开始丢弃）
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        # --- 把数据从 CPU 搬到 GPU 显存 ---
        images, labels = images.to(device), labels.to(device)

        # --- ① 清空上一轮累积的梯度 ---
        # PyTorch 默认梯度是累加的（grad += new_grad），不清零会导致梯度越来越大
        optimizer.zero_grad()

        # --- ② 前向传播 ---
        outputs = model(images)              # (B, 10) logits
        loss = criterion(outputs, labels)    # scalar cross-entropy loss

        # --- ③ 反向传播 ---
        loss.backward()  # autograd 引擎自动沿计算图反向追踪，算出所有参数的 .grad

        # --- ④ 参数更新 ---
        optimizer.step()  # W -= lr * dW  （SGD + Momentum）

        # --- 累计统计量 ---
        # loss.item() 把 0 维 tensor 转成 Python float；乘以 batch size 用于最后算加权平均
        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)            # 取每行最大值索引 = 预测类别
        correct += preds.eq(labels).sum().item()  # eq() 逐元素比较 → True 求和 → 正确个数
        total += images.size(0)              # 累计总样本数

    return total_loss / total, correct / total


@torch.no_grad()  # ★ 测试时关闭 autograd，省显存 + 加速推理
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """
    测试一轮：只前向传播，不反向、不更新参数

    返回：(平均 loss, 准确率)
    """
    model.eval()  # ★ 切换到测试模式（BatchNorm 用移动平均统计量，Dropout 停止丢弃）
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    epochs: int = 50,
    lr: float = 0.1,
    weight_decay: float = 5e-4,     # L2 正则化系数：把参数往 0 方向拉，防止过拟合
    device: torch.device = None,
    lr_milestones: tuple = (25, 40),  # 在第 25/40 轮时学习率降 10 倍
    label: str = "",
):
    """
    总控训练函数：
      - 自动选择设备（GPU > CPU）
      - 组装 SGD + MultiStepLR 调度器
      - 逐轮训练 → 测试 → 打印

    参数：
      epochs:         总训练轮数
      lr:             初始学习率（SGD 通常用 0.1）
      weight_decay:   L2 正则化系数（5e-4 是常用值）
      lr_milestones:  在哪几轮降低学习率
      label:          打印用的名字（如 "ResNet-18"）
    """
    # --- 自动选择设备（GPU 优先：CUDA > MPS > CPU）---
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    model = model.to(device)  # 把模型参数搬到 GPU

    # --- 损失函数：CrossEntropyLoss 内部已包含 softmax，所以 outputs 可以传 raw logits ---
    criterion = nn.CrossEntropyLoss()

    # --- 优化器：SGD + Nesterov 动量 + L2 正则化 ---
    # Nesterov 动量比普通动量收敛更快（看一步"前瞻"梯度）
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9,
        weight_decay=weight_decay, nesterov=True,
    )

    # --- 学习率调度器：阶梯式下降 ---
    # 训练初期大步幅快速接近最优点，后期小步幅精细搜索
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=lr_milestones, gamma=0.1,  # 每到一个 milestone，lr *= 0.1
    )

    # --- 训练日志表头 ---
    print(f"\n{'='*60}")
    print(f"  {label or 'Training'}")
    print(f"  Device: {device}, Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*60}")
    print(f"{'Epoch':>5} {'Train Loss':>12} {'Train Acc':>10} "
          f"{'Test Loss':>12} {'Test Acc':>10} {'LR':>8} {'Time':>8}")
    print("-" * 72)

    # 记录每轮指标，供后续可视化
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": [], "lr": []}

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # 训练一轮
        train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                            optimizer, device)
        scheduler.step()  # 更新学习率（每轮调用一次，别在 batch 循环里调）
        current_lr = scheduler.get_last_lr()[0]

        # 测试一轮
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        elapsed = time.time() - t0
        print(f"{epoch:>5} {train_loss:>12.4f} {train_acc:>9.2%} "
              f"{test_loss:>12.4f} {test_acc:>9.2%} {current_lr:>8.1e} {elapsed:>7.1f}s")

        # 记录历史
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["lr"].append(current_lr)

        if test_acc > best_acc:
            best_acc = test_acc

    print(f"\nBest Test Accuracy: {best_acc:.2%}")
    return best_acc, history
