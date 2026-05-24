"""
CIFAR-10 数据加载 + 数据增强

数据增强（Data Augmentation）原理：
  训练时对每张图片做随机变换（平移、翻转、归一化），
  让模型每轮看到"略有不同"的版本 → 相当于免费增加了数据量 → 泛化能力更强。

  测试时不增强（做确定性变换），保证评估结果可复现。
"""

import torch
import torchvision
import torchvision.transforms as transforms


def get_cifar10(batch_size: int = 128, augment: bool = True):
    """
    返回 (train_loader, test_loader)

    参数：
      batch_size: 每个 batch 的图片数（128 是常用值，显存不够就调小）
      augment:    True = 使用数据增强（用于正式训练）
                  False = 仅归一化（用于消融实验，观察增强的效果）
    """

    # --- 归一化参数：CIFAR-10 全体训练集的 RGB 均值和标准差 ---
    # 这 6 个数字是预先算好的，标准化后每个通道变成均值 0、方差 1 的分布
    # 换数据集（如 ImageNet、MNIST）必须重算这组数字！
    norm = transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),  # R, G, B 各自的均值
        std=(0.2470, 0.2435, 0.2616),   # R, G, B 各自的标准差
    )

    # --- 训练集的数据变换 ---
    if augment:
        train_transform = transforms.Compose([
            # ① RandomCrop(32, padding=4)：先 pad 到 40×40，再随机裁回 32×32
            #    效果：模拟平移，猫在左边/右边对模型应该是同一只猫
            transforms.RandomCrop(32, padding=4),

            # ② RandomHorizontalFlip：50% 概率左右翻转
            #    效果：猫朝左/朝右对分类没影响 → 免费把数据翻了一倍
            transforms.RandomHorizontalFlip(),

            # ③ 转 Tensor：PIL Image → PyTorch Tensor，像素 0-255 → 0.0-1.0
            transforms.ToTensor(),

            # ④ 标准化：(x - mean) / std，每通道独立做
            #    效果：所有特征统一尺度 → 训练更稳定
            norm,
        ])
    else:
        # 不增强时只做最基本的转换
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            norm,
        ])

    # --- 测试集的数据变换（始终不做增强）---
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        norm,
    ])

    # --- 下载并加载数据集 ---
    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
    )

    # --- 包装为 DataLoader ---
    # shuffle=True:  训练时打乱顺序，防止模型记住样本顺序而非学习特征
    # num_workers=2: 用 2 个子进程异步加载数据，减少 GPU 等待时间
    # pin_memory=True: 锁页内存 → CPU→GPU 数据传输更快
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2,
        pin_memory=True,
    )

    return train_loader, test_loader
