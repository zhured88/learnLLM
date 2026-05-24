"""CIFAR-10 数据加载 + 数据增强"""

import torch
import torchvision
import torchvision.transforms as transforms


def get_cifar10(batch_size: int = 128, augment: bool = True):
    """
    返回 (train_loader, test_loader)

    augment=True: 使用数据增强（RandomCrop + RandomHorizontalFlip + Normalize）
    augment=False: 仅 Normalize（用于消融实验）
    """
    norm = transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    )

    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),      # 先 pad 到 40×40 再随机裁回 32×32
            transforms.RandomHorizontalFlip(),          # 50% 概率水平翻转
            transforms.ToTensor(),
            norm,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            norm,
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        norm,
    ])

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2,
        pin_memory=True,
    )

    return train_loader, test_loader
