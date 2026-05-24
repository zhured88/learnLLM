"""ResNet-18 for CIFAR-10 — 从零实现残差网络"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """ResNet 基础残差块：两个 3×3 卷积 + 跳跃连接"""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 跳跃连接：如果维度不匹配，用 1×1 卷积对齐
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity          # 残差连接：F(x) + x
        out = F.relu(out)
        return out


class PlainBlock(nn.Module):
    """普通卷积块（无跳跃连接），用于对比实验"""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 无跳跃连接，但维度不匹配时仍需 1×1 卷积让形状一致
        self.use_proj = stride != 1 or in_channels != out_channels
        if self.use_proj:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.proj(x) if self.use_proj else x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity          # 只做加法不加跳跃连接的话退化为 PlainNet
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet-18 适配 CIFAR-10（32×32 输入）"""

    def __init__(self, block, num_blocks: list, num_classes: int = 10,
                 use_skip: bool = True):
        super().__init__()
        BlockCls = BasicBlock if use_skip else PlainBlock
        self.in_channels = 64

        # CIFAR-10 适配：3×3 conv，stride 1，无 maxpool（图片太小）
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(BlockCls, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(BlockCls, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(BlockCls, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(BlockCls, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block_cls, out_channels: int,
                    num_blocks: int, stride: int) -> nn.Sequential:
        layers = []
        layers.append(block_cls(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(block_cls(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)


def ResNet18(num_classes: int = 10) -> ResNet:
    """标准 ResNet-18"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, use_skip=True)


def PlainNet18(num_classes: int = 10) -> ResNet:
    """无残差连接的 18 层普通网络（用于对比实验）"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, use_skip=False)
