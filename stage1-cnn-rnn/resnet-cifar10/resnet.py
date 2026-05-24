"""
ResNet-18 for CIFAR-10 —— 从零实现残差网络

核心概念：
  - BasicBlock：残差块，输出 = 两层卷积的结果 + 原始输入（跳跃连接）
  - PlainBlock：普通块，无跳跃连接（用于对比实验）
  - ResNet：将多个块堆叠成 4 个"阶段"，空间尺寸逐阶段减半、通道数逐阶段翻倍
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    ResNet 基础残差块：两个 3×3 卷积 + 跳跃连接

    结构：
        x ── conv1 ── BN ── ReLU ── conv2 ── BN ── + ── ReLU → out
        │                                            │
        └──── shortcut（恒等或 1×1 投影）────────────┘

    关键点：
      - 跳跃连接让梯度可以绕过卷积层直接流向前面的层 → 解决深层网络的梯度消失
      - bias=False 因为 BatchNorm 后的 β 参数已经起到了偏置的作用
    """

    expansion = 1  # 输出通道数是输入通道数的倍数（Bottleneck 中是 4）

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        """
        参数：
          in_channels:  输入通道数
          out_channels: 输出通道数
          stride:       步长，>1 时空间尺寸减半（在 layer2/3/4 的第一个块中用到）
        """
        super().__init__()

        # --- 第一个 3×3 卷积 ---
        #
        # nn.Conv2d 参数详解（小白版）：
        # ┌──────────────────┬────────────────────────────────────────────────────┐
        # │ 参数              │ 解释                                               │
        # ├──────────────────┼────────────────────────────────────────────────────┤
        # │ in_channels       │ 输入有多少层"特征图"（RGB 图=3，隐藏层=64/128..）  │
        # │ out_channels      │ 输出有多少层特征图（= 卷积核的数量）                │
        # │ kernel_size=3     │ 卷积核大小 3×3（最常用，能检测边缘/纹理等局部模式） │
        # │ stride=1          │ 滑动步长。2=每两步跳一次，输出尺寸减半              │
        # │ padding=1         │ 边缘补 1 圈 0（让输出尺寸和输入一样大）             │
        # │ bias=False        │ 不加偏置项（因为后面 BatchNorm 的 β 已经充当偏置）  │
        # └──────────────────┴────────────────────────────────────────────────────┘
        #
        # 计算公式：输出尺寸 = floor((输入尺寸 + 2*padding - kernel_size) / stride + 1)
        #   例：32×32 图片 → (32+2-3)/1+1 = 32 → 尺寸不变
        #   例：32×32 图片 → (32+2-3)/2+1 = 16 → 尺寸减半
        #
        # 参数量：kernel_size × kernel_size × in_channels × out_channels
        #   例：3×3×64×128 = 73,728 个可学习的权重数字
        #
        # stride 由外部控制：当需要减半空间尺寸时传 stride=2
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)      # 批归一化：稳定训练，允许更大学习率

        # --- 第二个 3×3 卷积 ---
        # stride 固定为 1：空间尺寸变化只在第一个卷积完成
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # --- 跳跃连接（shortcut / skip connection）---
        # 如果输入和输出的维度完全一致 → 直接加法（恒等映射）
        # 如果维度不匹配（通道数变了或空间尺寸变了）→ 用 1×1 卷积投影到正确形状
        self.shortcut = nn.Identity()  # 默认：什么都不做
        if stride != 1 or in_channels != out_channels:
            # 1×1 卷积：只改变通道数，不改变空间结构
            # kernel_size=1 意味着每个输出像素只"看"输入的同位置像素
            # 可以理解为"给每个位置独立做一次全连接变换"
            #   例：(B, 64, 16, 16) → 1×1 conv(64→128, stride=2) → (B, 128, 8, 8)
            #       通道 64→128（乘法），空间 16→8（stride=2 减半）
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：残差连接的核心公式 —— out = ReLU( F(x) + x )

        参数:
          x: (batch, in_channels, height, width)
        返回:
          out: (batch, out_channels, height', width')
        """
        identity = self.shortcut(x)  # ① 保存跳跃连接分支（恒等或投影后的 x）

        out = self.conv1(x)          # ② 第 1 个卷积
        out = self.bn1(out)          # ③ 批归一化
        out = F.relu(out)            # ④ ReLU 激活函数：负数置零，正数保留

        out = self.conv2(out)        # ⑤ 第 2 个卷积
        out = self.bn2(out)          # ⑥ 批归一化
        # ⚠️ 注意：这里不激活！要在和 identity 相加之后再激活

        out += identity              # ⑦ 残差连接：F(x) + x
        out = F.relu(out)            # ⑧ 相加后再激活
        return out


class PlainBlock(nn.Module):
    """
    普通卷积块（无跳跃连接）—— 用于对比实验

    外观和 BasicBlock 几乎完全一样，但：
      - 当维度匹配时，identity = x（存下来的仅仅是用于形状加法的副本）
      - 这条路径不会形成"梯度高速公路"
      - 对比 ResNet vs PlainNet 的实验可以证明：加速训练的是跳跃连接，不是多出来的结构
    """

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 维度不匹配时仍然需要 1×1 投影让加法维度对齐
        # 但这仅仅是为了形状一致，不提供梯度高速公路
        self.use_proj = stride != 1 or in_channels != out_channels
        if self.use_proj:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.proj(x) if self.use_proj else x  # 无跳跃连接，只是形状对齐

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity              # 加法只是为了让输出形状一致
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """
    ResNet-18 主干网络，适配 CIFAR-10（32×32 小图片）

    和标准 ImageNet ResNet 的区别：
      - 首层 7×7 卷积 → 3×3 卷积（图片太小了，7×7 会丢掉太多信息）
      - 去掉首层 maxpool（CIFAR 的 32×32 不能再压缩了）

    4 个阶段的演化：
      layer1: 32×32,  64 通道  ← 高分辨率，小通道数
      layer2: 16×16, 128 通道  ← 空间减半，通道翻倍
      layer3:  8×8,  256 通道
      layer4:  4×4,  512 通道  ← 低分辨率，大通道数（参数集中在这里）
    """

    def __init__(self, block, num_blocks: list, num_classes: int = 10,
                 use_skip: bool = True):
        """
        参数：
          block:      残差块类（BasicBlock 或 PlainBlock）
          num_blocks: 每个阶段的块数量，ResNet-18 = [2, 2, 2, 2]
          num_classes: 分类数（CIFAR-10 = 10）
          use_skip:   True=用 BasicBlock（有跳跃连接），False=用 PlainBlock
        """
        super().__init__()
        # 一行代码切换残差/普通网络 —— 这就是 use_skip 参数的全部作用
        BlockCls = BasicBlock if use_skip else PlainBlock
        self.in_channels = 64  # 类变量，在构建各层时被逐步更新

        # --- CIFAR-10 专用首层：3×3 卷积，stride=1，无 maxpool ---
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                               padding=1, bias=False)  # 输入 3 通道 RGB → 64 通道
        self.bn1 = nn.BatchNorm2d(64)

        # --- 4 个"阶段"，每个阶段包含多个残差块 ---
        self.layer1 = self._make_layer(BlockCls, 64,  num_blocks[0], stride=1)   # 32×32, 64 通道
        self.layer2 = self._make_layer(BlockCls, 128, num_blocks[1], stride=2)   # 16×16, 128 通道
        self.layer3 = self._make_layer(BlockCls, 256, num_blocks[2], stride=2)   # 8×8,   256 通道
        self.layer4 = self._make_layer(BlockCls, 512, num_blocks[3], stride=2)   # 4×4,   512 通道

        # --- 分类头 ---
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应池化：不管输入多大都输出 1×1
        self.fc = nn.Linear(512 * block.expansion, num_classes)  # 全连接：从 512 特征 → 10 个类别得分

    def _make_layer(self, block_cls, out_channels: int,
                    num_blocks: int, stride: int) -> nn.Sequential:
        """
        构建一个"阶段"（多个残差块组成的 Sequential）

        每个阶段的第 1 个块负责维度转换（可能改变空间尺寸和通道数），
        后续的块输入输出维度完全一致，所以 stride 固定为 1。

        参数：
          block_cls:    用 BasicBlock 还是 PlainBlock
          out_channels: 该阶段输出通道数
          num_blocks:   该阶段包含几个残差块
          stride:       第 1 个块的步长（决定是否减半空间尺寸）
        """
        layers = []
        # 第 1 个块：处理维度转换
        layers.append(block_cls(self.in_channels, out_channels, stride))
        self.in_channels = out_channels  # 更新类变量：后续块用新通道数

        # 第 2~N 个块：维度已对齐，stride 固定为 1
        for _ in range(1, num_blocks):
            layers.append(block_cls(self.in_channels, out_channels, stride=1))

        return nn.Sequential(*layers)  # * 号把列表展开为 Sequential 的多个参数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播 —— 维度变化追踪：

        输入:  (B,  3, 32, 32)    CIFAR-10 彩色图片
        conv1: (B, 64, 32, 32)    通道 3→64，空间不变
        layer1:(B, 64, 32, 32)    stride=1，空间不变
        layer2:(B,128, 16, 16)    stride=2，空间减半，通道翻倍
        layer3:(B,256,  8,  8)    继续减半+翻倍
        layer4:(B,512,  4,  4)    最终 4×4 特征图
        avgpool:(B,512,  1,  1)   池化成单个像素
        fc:    (B, 10)            10 分类得分（logits）
        """
        out = F.relu(self.bn1(self.conv1(x)))  # 首层：卷积 → BN → ReLU
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)                # (B, 512, 4, 4) → (B, 512, 1, 1)
        out = out.view(out.size(0), -1)        # (B, 512, 1, 1) → (B, 512)  展平
        return self.fc(out)                    # (B, 512) → (B, 10)


def ResNet18(num_classes: int = 10) -> ResNet:
    """
    标准 ResNet-18
    - 18 层 = 1（首层卷积）+ 4×2×2（4 个阶段 × 每层 2 卷积 × 每阶段 2 块）+ 1（全连接） = 18
    - [2, 2, 2, 2] 表示 4 个阶段各包含 2 个 BasicBlock
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, use_skip=True)


def PlainNet18(num_classes: int = 10) -> ResNet:
    """
    无残差连接的 18 层普通网络（用于对比实验）
    - 结构和 ResNet-18 完全一致，只是没有跳跃连接
    - 在 18 层这个深度，PlainNet 的梯度衰减会让前端层几乎学不到东西
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, use_skip=False)
