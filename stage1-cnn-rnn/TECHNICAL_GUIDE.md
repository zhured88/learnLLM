# 阶段 1 技术详解：CNN 残差网络 & RNN 文本生成

> 配套项目 `stage1-cnn-rnn/`，包含两个 Mini 项目：
> - `resnet-cifar10/` — PyTorch 复现 ResNet-18，CIFAR-10 分类
> - `lstm-text-gen/` — LSTM/GRU/RNN 字符级文本生成

---

## 目录

1. [CNN 核心概念](#1-cnn-核心概念)
   - [1.1 卷积：滑动窗口的特征检测器](#11-卷积滑动窗口的特征检测器)
   - [1.2 感受野：神经元"看到"的范围](#12-感受野神经元看到的范围)
   - [1.3 BatchNorm：让每一层输入都"温顺"](#13-batchnorm让每一层输入都温顺)
2. [ResNet 残差网络](#2-resnet-残差网络)
   - [2.1 深层网络的退化问题](#21-深层网络的退化问题)
   - [2.2 残差连接：F(x) + x](#22-残差连接fx--x)
   - [2.3 BasicBlock 逐行解读](#23-basicblock-逐行解读)
3. [RNN 循环神经网络](#3-rnn-循环神经网络)
   - [3.1 为什么需要 RNN](#31-为什么需要-rnn)
   - [3.2 BPTT：沿时间反向传播](#32-bptt沿时间反向传播)
   - [3.3 梯度消失：RNN 的阿喀琉斯之踵](#33-梯度消失rnn-的阿喀琉斯之踵)
4. [LSTM 和 GRU](#4-lstm-和-gru)
   - [4.1 LSTM：三道门控的记忆管理](#41-lstm三道门控的记忆管理)
   - [4.2 GRU：LSTM 的精简版](#42-grulstm-的精简版)
   - [4.3 为什么门控能缓解梯度消失](#43-为什么门控能缓解梯度消失)
5. [实验结果与对比](#5-实验结果与对比)
6. [动手练习](#6-动手练习-1)
7. [代码逐行讲解（小白友好版）](#7-代码逐行讲解小白友好版)
   - [7.1 ResNet-18 项目代码讲解](#71-resnet-18-项目代码讲解)
     - [7.1.1 BasicBlock：残差块的核心](#711-basicblock残差块的核心)
     - [7.1.2 PlainBlock：对照实验的关键](#712-plainblock对照实验的关键)
     - [7.1.3 ResNet 整体结构](#713-resnet-整体结构)
     - [7.1.4 训练循环逐行讲解](#714-训练循环trainpy逐行讲解)
     - [7.1.5 数据增强逐行讲解](#715-数据增强datapy逐行讲解)
   - [7.2 LSTM 文本生成项目代码讲解](#72-lstm-文本生成项目代码讲解)
     - [7.2.1 CharRNN 模型结构](#721-charrnn-模型结构)
     - [7.2.2 generate()：温度采样生成文本](#722-generate温度采样生成文本)
     - [7.2.3 BPTT 训练循环逐行讲解](#723-bptt-训练循环逐行讲解)
     - [7.2.4 字符级数据集 CharDataset](#724-字符级数据集-chardataset)

---

## 1. CNN 核心概念

### 1.1 卷积：滑动窗口的特征检测器

#### 白话先行

想象你在图片上滑动一个 3×3 的小窗格。窗格每次停下来，你计算"窗格内的像素"和"一个固定模板"的逐元素乘积之和。这个模板就是**卷积核**——它可能是一个竖线检测器、一个横线检测器、或者一个颜色过渡检测器。

#### 数学定义

$$(I * K)_{i,j} = \sum_{u=0}^{k-1} \sum_{v=0}^{k-1} I_{i+u, j+v} \cdot K_{u,v}$$

| 符号 | 含义 | 本项目中 |
|------|------|---------|
| $I$ | 输入特征图 | 32×32 CIFAR 图片 |
| $K$ | 卷积核 | 3×3，可学习的权重 |
| $k$ | 核大小 | 3 |
| 输出 | 特征图 | 对每种卷积核产生一张"热力图" |

#### 卷积和全连接的本质区别

| 特性 | 全连接 (Linear) | 卷积 (Conv2d) |
|------|----------------|---------------|
| 连接方式 | 每个输入对每个输出 | 局部连接（只看 3×3 窗口） |
| 参数共享 | 无 | 同一卷积核在整张图上复用 |
| 参数量 | 784×256 = 200,960 | 3×3×64 = 576 |
| 对平移的鲁棒性 | 差（像素移动=全新输入） | 好（卷积核不关心特征位置） |

**卷积的两个超级能力**：
1. **平移等变性**：物体从左移到右，特征图的激活也跟着平移。全连接层做不到。
2. **参数效率**：一个 3×3 的卷积核只有 9 个参数，却能检测整张图上任意位置的同一模式。

### 1.2 感受野：神经元"看到"的范围

感受野（Receptive Field）是指输出特征图上的一个像素，对应输入图像的有多大区域。

**计算规则**（简化版）：每过一层 3×3 卷积，感受野半径 +1。

```
输入图像 (32×32)
  │  第 1 层 3×3 conv: 每个输出像素"看到" 3×3 的输入
  │  第 2 层 3×3 conv: 每个输出像素"看到" 5×5 的输入
  │  第 3 层 3×3 conv: 每个输出像素"看到" 7×7 的输入
  │  ...
  │  第 N 层: 感受野 = (2N+1) × (2N+1)
  ▼
```

**这对设计的启示**：
- 浅层神经元看到的是局部纹理（边缘、角点）
- 深层神经元看到的是全局结构（物体的轮廓）
- ResNet-18 有 17 层卷积，最后一层的感受野覆盖整个 32×32 图像

### 1.3 BatchNorm：让每一层输入都"温顺"

#### 白话先行

训练过程中，前面层的参数在变，后面层收到的输入的统计分布（均值、方差）就会"漂移"。这叫 **Internal Covariate Shift**。

BatchNorm 的做法简单粗暴：**每个 mini-batch 内部，把输入强行标准化到均值 0、方差 1，然后再用可学习的参数放缩和平移回来**。

#### 数学

$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$y = \gamma \hat{x} + \beta$$

- $\mu_B$：batch 内均值
- $\sigma_B^2$：batch 内方差
- $\gamma, \beta$：可学习的放缩和平移参数（让网络可以"撤销"标准化）
- $\epsilon$：防止除零

#### 训练 vs 测试

| 阶段 | 均值/方差来源 |
|------|-------------|
| 训练 | 当前 mini-batch 的实时统计量 |
| 测试 | 训练期间累积的移动平均（running_mean, running_var） |

这就是为什么 `model.train()` 和 `model.eval()` 对 BatchNorm 至关重要——忘记切模式是新手最常见的 bug 之一。

#### BatchNorm 的四大收益

1. **允许更大的学习率**：标准化后的梯度不会因为输入尺度不同而爆炸
2. **充当正则化**：batch 统计量的噪声类似 Dropout 的效果
3. **缓解梯度消失**：ReLU + BN 的组合让梯度流经深层网络时不会衰减
4. **减少对初始化的敏感度**：标准化抹平了不良初始化的大部分影响

---

## 2. ResNet 残差网络

### 2.1 深层网络的退化问题

2015 年之前，一个反直觉的现象困扰着学界：

```
20 层网络: 训练误差 5%，测试误差 8%
56 层网络: 训练误差 12%，测试误差 15%  ← 更深的网络训练误差反而更高！
```

这不能用过拟合解释（训练误差也高了），而是**优化困难**——56 层的网络因为梯度消失/爆炸，就是训不动。

ResNet 的洞见：如果 20 层已经学到了有用的东西，56 层网络至少可以通过"恒等映射"（后面 36 层什么都不做）达到同样的效果。问题是 SGD 找不到这个恒等映射。

**解决方案**：让网络显式学习"残差"。

### 2.2 残差连接：F(x) + x

$$\text{Output} = F(x) + x$$

- $x$：输入（跳跃连接，直接送到输出）
- $F(x)$：残差函数（两层卷积学到的"修正量"）

**关键直觉**：

```
传统网络: 每一层必须学到"完整输出"
残差网络: 每一层只需学到"输出和输入的差异"

如果恒等映射是最优的，残差网络只需要把 F(x) 推到 0。
而在传统网络中，需要让权重矩阵精确学到 W=I 才能实现恒等映射，
这在 SGD 优化下几乎不可能。
```

### 2.3 BasicBlock 逐行解读

```python
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        # 两层 3×3 卷积
        self.conv1 = Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1   = BatchNorm2d(out_channels)
        self.conv2 = Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2   = BatchNorm2d(out_channels)

        # 跳跃连接：维度不匹配时用 1×1 卷积对齐
        self.shortcut = Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Sequential(
                Conv2d(in_channels, out_channels, 1, stride, bias=False),
                BatchNorm2d(out_channels),
            )
```

**为什么卷积层 `bias=False`？** 跟在 BN 后面的卷积不需要偏置——BatchNorm 的 $\beta$ 参数已经起到了偏置的作用。加两个偏置反而冗余。

**短路连接的两种情况**：
- 实线：维度匹配，直接 `out = F(x) + x`
- 虚线：维度不匹配（stride=2 导致空间尺寸减半，或通道数翻倍），用 1×1 卷积将 $x$ 投影到正确形状

**ResNet-18 的 4 个阶段**：

| 阶段 | 输出尺寸 | 通道数 | 块数 | 参数量占比 |
|------|---------|--------|------|-----------|
| conv1 | 32×32 | 64 | 1 | 忽略不计 |
| layer1 | 32×32 | 64 | 2 | 5% |
| layer2 | 16×16 | 128 | 2 | 15% |
| layer3 | 8×8 | 256 | 2 | 35% |
| layer4 | 4×4 | 512 | 2 | 45% |

**设计规律**：空间尺寸每减半，通道数翻倍，保持计算量大致恒定。大部分参数集中在最后几层（通道数大）。

---

## 3. RNN 循环神经网络

### 3.1 为什么需要 RNN

CNN 和 MLP 都假设输入是**定长向量**，且样本间独立。文本不满足这个假设：
- 句子长度可变
- 第 5 个词和第 1 个词有关联（长程依赖）

RNN 的核心思想：**维护一个"隐藏状态"向量 h，在每个时间步更新它，作为到目前为止读过的所有内容的摘要**。

$$h_t = \tanh(W_{ih} x_t + W_{hh} h_{t-1} + b)$$
$$y_t = W_{ho} h_t + b_o$$

```python
# RNN 的本质在循环结构中：
h = torch.zeros(batch, hidden_dim)
for t in range(seq_len):
    h = tanh(x[t] @ W_ih + h @ W_hh + b)   # h 同时依赖当前输入和历史状态
    output[t] = h @ W_ho + b_o
```

**关键观察**：权重矩阵 $W_{hh}$ 在整个序列上共享。这意味着 RNN 在"时间维度上做参数共享"，就像 CNN 在"空间维度上做参数共享"。

### 3.2 BPTT：沿时间反向传播

训练 RNN 的反向传播叫 **BPTT（Backpropagation Through Time）**——沿着时间轴展开计算图，逐时间步反向传播梯度。

```
Forward (seq_len=4):
  x1 → h1 → y1
        ↓
  x2 → h2 → y2
        ↓
  x3 → h3 → y3
        ↓
  x4 → h4 → y4

Backward:
  ∂L/∂y4 → ∂L/∂h4 → ∂L/∂h3 → ∂L/∂h2 → ∂L/∂h1
                      ↓  ∂L/∂W_hh 在每个时间步
                       共享权重的梯度累加
```

**实际实现**：PyTorch 的 `nn.RNN` 内部通过 `detach()` 截断梯度流，防止计算图沿时间无限展开。我们的训练代码里：

```python
hidden = hidden.detach()  # 截断梯度，每个 batch 独立
```

这称为 **Truncated BPTT**，是标准做法。

### 3.3 梯度消失：RNN 的阿喀琉斯之踵

BPTT 中，$W_{hh}$ 的梯度需要连乘 T 次：

$$\frac{\partial L}{\partial h_1} = \frac{\partial L}{\partial h_T} \cdot \prod_{t=2}^{T} \frac{\partial h_t}{\partial h_{t-1}}$$

如果 $|\frac{\partial h_t}{\partial h_{t-1}}| < 1$，连乘 T=100 次后梯度趋近于 0——模型无法学到第 1 个词和第 100 个词之间的依赖。

**这解释了为什么 vanilla RNN 形同虚设**：
- 序列长度 > 20 时，首尾信息基本断开
- 只能学到局部模式（相邻词的关系）
- 对长文本生成，RNN 会迅速"忘记"开头的内容，开始胡说八道

---

## 4. LSTM 和 GRU

### 4.1 LSTM：三道门控的记忆管理

LSTM 引入**细胞状态（Cell State）** $c_t$，作为贯穿时间轴的"信息高速公路"。三个门控制这条高速上信息的进出：

```
          遗忘门                      输入门                     输出门
    决定扔掉 c_{t-1} 的哪些部分   决定写入哪些新信息到 c_t    决定暴露 c_t 的哪些部分到 h_t

          ┌──┐                       ┌──┐                       ┌──┐
          │σ │                       │σ │                       │σ │
          └──┘                       └──┘                       └──┘
            │                          │                          │
    f_t ⊙ c_{t-1}    +         i_t ⊙ c̃_t         →        o_t ⊙ tanh(c_t)
```

**门控公式**：

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \quad \text{（遗忘门：0=全忘, 1=全留）}$$
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \quad \text{（输入门：0=忽略, 1=写入）}$$
$$\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c) \quad \text{（候选记忆）}$$
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \quad \text{（输出门）}$$
$$h_t = o_t \odot \tanh(c_t)$$

#### 白话理解

把细胞状态 $c_t$ 想象成一条传送带。遗忘门是"质检员"，决定传送带上哪些东西该扔了。输入门是"入库员"，决定哪些新东西该放上传送带。输出门是"发货员"，决定传送带上的哪些东西该打包发给下游。

**为什么 LSTM 能记住 100 步前的信息？** 因为 $c_t$ 的更新是**加法**（$c_t = f_t \odot c_{t-1} + ...$），而非 RNN 的矩阵乘法。加法不会让梯度指数衰减。

### 4.2 GRU：LSTM 的精简版

GRU 把遗忘门和输入门合并为**更新门** $z_t$，把细胞状态和隐藏状态合并为一个状态 $h_t$。只有两个门，参数更少，效果往往不输 LSTM。

$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t]) \quad \text{（更新门：新旧信息混合比例）}$$
$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t]) \quad \text{（重置门：忽略多少旧状态）}$$
$$\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t])$$
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

| 特性 | LSTM | GRU | Vanilla RNN |
|------|------|-----|-------------|
| 门数量 | 3 | 2 | 0 |
| 状态数量 | 2 (h, c) | 1 (h) | 1 (h) |
| 长程记忆 | 强 | 强 | 差 |
| 参数量 | 多 | 中 | 少 |
| 训练速度 | 慢 | 中 | 快 |

### 4.3 为什么门控能缓解梯度消失

LSTM 的细胞状态更新是**线性加法**：

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

对 $c_{t-1}$ 求导：$\frac{\partial c_t}{\partial c_{t-1}} = f_t$。

如果遗忘门 $f_t \approx 1$（这是 sigmoid 很容易做到的——让输入比较大即可），那么梯度 $\frac{\partial c_T}{\partial c_1} \approx 1$，传播 100 步也不会消失。

对比 RNN：$\frac{\partial h_t}{\partial h_{t-1}} = \text{diag}(\tanh'(\dots)) \cdot W_{hh}$，其中 $\tanh' \leq 1$，$W_{hh}$ 的元素通常也 < 1，连乘后指数衰减。

---

## 5. 实验结果与对比

### 5.1 ResNet-18 vs PlainNet-18（CIFAR-10）

运行：

```bash
# ResNet-18（有跳跃连接）
python main.py --epochs 50

# PlainNet-18（无跳跃连接）
python main.py --plain --epochs 50

# 关闭数据增强
python main.py --no-aug --epochs 50
```

| 配置 | 预期准确率 |
|------|-----------|
| ResNet-18 + 数据增强 | 93-95% |
| ResNet-18 无数据增强 | 88-91% |
| PlainNet-18 + 数据增强 | 85-89% |
| PlainNet-18 无数据增强 | 80-85% |

**差距来源**：PlainNet-18 的退化不是来自参数不够，而是来自优化困难——18 层的梯度衰减让前端层几乎学不到东西。

### 5.2 RNN vs LSTM vs GRU（莎士比亚文本生成）

运行：

```bash
python main.py --rnn lstm --epochs 20    # LSTM
python main.py --rnn gru  --epochs 20    # GRU
python main.py --rnn rnn  --epochs 20    # Vanilla RNN
```

| 模型 | 预期 Val Perplexity | 文本质量 |
|------|-------------------|---------|
| LSTM | ~1.5-2.0 | 合理的莎士比亚风格对话 |
| GRU | ~1.5-2.2 | 与 LSTM 接近 |
| RNN | ~3.0-5.0+ | 无法维持长句结构 |

**Perplexity（困惑度）是什么？** $\exp(\text{loss})$。含义是"模型在猜下一个字符时，平均要从多少个候选中选"。Perplexity=2 意味着模型平均只在 2 个候选中犹豫——非常确定。

---

## 6. 动手练习

| 序号 | 练习 | 难度 | 要改的文件 |
|------|------|------|-----------|
| 1 | 把 ResNet-18 改成 ResNet-34（每个 layer 的 block 数从 [2,2,2,2] 改为 [3,4,6,3]） | ⭐ | `resnet.py` |
| 2 | 在 PlainNet 上打印各层梯度范数，对比 ResNet，观察梯度衰减 | ⭐⭐ | `resnet.py`, `train.py` |
| 3 | 把 LSTM 的 num_layers 从 2 改为 4，观察训练是否变慢、文本质量是否提升 | ⭐ | `main.py` |
| 4 | 实现温度退火生成：temperature 从 1.5 线性降到 0.3 | ⭐⭐ | `rnn_gen.py` |
| 5 | 把莎士比亚换成唐诗数据集，训练中文 LSTM | ⭐⭐ | `data.py` |
| 6 | 实现 Gradient Clipping 的消融实验：对比 clip=0.5 / 1.0 / 无裁剪的训练曲线 | ⭐⭐ | `train.py` |

### 练习 1 提示

```python
# ResNet-18: [2, 2, 2, 2]
# ResNet-34: [3, 4, 6, 3]
def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)
```

---

## 7. 代码逐行讲解（小白友好版）

> 配合项目源码阅读。每个代码块先给出**完整代码**，再逐行解释**每一行在干什么、为什么这样写**。

### 7.1 ResNet-18 项目代码讲解

#### 7.1.1 BasicBlock：残差块的核心

```python
class BasicBlock(nn.Module):
    expansion = 1                                   # ①

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()                          # ②
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)  # ③
        self.bn1 = nn.BatchNorm2d(out_channels)     # ④
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)       # ⑤
        self.bn2 = nn.BatchNorm2d(out_channels)     # ⑥

        self.shortcut = nn.Identity()               # ⑦
        if stride != 1 or in_channels != out_channels:  # ⑧
            self.shortcut = nn.Sequential(          # ⑨
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
```

**逐行解释：**

| 行 | 代码 | 解释 |
|----|------|------|
| ① | `expansion = 1` | 输出通道数是输入通道数的几倍。BasicBlock 输入和输出通道数相同，所以是 1。在 Bottleneck（ResNet-50/101 用的块）里这个是 4，表示输出通道数要乘 4。 |
| ② | `super().__init__()` | 调用 PyTorch 的 `nn.Module` 初始化。**必须写**，否则模型无法注册参数。 |
| ③ | `Conv2d(..., bias=False)` | 第一个 3×3 卷积。`stride=stride` 控制空间尺寸是否减半，`padding=1` 保证图像边缘不丢信息。**`bias=False` 是因为后面跟着 BatchNorm**——BN 的 β 参数已经充当了偏置。 |
| ④ | `BatchNorm2d(out_channels)` | 跟在 conv1 后面的批归一化。对 4 维张量 (N, C, H, W) 在 C 维度上做标准化。 |
| ⑤ | `Conv2d(..., stride=1)` | 第二个 3×3 卷积。**无论传入的 stride 是多少，这里固定 stride=1**，因为空间尺寸变化只发生在块的第一个卷积上。 |
| ⑥ | `BatchNorm2d(out_channels)` | 第二个 BN，跟在 conv2 后面。 |
| ⑦ | `nn.Identity()` | "什么都不做"的占位层。当维度匹配时，shortcut 直接把 x 原样传给加法。 |
| ⑧ | `if stride != 1 or ...` | 判断条件：步长不为 1（空间尺寸变了）**或者** 输入输出通道数不同（通道数变了）。满足任一条件就需要用 1×1 卷积把 x 投影到正确形状。 |
| ⑨ | `nn.Sequential(...)` | 1×1 卷积 + BN。**为什么是 1×1？** 1×1 卷积不改变空间尺寸（只改变通道数），刚好用来做通道数对齐。当 stride=2 时配合步长减半空间尺寸。 |

**forward 方法逐行解释：**

```python
def forward(self, x):
    identity = self.shortcut(x)    # ① 跳跃连接分支：把 x 存下来（或投影）

    out = self.conv1(x)            # ② 第 1 个卷积
    out = self.bn1(out)            # ③ 批归一化
    out = F.relu(out)              # ④ 激活
    out = self.conv2(out)          # ⑤ 第 2 个卷积
    out = self.bn2(out)            # ⑥ 批归一化
                                    # ⚠️ 注意：BN 之后、加法之前不激活！
    out += identity                 # ⑦ 残差连接：F(x) + x
    out = F.relu(out)              # ⑧ 加法之后再激活
    return out
```

**关键设计细节**：

- **步骤 ⑥→⑦ 之间没有 ReLU**：BN 之后直接加 identity。如果在加法前激活，等于改变了 $F(x)$ 的分布，残差的意义就被破坏了。
- **ReLU 在加法之后**：`ReLU(F(x) + x)` 的语义是"修正后的值如果还是负的，就置零"。这相当于网络在说"这个修正方向不对，我完全不走这条路"。
- **步骤 ① 的 `identity` 是一条直接通往输出的"高速公路"**：梯度通过它可以无损传播到前面的层，这就是 ResNet 能训练 100+ 层的核心原因。

---

#### 7.1.2 PlainBlock：对照实验的关键

```python
class PlainBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        # ... conv1, bn1, conv2, bn2 与 BasicBlock 完全相同 ...
        self.use_proj = stride != 1 or in_channels != out_channels  # ①
        if self.use_proj:
            self.proj = nn.Sequential(                              # ②
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.proj(x) if self.use_proj else x             # ③
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity                                              # ④
        out = F.relu(out)
        return out
```

**PlainBlock 和 BasicBlock 的唯一区别**：

| 行 | 解释 |
|----|------|
| ① | 判断是否需要投影。和 BasicBlock 的条件一样。 |
| ② | 1×1 投影卷积。和 BasicBlock 完全一样。 |
| ③ | **核心区别**：当维度匹配时，`identity = x`（直接把输入存下来）。但这里的"存下来"只是做加法用的，**这个 x 没有经过任何变换**——它不会形成跨层的梯度高速公路。 |
| ④ | `out += identity` 这个加法在 PlainBlock 里只是为了让输出形状一致。**外观看上去和残差连接一模一样，但梯度流是不同的**：BasicBlock 的 identity 是建立了一条跳过卷积层的梯度通道，PlainBlock 没有这个通道。 |

**为什么要写 PlainBlock？** 为了做公平对比实验。如果 ResNet 比 PlainNet 好，你可以确认原因是"跳跃连接提供了梯度高速公路"，而不是"模型结构不同"。

---

#### 7.1.3 ResNet 整体结构

```python
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, use_skip=True):
        super().__init__()
        BlockCls = BasicBlock if use_skip else PlainBlock    # ① 一行切换残差/普通
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                               padding=1, bias=False)        # ② CIFAR-10 适配
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(BlockCls, 64,  num_blocks[0], stride=1)   # ③
        self.layer2 = self._make_layer(BlockCls, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(BlockCls, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(BlockCls, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))         # ④
        self.fc = nn.Linear(512, num_classes)
```

**逐行解释：**

| 行 | 代码 | 解释 |
|----|------|------|
| ① | `BlockCls = BasicBlock if use_skip else PlainBlock` | **一行切换残差/普通网络**。`use_skip=True` 用 BasicBlock（有跳跃连接），`False` 用 PlainBlock（无跳跃）。这是用"策略模式"做对比实验——其余所有代码完全不变。 |
| ② | `Conv2d(3, 64, 3, 1, 1)` | **CIFAR-10 专用的首层卷积**。原版 ResNet 用 7×7 卷积 + stride 2（适配 224×224 的 ImageNet）。CIFAR 图片只有 32×32，7×7 太大，改用 3×3 + stride 1 + 无 maxpool。 |
| ③ | `_make_layer(...)` | 构建一个"阶段"（包含多个残差块）。`num_blocks[0]=2` 表示 layer1 有 2 个 BasicBlock。stride=1 不改变空间尺寸。 |
| ④ | `AdaptiveAvgPool2d((1, 1))` | **自适应平均池化**。不管输入特征图是什么尺寸（4×4、8×8……），都输出 1×1。这样即使换了输入图片尺寸，也不用改全连接层的输入维度。 |

**_make_layer 详解：**

```python
def _make_layer(self, block_cls, out_channels, num_blocks, stride):
    layers = []
    # 第 1 个块：可能改变空间尺寸（stride=2）或通道数
    layers.append(block_cls(self.in_channels, out_channels, stride))  # ①
    self.in_channels = out_channels                                   # ②

    # 第 2~N 个块：维度已对齐，stride 固定为 1
    for _ in range(1, num_blocks):                                    # ③
        layers.append(block_cls(self.in_channels, out_channels, stride=1))

    return nn.Sequential(*layers)                                     # ④
```

| 行 | 解释 |
|----|------|
| ① | 第一个块承担"维度转换"任务——如果 stride=2 或通道数要翻倍，就在这里完成。 |
| ② | 更新 `self.in_channels`——后续块用新通道数。`self.in_channels` 是类变量，在构建 4 个 layer 时被逐步更新。 |
| ③ | `range(1, num_blocks)`：从 1 开始而非 0。因为第 0 个块在步骤①已经加了。以 ResNet-18 为例，`num_blocks=[2,2,2,2]`，每个 layer 的循环只执行 1 次（加第 2 个块）。 |
| ④ | `nn.Sequential(*layers)`：把列表展开为 Sequential 容器。`*` 是 Python 的列表解包操作符。 |

**前向传播的维度变化追踪：**

```
输入: (B, 3, 32, 32)     ← CIFAR-10 彩色图片
conv1:  (B, 64, 32, 32)   ← 通道 3→64，空间不变
layer1: (B, 64, 32, 32)   ← stride=1，空间不变
layer2: (B, 128, 16, 16)  ← stride=2，空间减半，通道翻倍
layer3: (B, 256, 8, 8)    ← 继续减半 + 翻倍
layer4: (B, 512, 4, 4)    ← 最终 4×4 特征图
avgpool: (B, 512, 1, 1)   ← 压成单个像素
fc: (B, 10)               ← 10 分类 logits
```

---

#### 7.1.4 训练循环（train.py）逐行讲解

```python
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()                                    # ①
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:                    # ②
        images, labels = images.to(device), labels.to(device)  # ③

        optimizer.zero_grad()                        # ④
        outputs = model(images)                      # ⑤
        loss = criterion(outputs, labels)            # ⑥
        loss.backward()                              # ⑦
        optimizer.step()                             # ⑧

        total_loss += loss.item() * images.size(0)   # ⑨
        _, preds = outputs.max(1)                    # ⑩
        correct += preds.eq(labels).sum().item()     # ⑪
        total += images.size(0)                      # ⑫

    return total_loss / total, correct / total       # ⑬
```

| 行 | 代码 | 解释 |
|----|------|------|
| ① | `model.train()` | **切换到训练模式**。BatchNorm 和 Dropout 层会根据这个标志改变行为——BN 用当前 batch 的统计量，Dropout 会随机丢弃神经元。**不写这行，BN 会用测试时的移动平均，模型几乎训不动。** |
| ② | `for images, labels in loader` | DataLoader 每次返回一个 batch 的 (图片, 标签)。`images` 形状是 `(128, 3, 32, 32)`，`labels` 是 `(128,)`。 |
| ③ | `.to(device)` | 把数据从 CPU 内存搬到 GPU 显存（或 MPS）。所有张量都必须在同一个设备上才能做运算。 |
| ④ | `optimizer.zero_grad()` | **清空上一轮的梯度缓存**。PyTorch 默认梯度是累加的（`grad += new_grad`），不先清零会导致梯度越来越大。 |
| ⑤ | `model(images)` | 前向传播。调用 `ResNet.forward()`，返回 `(B, 10)` 的 logits。 |
| ⑥ | `criterion(outputs, labels)` | 计算交叉熵 loss。PyTorch 的 `CrossEntropyLoss` 内部已包含 softmax，所以 outputs 可以传 raw logits。 |
| ⑦ | `loss.backward()` | 自动反向传播。PyTorch 的 autograd 引擎从 loss 出发，沿计算图反向追踪，给每个 `requires_grad=True` 的张量算出 `.grad`。 |
| ⑧ | `optimizer.step()` | 用算好的梯度更新参数：`W -= lr * dW`。 |
| ⑨ | `loss.item() * images.size(0)` | `.item()` 把 0 维 tensor 转成 Python 浮点数。乘以 `images.size(0)`（batch size）是为了最后算加权平均。 |
| ⑩ | `outputs.max(1)` | 沿维度 1（10 个类别）取最大值。返回 `(最大值, 索引)`，索引就是模型预测的类别。 |
| ⑪ | `preds.eq(labels).sum().item()` | `eq()` 逐元素比较，返回 `[True, False, True, ...]`。`.sum()` 统计 True 的数量。`.item()` 转 Python 整数。 |
| ⑫ | `total += images.size(0)` | 累计总样本数。为什么不用 `len(loader.dataset)`？因为最后一个 batch 可能不完整（drop_last=False 时）。 |

**evaluate 函数的关键差异：**

```python
@torch.no_grad()          # ① 关闭梯度追踪
def evaluate(model, loader, criterion, device):
    model.eval()           # ② 切换测试模式

    for images, labels in loader:
        outputs = model(images)
        # ⚠️ 没有 optimizer.zero_grad()
        # ⚠️ 没有 loss.backward()
        # ⚠️ 没有 optimizer.step()
```

| 标记 | 作用 |
|------|------|
| ① `@torch.no_grad()` | **停用 autograd**。测试时不需要梯度，关掉可以节省大量显存并加速推理。 |
| ② `model.eval()` | **切换测试模式**。BN 切换到移动平均统计量，Dropout 停止丢弃神经元。 |

**MultiStepLR 调度器详解：**

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                            weight_decay=5e-4, nesterov=True)        # ①
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=(25, 40), gamma=0.1)                       # ②

# 每轮训练后
scheduler.step()                                                      # ③
```

| 行 | 解释 |
|----|------|
| ① | SGD + Nesterov 动量 + L2 正则化。`weight_decay=5e-4` 是 L2 正则化系数——每次更新时把参数往 0 方向拉一点，防止过拟合。 |
| ② | 在第 25 和第 40 个 epoch 时将学习率乘以 0.1（$\gamma$）。从 0.1 降到 0.01，再到 0.001。 |
| ③ | **每轮调用一次** `scheduler.step()`。别在 batch 循环里调用——那是 WarmupLR 等更细粒度调度器的用法。 |

**为什么学习率要阶梯式下降？** 训练初期需要大步幅快速接近最优点，后期需要小步幅精细搜索。如果在最优点附近还用大步幅，会震荡甚至跳出最优区域。

---

#### 7.1.5 数据增强（data.py）逐行讲解

```python
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),          # ①
    transforms.RandomHorizontalFlip(),              # ②
    transforms.ToTensor(),                          # ③
    transforms.Normalize((0.4914, 0.4822, 0.4465),   # ④
                         (0.2470, 0.2435, 0.2616)),
])
```

| 行 | 操作 | 输入 → 输出 | 为什么要这样做 |
|----|------|------------|--------------|
| ① | `RandomCrop(32, padding=4)` | 32×32 → pad 到 40×40 → 随机裁回 32×32 | 模拟平移变化。猫在图片左边和右边，对模型来说应该是同一只猫。每轮看到略微不同的裁切位置，相当于变相增加了数据量。 |
| ② | `RandomHorizontalFlip()` | 50% 概率左右翻转 | 猫朝左和朝右，对 CIFAR-10 的分类来说不影响。相当于免费把数据集翻了一倍。 |
| ③ | `ToTensor()` | PIL Image → Tensor，像素值 [0, 255] → [0.0, 1.0] | PyTorch 模型只接受 Tensor 输入。 |
| ④ | `Normalize(mean, std)` | `(x - mean) / std`，每通道独立 | 这 6 个数字是 CIFAR-10 全体训练集的 RGB 均值和标准差。标准化后每个通道的分布变成均值 0、方差 1，训练更稳定。**注意**：这组数字是针对 CIFAR-10 的，换数据集必须重新算。 |

**为什么不增强测试集？** 测试时你用 `RandomCrop` 意味着每次评估结果都不一样——没法判断模型是否真的进步了。测试集只能做确定性变换（ToTensor + Normalize）。

---

### 7.2 LSTM 文本生成项目代码讲解

#### 7.2.1 CharRNN 模型结构

```python
class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512,
                 num_layers=2, rnn_type="lstm", dropout=0.3):
        super().__init__()
        self.rnn_type = rnn_type.lower()                     # ①
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim) # ②

        rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[self.rnn_type]  # ③
        self.rnn = rnn_cls(embed_dim, hidden_dim, num_layers, # ④
                           batch_first=True,
                           dropout=dropout if num_layers > 1 else 0)

        self.fc = nn.Linear(hidden_dim, vocab_size)          # ⑤
        self.dropout = nn.Dropout(dropout)                    # ⑥
```

| 行 | 代码 | 解释 |
|----|------|------|
| ① | `rnn_type.lower()` | 统一大小写。用户传 `"LSTM"` 也能正确匹配。 |
| ② | `nn.Embedding(vocab_size, embed_dim)` | **嵌入层**：把字符索引（整数 0-64）映射到 256 维连续向量。可以理解为"每个字符有一个 256 维的含义表示"。初始随机，训练过程中学会"相似字符有相似向量"。 |
| ③ | 字典映射选择 RNN 类型 | 一行代码切换三种架构：`"lstm"→nn.LSTM`，`"gru"→nn.GRU`，`"rnn"→nn.RNN`。这是"策略模式"的 Pythonic 写法。 |
| ④ | `batch_first=True` | 输入形状约定：`(batch, seq_len, features)`。默认是 `(seq_len, batch, features)`，设了这个就不用手动 transpose 了。 |
| ⑤ | `nn.Linear(hidden_dim, vocab_size)` | 把 512 维隐藏状态映射回 65 维（词表大小），每个维度是"下一个字符是 $c$ 的得分"。 |
| ⑥ | `nn.Dropout(dropout)` | 随机丢弃 30% 的神经元，防止过拟合。注意 Dropout 放在 embedding 和 RNN 输出之后、FC 之前。 |

**forward 的维度流转：**

```python
def forward(self, x, hidden=None):
    # x: (B, T)         ← 字符索引，B=64, T=100
    embed = self.dropout(self.embedding(x))   # → (B, T, 256)
    out, hidden = self.rnn(embed, hidden)    # → (B, T, 512)
    out = self.dropout(out)                  #   Dropout 正则化
    logits = self.fc(out)                   # → (B, T, 65)
    return logits, hidden
```

**每一步的 shape 变化**：

| 步骤 | 操作 | 输入 shape | 输出 shape | 含义 |
|------|------|-----------|-----------|------|
| 1 | `embedding(x)` | (64, 100) | (64, 100, 256) | 100 个字符，每个变成 256 维向量 |
| 2 | `rnn(embed, hidden)` | (64, 100, 256) | (64, 100, 512) | RNN 逐时间步处理，输出 512 维隐藏状态 |
| 3 | `fc(out)` | (64, 100, 512) | (64, 100, 65) | 每个时间步、每个样本，预测下一个字符的概率分布 |

**init_hidden：LSTM 需要两个状态**

```python
def init_hidden(self, batch_size, device):
    if self.rnn_type == "lstm":
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h0, c0)                        # LSTM 需要 (h, c) 两个状态
    else:
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
```

| 细节 | 解释 |
|------|------|
| `self.num_layers` 在维度 0 | 2 层 LSTM 就需要 2 组隐藏状态，每层独立。 |
| `batch_size` 在维度 1 | 每个样本有自己独立的隐藏状态——RNN 的记忆是"每样本独立"的。 |
| LSTM 返回元组 | LSTM 有 h（短期记忆，暴露给外部的输出）和 c（长期记忆，内部细胞状态）。GRU 和 RNN 只有一个状态。 |

---

#### 7.2.2 generate()：温度采样生成文本

这是整个项目最有趣的函数——让训练好的模型"创作"文本。

```python
def generate(self, start_str, char_to_idx, idx_to_char,
             length=200, temperature=0.8, device="cpu"):
    self.eval()                                          # ①
    with torch.no_grad():                                # ②
        # 编码起始字符串
        chars = [char_to_idx.get(c, 0) for c in start_str]  # ③
        inp = torch.tensor([chars], dtype=torch.long, device=device)

        hidden = self.init_hidden(1, device)             # ④
        _, hidden = self(inp, hidden)                    # ⑤

        result = list(start_str)
        next_char_idx = chars[-1]

        for _ in range(length):                          # ⑥
            inp = torch.tensor([[next_char_idx]], dtype=torch.long, device=device)
            logits, hidden = self(inp, hidden)

            logits = logits[0, -1] / max(temperature, 1e-8)  # ⑦
            probs = torch.softmax(logits, dim=-1)

            next_char_idx = int(torch.multinomial(probs, 1).item())  # ⑧
            result.append(idx_to_char[next_char_idx])

    return "".join(result)
```

| 行 | 代码 | 解释 |
|----|------|------|
| ① | `self.eval()` | 切换到测试模式——Dropout 停止丢弃。 |
| ② | `torch.no_grad()` | 生成时不需要反向传播，关掉 autograd 省显存。 |
| ③ | `char_to_idx.get(c, 0)` | 把起始字符串的每个字符转成索引。`.get(c, 0)` 表示遇到不认识的字符就返回 0（第一个字符的索引），作为容错处理。 |
| ④ | `init_hidden(1, device)` | batch_size=1（我们一次只生成一段文本，而不是 64 段）。 |
| ⑤ | `self(inp, hidden)` | **先喂入起始字符串**，目的是用这段文本"预热"隐藏状态。比如起始字符串是 `"ROMEO:"`，模型就可以从罗密欧的"语境"出发继续生成。丢弃 logits 的输出，只要 hidden。 |
| ⑥ | `for _ in range(length)` | 逐字符循环生成。每次拿上一个生成的字符，喂给 RNN，产生下一个字符。 |
| ⑦ | **温度采样**（见下方详解） | 控制生成文本的"创造性"。 |
| ⑧ | `torch.multinomial(probs, 1)` | **按概率分布采样**，而非取 argmax。argmax 永远返回概率最大的字符（贪心解码），生成文本会单调重复。`multinomial` 按概率采样保留了多样性。 |

**温度（temperature）如何控制创造性：**

```python
logits = logits / temperature         # temperature < 1 时放大差异
probs = softmax(logits)
```

| 温度值 | 效果 | 适用场景 |
|--------|------|----------|
| temperature → 0 | 接近 argmax，生成极保守、重复 | 需要严谨回答时 |
| temperature = 0.8 | 适度随机，保持连贯 | **本项目默认**，平衡连贯性和创造性 |
| temperature = 1.5 | 分布接近均匀，生成天马行空 | 创意写作 |
| temperature → ∞ | 完全随机，胡说八道 | 没什么用 |

**直觉**：温度调低 = "模型你确定一点，别乱选"；温度调高 = "模型你大胆一点，给冷门字符一些机会"。

---

#### 7.2.3 BPTT 训练循环逐行讲解

```python
def train_epoch(model, loader, criterion, optimizer, device, clip=1.0):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        batch_size = x.size(0)

        hidden = model.init_hidden(batch_size, device)           # ①
        if isinstance(hidden, tuple):
            hidden = tuple(h.detach() for h in hidden)           # ②
        else:
            hidden = hidden.detach()

        optimizer.zero_grad()
        logits, hidden = model(x, hidden)                        # ③

        loss = criterion(logits.view(-1, logits.size(-1)),       # ④
                         y.view(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip) # ⑤

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)
```

| 行 | 代码 | 解释 |
|----|------|------|
| ① | `init_hidden(batch_size, device)` | 每个新 batch 开始时，隐藏状态从零初始化。**为什么从零开始？** 因为每个 batch 是一个独立的文本片段，前一个 batch 的"记忆"对当前不连续。 |
| ② | `hidden.detach()` | **截断计算图！** 这是 BPTT 的关键操作。如果不 detach，计算图会从第一个 batch 一直延伸到第 N 个 batch——显存会爆炸。`detach()` 告诉 PyTorch："从这里断开，之前的梯度别往回传了"。 |
| ③ | 这里没有写 `hidden.detach()` | 因为我们是把新的 hidden 用于 loss 计算，梯度需要流过 RNN 的时间展开（在这个 batch 的 100 个时间步内）。这 100 步内梯度是连通的，这就是 BPTT 的"T"（Through Time）。 |
| ④ | `logits.view(-1, vocab_size), y.view(-1)` | **展平操作**。logits 从 `(B, T, V)` 展成 `(B×T, V)`，y 从 `(B, T)` 展成 `(B×T,)`。`-1` 表示"PyTorch 你帮我自动算这个维度的大小"。这样每个时间步的每个字符都被独立监督。 |
| ⑤ | `clip_grad_norm_(model.parameters(), 1.0)` | **梯度裁剪**：如果所有参数梯度的总范数超过 1.0，就等比例缩放到 1.0。RNN 训练时梯度偶尔会爆炸（尤其是序列中有罕见字符时），裁剪是最简单有效的防护。 |

**clip 参数的选择**：

| clip 值 | 效果 |
|---------|------|
| 0.5 | 激进裁剪，可能阻碍正常学习 |
| **1.0** | **本项目的选择**，平衡保护和自由 |
| 5.0 | 宽松，对梯度爆炸防护有限 |
| 无裁剪 | RNN/LSTM 训练时有概率在某个 batch 直接 NaN |

---

#### 7.2.4 字符级数据集（CharDataset）

```python
class CharDataset(Dataset):
    def __init__(self, text, char_to_idx, seq_len=100):
        self.seq_len = seq_len
        self.data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)  # ①

    def __len__(self):
        return len(self.data) - self.seq_len          # ②

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]       # ③
        y = self.data[idx + 1 : idx + self.seq_len + 1]  # ④
        return x, y
```

| 行 | 代码 | 解释 |
|----|------|------|
| ① | `[char_to_idx[c] for c in text]` | **整段文本转成一维长向量**。111 万字符的莎士比亚文集变成 111 万个整数。 |
| ② | `len(self.data) - self.seq_len` | 总样本数。文本长 1,000,000，seq_len=100，则有 999,900 个样本——每个样本是连续 100 个字符的滑动窗口。 |
| ③ | `x = data[idx : idx+100]` | 输入：第 idx 到 idx+99 个字符。 |
| ④ | `y = data[idx+1 : idx+101]` | **标签是输入右移一位**。RNN 语言模型的任务是"看到前面的字符，预测下一个字符"。y 和 x 完全重叠但差 1 位——x 是 `[a, b, c, d]`，y 是 `[b, c, d, e]`。 |

**为什么用滑动窗口？** 一段 100 万字符的文本，你不可能一次性塞进 RNN（显存放不下，梯度也传不动）。切成 100 字符长的窗口，既能利用长文本数据，又能控制显存。

**示例**：文本 `"Hello World"`，seq_len=4

```
样本 0: x=[H, e, l, l], y=[e, l, l, o]
样本 1: x=[e, l, l, o], y=[l, l, o,  ]
样本 2: x=[l, l, o,  ], y=[l, o,  , W]
...
```
