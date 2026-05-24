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
          │σ│                        │σ│                       │σ│
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
