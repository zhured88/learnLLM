# 阶段 0 技术详解：纯 NumPy 手写 MLP

> 配套项目 `stage0-mlp-numpy/`，从零构建一个能跑 MNIST 到 95%+ 的两层神经网络。

---

## 目录

- [阶段 0 技术详解：纯 NumPy 手写 MLP](#阶段-0-技术详解纯-numpy-手写-mlp)
  - [目录](#目录)
  - [1. 架构总览](#1-架构总览)
  - [2. 数据流：从像素到预测](#2-数据流从像素到预测)
  - [3. 逐层拆解](#3-逐层拆解)
    - [3.1 Linear（全连接层）](#31-linear全连接层)
      - [数学定义](#数学定义)
      - [前向传播代码](#前向传播代码)
      - [反向传播推导](#反向传播推导)
      - [反向传播代码](#反向传播代码)
    - [3.2 ReLU（激活函数）](#32-relu激活函数)
      - [数学定义](#数学定义-1)
      - [导数](#导数)
      - [实现](#实现)
    - [3.3 Softmax + CrossEntropyLoss（合并计算）](#33-softmax--crossentropyloss合并计算)
      - [Softmax 是什么？（类比：分蛋糕）](#softmax-是什么类比分蛋糕)
      - [CrossEntropyLoss 是什么？（类比：惩罚机制）](#crossentropyloss-是什么类比惩罚机制)
      - [为什么合并？](#为什么合并)
      - [反向传播推导（重点）](#反向传播推导重点)
      - [代码](#代码)
    - [3.4 SGD + Momentum（优化器）](#34-sgd--momentum优化器)
      - [SGD 是什么？（类比：蒙眼下山）](#sgd-是什么类比蒙眼下山)
      - [Momentum 是什么？（类比：滚雪球）](#momentum-是什么类比滚雪球)
      - [加入动量](#加入动量)
      - [代码](#代码-1)
  - [4. 反向传播全景](#4-反向传播全景)
  - [5. 关键设计决策](#5-关键设计决策)
    - [5.1 权重初始化：Kaiming 而非 Xavier](#51-权重初始化kaiming-而非-xavier)
    - [5.2 Softmax 和 CE 合并计算](#52-softmax-和-ce-合并计算)
    - [5.3 学习率衰减](#53-学习率衰减)
  - [6. 动手练习](#6-动手练习)
    - [练习 1 提示](#练习-1-提示)
  - [7. 模型可解释性分析](#7-模型可解释性分析)
    - [7.1 Saliency Map：模型在看哪里](#71-saliency-map模型在看哪里)
      - [原理](#原理)
      - [实现](#实现-1)
      - [实测结果](#实测结果)
    - [7.2 逐轮预测演变](#72-逐轮预测演变)
    - [7.3 置信度分布：危险区识别](#73-置信度分布危险区识别)
    - [7.4 混淆矩阵：错误模式分析](#74-混淆矩阵错误模式分析)
    - [7.5 权重模板：神经元学到了什么](#75-权重模板神经元学到了什么)
    - [7.6 工程启示](#76-工程启示)

---

## 1. 架构总览

```
输入图片 (784,) ──▶ fc1 (Linear) ──▶ ReLU ──▶ fc2 (Linear) ──▶ logits (10,)
                                      │                            │
                                      │    CrossEntropyLoss ◀──────┘
                                      │           │
                                      │     dout = (probs - y) / N
                                      │           │
                                      ▼           ▼
                              ◀────── 反向传播 ──────
```

**模型结构**：`Linear(784, 256) → ReLU → Linear(256, 10)`

| 组件 | 输入形状 | 输出形状 | 参数量 |
|------|---------|---------|--------|
| fc1 | (N, 784) | (N, 256) | 784×256 + 256 = 200,960 |
| ReLU | (N, 256) | (N, 256) | 0 |
| fc2 | (N, 256) | (N, 10) | 256×10 + 10 = 2,570 |
| **总计** | | | **203,530** |

---

## 2. 数据流：从像素到预测

一次完整的训练迭代包含 5 步，用伪代码表示：

```python
# 1. 取一小批数据
X_batch, y_batch = next(batch_generator)   # X: (64, 784), y: (64,)

# 2. 前向传播：算 logits
logits = model.forward(X_batch)             # (64, 10)

# 3. 算 loss（内部完成了 softmax）
loss = criterion.forward(logits, y_batch)   # 标量

# 4. 反向传播：算梯度
dout = criterion.backward()                 # ∂L/∂logits: (64, 10)
model.backward(dout)                        # 逐层算出 dW, db

# 5. 参数更新
optimizer.step()                            # W -= lr * dW
```

**核心问题**：第 4 步的 `dout` 怎么来的？为什么它是 `(probs - y_onehot) / N`？这是整个学习链路里最重要的一个推导。

---

## 3. 逐层拆解

### 3.1 Linear（全连接层）

#### 数学定义

$$
Y = XW + b
$$

| 符号 | 形状 | 含义 |
|------|------|------|
| $X$ | $(N, d_{in})$ | 输入，N 个样本 |
| $W$ | $(d_{in}, d_{out})$ | 权重矩阵 |
| $b$ | $(d_{out},)$ | 偏置（广播到每一行） |
| $Y$ | $(N, d_{out})$ | 输出 |

#### 前向传播代码

```python
def forward(self, x):
    self.x = x               # 缓存输入，反向传播要用
    return x @ self.W + self.b
```

#### 反向传播推导

链式法则告诉我们：已知 loss 对本层输出 $Y$ 的梯度 $\frac{\partial L}{\partial Y}$（变量名 `dout`），需要求：

1. **$\frac{\partial L}{\partial W}$**（更新权重用）

   $Y = XW + b$，对 $W$ 的偏导：
   $$\frac{\partial L}{\partial W} = X^T \frac{\partial L}{\partial Y}$$

   直觉：每个输入特征 $x_i$ 对梯度的贡献，是它在所有样本上的激活值乘以输出梯度。

2. **$\frac{\partial L}{\partial b}$**（更新偏置用）

   $b$ 的广播机制意味着每个样本都加上了完整偏置，梯度是各样本之和：
   $$\frac{\partial L}{\partial b} = \sum_{i=1}^{N} \frac{\partial L}{\partial Y_i}$$

3. **$\frac{\partial L}{\partial X}$**（传给上一层）

   $$\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} W^T$$

#### 反向传播代码

```python
def backward(self, dout):          # dout: (N, out_features)
    self.dW = self.x.T @ dout       # (in, out)  — X^T × dout
    self.db = dout.sum(axis=0)      # (out,)     — 沿 batch 维度累加
    return dout @ self.W.T          # (N, in)    — 传给上一层
```

**为什么缓存 `self.x`？**
反向传播计算 `dW = X^T @ dout` 时，必须知道前向传播时的输入 $X$。这是所有层的通用模式：**前向时缓存任何反向时需要的东西**。

---

### 3.2 ReLU（激活函数）

#### 数学定义

$$\text{ReLU}(x) = \max(0, x)$$

#### 导数

$$
\frac{d\text{ReLU}}{dx} =
\begin{cases}
1, & x > 0 \\
0, & x \leq 0
\end{cases}
$$

#### 实现

```python
def forward(self, x):
    self.mask = x > 0          # 记住哪些位置 > 0
    return x * self.mask       # 正数保留，负数清零

def backward(self, dout):
    return dout * self.mask    # 正数位置梯度通过，负数位置阻断
```

**为什么用 mask 而非 if-else？** 向量化。`x > 0` 产生布尔数组，与 `dout` 逐元素相乘，GPU/CPU 上都能并行。

---

### 3.3 Softmax + CrossEntropyLoss（合并计算）

> **白话先行 · 如果你只记住一句话**：Softmax 把任意的"打分"变成"概率"；CrossEntropyLoss 衡量这个概率分布和正确答案之间的"差距"。两者配合就像"打分→排名→扣分"一条龙。

#### Softmax 是什么？（类比：分蛋糕）

想象你有 10 个数字候选，模型的原始输出 `logits` 是随便写的分数，比如 `[-2.3, 0.5, 4.7, ...]`——正负不定，加起来也不是 1。

Softmax 做了三件事：
1. **放大差异**：对所有分数做 `e^x`（指数函数），好的分数会飙升，差的被压扁
2. **转成非负数**：`e^x` 永远 > 0，冲突消失
3. **归一化**：除以总和，让 10 个数加起来正好等于 1

所以 Softmax(logits) = **概率分布**。最大的那个概率 ≈ 模型当前最倾向于选哪个数字。

用生活类比：10 个人考试得了原始分（有的负分有的几百分），你对他们说："把分数转成全班占比，每人分得总分的多少%"。Softmax 干的就是这事。

#### CrossEntropyLoss 是什么？（类比：惩罚机制）

有了概率分布后，你怎么告诉模型"你做得好还是不好"？

CrossEntropyLoss 的计算公式：`loss = -log(正确类别的概率)`

这条曲线的特性是理解一切的关键：

| 模型把正确类别预测为 | 概率值 | -log(概率) | 含义 |
|---------------------|--------|-----------|------|
| 极度自信且正确 | 0.99 | 0.01 | 基本不罚 |
| 比较确定 | 0.7 | 0.36 | 轻微惩罚 |
| 犹豫 | 0.5 | 0.69 | 明显惩罚 |
| 基本不认为是正确 | 0.1 | 2.3 | 重罚 |
| 完全不认为是正确 | 0.01 | 4.6 | 极高的惩罚 |

**直觉**：模型越觉得"这不会是正确答案"，`-log(概率)` 就越大，惩罚就越痛。这符合我们想要的行为——逼模型把概率质量集中到正确答案上。

**为什么叫"交叉熵"？** 它是信息论的概念，衡量"你以为的分布"和"真实分布"的差距。你在沟通中每传递一个 bit 都要消耗能量（熵），交叉熵就是"你用错误的猜测去描述真相时多浪费的能量"。但作为初学者，理解为"惩罚机制"就足够了。

这是整个项目最重要的技术点。**这个项目里，Softmax 和 CrossEntropyLoss 永远合并使用，从不单独调用 Softmax.backward()。**

#### 为什么合并？

分开计算存在数值问题：

```python
# ❌ 危险做法：先 softmax 再 log
probs = softmax(logits)       # 如果 logits 值很大，exp 溢出
loss = -log(probs[label])     # 如果 probs ≈ 0，log 得 -inf
```

合并后用 **LogSumExp 技巧** 统一处理：

```python
# ✅ 数值稳定的做法
shifted = logits - logits.max(axis=1, keepdims=True)   # 减去最大值，防溢出
exp_shifted = np.exp(shifted)                           # 现在最大值是 e^0 = 1
probs = exp_shifted / exp_shifted.sum(axis=1, keepdims=True)
loss = -np.log(probs[correct_indices] + 1e-12).mean()
```

#### 反向传播推导（重点）

分开看：

1. **CrossEntropyLoss 对 softmax 输出的梯度**：
   $$\frac{\partial L}{\partial p_k} = -\frac{1}{p_{y_i}} \quad \text{（仅在正确类别位置）}$$

2. **Softmax 的雅可比矩阵**（$C$ 个类别的输出对输入的全微分）：
   $$\frac{\partial p_k}{\partial z_j} = p_k(\delta_{kj} - p_j)$$
   其中 $\delta_{kj}$ 是 Kronecker delta（$k=j$ 时为 1，否则为 0）。

3. **合并梯度**（链式法则连乘后化简）：

   经过 $C \times C$ 雅可比矩阵连乘后，得到一个极简结果：

   $$\boxed{\frac{\partial L}{\partial z_j} = \frac{p_j - \mathbf{1}[j = y_i]}{N}}$$

   翻译成人话：**loss 对 logit 的梯度 = (预测概率 - 真实标签的 one-hot) / batch_size**。

#### 代码

```python
def backward(self):
    batch_size = self.probs.shape[0]
    num_classes = self.probs.shape[1]
    y_onehot = np.zeros((batch_size, num_classes))
    y_onehot[np.arange(batch_size), self.y] = 1
    return (self.probs - y_onehot) / batch_size
```

**直觉解释**：
- 对正确类别（$j = y_i$）：梯度 = `(probs[j] - 1) / N`，负值表示"往更大方向推"
- 对错误类别（$j \neq y_i$）：梯度 = `probs[j] / N`，正值表示"往更小方向压"
- `1e-12` 防 `log(0)` 的微小 epsilon，不影响梯度方向

这个 3 行的 `backward()` 是神经网络训练里最常见的梯度，值得反复看透。

---

### 3.4 SGD + Momentum（优化器）

> **白话先行 · 如果你只记住一句话**：SGD 是"每次看一小撮数据，沿着错误减少最快的方向走一小步"；Momentum 是"像滚雪球一样，之前往哪里走，现在继续往那个方向加速"。

#### SGD 是什么？（类比：蒙眼下山）

想象你被蒙眼扔到一座山上，任务是走到山谷最低点。你唯一能感知的是脚下的**坡度方向**（梯度）。

SGD（随机梯度下降）的策略：
1. 伸出脚感受坡度，找到最陡的下降方向
2. 朝那个方向走一小步（步长 = 学习率 η）
3. 重复

**为什么不叫"梯度下降"而叫"随机梯度下降"？**

- **梯度下降（GD）**：每次看完整座山（全部 6 万张图片），算一个总坡度，然后走一步。准确但慢得离谱。
- **随机梯度下降（SGD）**：每次随机看一小块地（64 张图片），估计个大概坡度，立刻走一步。噪声大但极快。
- **折中 = Mini-batch SGD**：每次看 64 张，兼顾速度和稳定。本项目用的就是这个。

**学习率 η 的本质**：决定每一步跨多大。太大容易冲过头（震荡甚至发散），太小走得太慢（训练到天荒地老）。选学习率是深度学习里最重要的调参手艺活。

#### Momentum 是什么？（类比：滚雪球）

纯 SGD 有一个致命问题——在窄而深的沟壑里会左右震荡：

```
纯 SGD 的轨迹（无动量）：          SGD + Momentum 的轨迹：
    ↙↗↙↗↙↗  （锯齿形，慢）          ⬇⬇⬇  （一路冲到底，快）
```

Momentum 的解决方案：**记住之前的方向，继续加速**。

$$v_t = \underbrace{0.9 \cdot v_{t-1}}_{\text{之前累积的方向}} - \underbrace{\eta \cdot \nabla L}_{\text{当前坡度方向}}$$

- Moment = 0.9：每步保留 90% 的旧速度。在持续下降的方向（如山谷底部），累计效应让速度越来越快。
- 左右震荡的方向呢？速度正负交替，Momentum 把它们抵消掉了。

用生活类比：你推一辆购物车。每次推一下 = SGD。如果购物车本身已经在往前滑行，你再推一下它会更快——这就是 Momentum。而如果购物车在左右晃，惯性会让晃动幅度越来越小。

$$\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta L$$

#### 加入动量

动量模拟物理中的惯性——球滚下山时，梯度方向一致的地方加速，震荡的地方减速：

$$v_t = \beta v_{t-1} - \eta \cdot \nabla_\theta L$$
$$\theta_{t+1} = \theta_t + v_t$$

| 符号 | 含义 | 本项目中 |
|------|------|---------|
| $\eta$ (lr) | 学习率 | 初始 0.1 |
| $\beta$ (momentum) | 动量系数 | 0.9 |
| $v_t$ | 速度（累计梯度） | `optimizer.v_W[i]` |

#### 代码

```python
def step(self):
    for i, layer in enumerate(self.layers):
        # 速度更新：旧速度 * 0.9 - 学习率 * 新梯度
        self.v_W[i] = self.momentum * self.v_W[i] - self.lr * layer.dW
        self.v_b[i] = self.momentum * self.v_b[i] - self.lr * layer.db
        # 参数更新
        layer.W += self.v_W[i]
        layer.b += self.v_b[i]
```

**关键的实现细节**：优化器直接持有 `layer` 对象的引用，每次 `step()` 时动态读取 `layer.dW`，而非在构造时缓存梯度引用。这避免了 `backward()` 更新梯度对象后，旧引用（`None`）残留的 bug。

---

## 4. 反向传播全景

以一个 batch_size=2、单样本简化的情况，追踪梯度流动：

```
Forward:
  X(784) → fc1.forward → h1(256) → relu → h2(256) → fc2.forward → logits(10)

Backward (逆序):
  dlogits(10) ←── criterion.backward()
    │                = (probs - y_onehot) / 2
    ▼
  fc2.backward(dlogits) → dW2(256,10), db2(10,), dh2(2,256)
    │
    ▼
  relu.backward(dh2) → dh1(2,256) [mask 阻挡了负值梯度]
    │
    ▼
  fc1.backward(dh1) → dW1(784,256), db1(256,)
```

**梯度更新计算量统计**（单次 backward）：

| 操作 | 表达式 | 计算量 |
|------|--------|--------|
| fc2.dW | `h2.T @ dout` | (256×2) × (2×10) = 5,120 次乘法 |
| fc2.db | `dout.sum(0)` | 2×10 = 20 次加法 |
| fc1.dW | `X.T @ dh1` | (784×2) × (2×256) = 401,408 次乘法 |
| fc1.db | `dh1.sum(0)` | 2×256 = 512 次加法 |

**fc1 的计算量远大于 fc2**（401K vs 5K），因为 784 维输入到 256 维的映射矩阵是参数大户。

---

## 5. 关键设计决策

### 5.1 权重初始化：Kaiming 而非 Xavier

```python
self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
```

- **Xavier 初始化**：$\text{std} = \sqrt{\frac{2}{n_{in} + n_{out}}}$，假设线性激活 → 适合 tanh/sigmoid
- **Kaiming 初始化**：$\text{std} = \sqrt{\frac{2}{n_{in}}}$，考虑 ReLU 置零一半神经元 → 适合 ReLU

我们用了 ReLU，所以用 Kaiming。换 tanh 的话，第一件事就是改初始化。

### 5.2 Softmax 和 CE 合并计算

不从 Softmax 单独 `.backward()`，而是直接在 `CrossEntropyLoss` 内部完成 softmax + NLL 的联合计算，原因：
- 数值稳定：LogSumExp 技巧在合并时生效
- 计算高效：跳过 $C \times C$ 雅可比矩阵的构造（O(C²) → O(C)）

### 5.3 学习率衰减

每轮 `optimizer.lr *= 0.95`，即指数衰减。前几轮大步快走接近最优区域，后面小步精细搜索。一种简单但有效的训练技巧。

---

## 6. 动手练习

掌握这些知识后，尝试以下改动来验证理解：

| 序号 | 练习 | 难度 | 要改的文件 |
|------|------|------|-----------|
| 1 | 把 ReLU 换成 Sigmoid，观察训练是否收敛变慢 | ⭐ | `layers.py` |
| 2 | 增加第 3 层 Linear(256→128)，看准确率变化 | ⭐ | `model.py` |
| 3 | 手写 Dropout 层（训练时随机置零，测试时缩放），防过拟合 | ⭐⭐ | `layers.py` |
| 4 | 实现 Adam 优化器，对比 SGD+Momentum 的收敛速度 | ⭐⭐ | `optim.py` |
| 5 | 不用 batch，每次只用一个样本（纯 SGD），观察 loss 曲线震荡 | ⭐ | `train.py` |
| 6 | 打印每层的梯度范数 `np.linalg.norm(dW)`，观察梯度是否消失/爆炸 | ⭐ | `train.py` |

### 练习 1 提示

```python
class Sigmoid:
    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, dout):
        return dout * self.out * (1 - self.out)  # sigmoid 导数
```

同时需要把 `Linear.__init__` 的 Kaiming init 改成 Xavier：
```python
self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / (in_features + out_features))
```

---

## 7. 模型可解释性分析

> 配套脚本 `interpret.py`，每轮训练输出 10 张测试图片的预测截图到 `viz/` 目录。

### 7.1 Saliency Map：模型在看哪里

#### 原理

Saliency Map 是最简单直接的归因方法——**对输入像素求梯度**：

$$\text{Saliency}(x)_{ij} = \left| \frac{\partial \text{logit}_{target}}{\partial x_{ij}} \right|$$

直觉：梯度绝对值大的像素，说明微小变化就会剧烈影响目标类别的 logit，这些像素就是模型做判断时最依赖的区域。

#### 实现

```python
def saliency_map(model, x, target_class):
    # 前向传播
    h1 = model.fc1.forward(x)
    h1_relu = model.relu.forward(h1)
    logits = model.fc2.forward(h1_relu)

    # 构造"虚拟梯度"：只对目标类别求导
    dlogits = np.zeros_like(logits)
    dlogits[0, target_class] = 1.0

    # 反向传播到输入层
    model.fc2.backward(dlogits)
    model.relu.backward(...)
    dx = model.fc1.backward(...)  # ∂(target_logit)/∂x

    return |dx|.reshape(28, 28)
```

和训练时 backprop 的区别：训练反向传播的起点是 `(probs - y) / N`（CE loss 的梯度），而 saliency 的起点是一个 one-hot 向量（只关心目标类别）。数学上，这等价于问："输入像素往哪个方向改，能让目标类别的得分增加最多？"

#### 实测结果

正确预测时，saliency 热点集中在数字笔画的**边缘和交叉处**——这些区域确实是区分数字的关键特征。错误预测时，热点常常"跑偏"到无关背景区域，说明模型在"瞎看"。

```
正确预测 7（置信度 1.0）：              错误预测 4→9（置信度 0.86）：
高亮点在横竖笔画交界处                  热点分散，大量激活在空白区域
      .. .                                     .
    ..-=+-:.                              .     .. .
   :-:.:..                                 :.--.. .
  .:==:=.....                            . .. .:*=:...
 ...:.+=...:.                             ..::.:-::.
 .:-=.+:.::-                               ..:.:.-..
  ::.:=.-:.. .                              .:.:+-=-.-.
```

### 7.2 逐轮预测演变

`interpret.py` 每轮训练后，对 10 个固定测试样本（每个数字 1 张）生成可视化。运行：

```bash
python interpret.py
# 输出到 viz/epoch_01.png ... epoch_10.png + viz/final_report.png
```

观察点：
- **第 1 轮**：通常只有 7-8 个样本预测正确，模型还在"摸索"
- **第 3-5 轮**：大部分样本已正确，但置信度还不够高（0.7-0.9）
- **第 8-10 轮**：正确样本置信度接近 1.0，saliency 热点精确收敛到笔画

**关键的观察**：有些样本即使模型"猜对了"，saliency 热点也未必合理——这说明高准确率不等于模型真正"理解"了数字。模型可能依赖了错误的特征但"歪打正着"。

### 7.3 置信度分布：危险区识别

置信度直方图揭示了模型校准状态：

| 置信度区间 | 样本数（正确） | 样本数（错误） | 错误占比 |
|-----------|-------------|-------------|---------|
| 0.00~0.50 | 7 | 5 | **41.7%** |
| 0.50~0.70 | 33 | 42 | **56.0%** |
| 0.70~0.80 | 43 | 19 | 30.6% |
| 0.80~0.90 | 69 | 33 | 32.4% |
| 0.90~0.95 | 78 | 20 | 20.4% |
| 0.95~0.99 | 222 | 27 | 10.8% |
| 0.99~1.00 | 9374 | 22 | **0.2%** |

三条规律：

1. **低置信 = 高风险**：置信度 0.5-0.7 的区间里，超过一半的预测都是错的。如果这是生产系统，这个区间的预测应该转人工审核。

2. **高置信 ≠ 一定对**：0.99+ 置信度区间仍有 22 个错误（0.2%）。这些是模型最危险的盲点——它对自己的错误判断极度自信。

3. **模型被"惯坏了"**：94% 的正确预测落在 0.95+ 区间，因为这个任务太简单。真实场景（如医疗影像、金融风控）不会有这么漂亮的分布。

### 7.4 混淆矩阵：错误模式分析

```
最容易混淆的数字对（top 5）：
  真实 5 → 预测 3: 8 次     下半部曲线相似，人手写也容易混
  真实 4 → 预测 9: 8 次     顶部环状结构在低分辨率下难以区分
  真实 9 → 预测 4: 7 次     同上，对称混淆
  真实 8 → 预测 3: 7 次     双环 8 的顶部和 3 相似
  真实 7 → 预测 9: 7 次     7 的横线 + 竖线在模糊时像 9
```

这些混淆对与人类直觉高度一致。**模型犯的"错"不是随机的——它反映的是特征空间的真实重叠**。如果你需要优化这些混淆对，策略应该是：

- **数据增强**：对这些混淆对做针对性增强（旋转、加噪），让模型看到更多边界样本
- **hard negative mining**：训练后期专门收集混淆对样本做微调

### 7.5 权重模板：神经元学到了什么

fc1 的 256 个隐藏神经元，每一个都对应一个 28×28 的权重向量。可视化后：

```
前 10 个隐藏神经元的权重模式：

神经元 0-4:
+++++-+  +++++++  -------  +++++++  -------
+--+--+  +++++++  -++++--  +++-+++  ----+--
+--- -+  ++++-++  -+++---  +++-+++  -+-----
--#++--  ++-#+++  ---+---  +++-+--  -+-----
+-+++--  ++--+++  --  -+-  +++++++  -  ++--
------+  ++#++++  -++----  +--#-++  --+++--
+++++-+  +++++++  --+----  ++-++++  -------
```

观察到的模式：

| 模式 | 含义 |
|------|------|
| 全正权（`+++++++`） | 偏置神经元，提供全局激活基线 |
| 全负权（`-------`） | 抑制神经元，过滤无关背景 |
| 正负交替（`+--+--+`） | 边缘检测器，类似卷积网络的底层滤波器 |
| 局部热点（`#` 符号） | 特定位置的强激活，可能专为某个数字笔画设计 |

**关键洞察**：MLP 的第一层实际上在隐式地学习"卷积核"。256 个神经元里，有的擅长检测竖线，有的擅长检测横线，有的对弯曲边缘敏感——这些和 CNN 第一层学到的 Gabor 滤波器是同一类东西。区别在于 MLP 的神经元是全局连接的（每个神经元看全图），而 CNN 是局部连接的（卷积核只看一个小窗口）。

### 7.6 工程启示

1. **不要只看准确率**。98.36% 的准确率听起来很好，但 saliency 分析和置信度分布暴露了模型"高分低能"的一面——某些预测靠的是错误特征。

2. **置信度 0.5-0.7 是危险区**。这是一条可以直接落地的规则：生产系统中，置信度低于 0.7 的自动决策都应触发人工审核。

3. **混淆对是优化杠杆**。与其泛泛地加数据，不如针对 top-5 混淆对做数据增强，投入产出比更高。

4. **Saliency map 是调试利器**。当模型做出错误预测时，先看 saliency map——如果热点在背景而非数字上，说明数据预处理有问题；如果热点在正确区域但预测错了，说明是特征表征能力不足。
