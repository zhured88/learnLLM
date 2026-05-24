# 阶段 0 动手练习 · 参考答案

> 配套 [TECHNICAL_GUIDE.md](./TECHNICAL_GUIDE.md) 第 6 章。每个练习包含：原理 → 代码 → 预期结果 → 你能学到什么。

---

## 目录

- [练习 1：ReLU 换成 Sigmoid](#练习-1relu-换成-sigmoid)
- [练习 2：增加第 3 层 Linear(256→128)](#练习-2增加第-3-层-linear256128)
- [练习 3：手写 Dropout 层](#练习-3手写-dropout-层)
- [练习 4：实现 Adam 优化器](#练习-4实现-adam-优化器)
- [练习 5：纯 SGD（batch_size=1）](#练习-5纯-sgdbatch_size1)
- [练习 6：打印梯度范数](#练习-6打印梯度范数)

---

## 练习 1：ReLU 换成 Sigmoid

**难度**：⭐　　**改文件**：`layers.py`、`model.py`

### 原理

ReLU 的成功很大程度上来自它的导数简单——正数区域梯度恒为 1，不存在饱和区。Sigmoid 的问题是两端饱和：

$$\sigma'(x) = \sigma(x)(1 - \sigma(x))$$

当输入 $|x| > 4$ 时，$\sigma'(x) \approx 0$，梯度几乎消失。深层网络里这会累积成"梯度消失"，导致靠前的层几乎学不到东西。

### 代码改动

**Step 1**：在 `layers.py` 末尾添加 Sigmoid 类：

```python
class Sigmoid:
    """Sigmoid 激活: 1 / (1 + e^{-x})"""

    def __init__(self):
        self.out: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # 数值稳定版：x>0 时用 e^{-x} 计算，避免 e^x 溢出
        self.out = np.where(
            x >= 0,
            1.0 / (1.0 + np.exp(-x)),
            np.exp(x) / (1.0 + np.exp(x)),
        )
        return self.out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * self.out * (1.0 - self.out)
```

**Step 2**：修改 `layers.py` 的 `Linear.__init__`，把 Kaiming 改成 Xavier：

```python
# 原来（适配 ReLU）
self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)

# 改为（适配 Sigmoid/tanh）
self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / (in_features + out_features))
```

**Step 3**：修改 `model.py`，把 `ReLU` 替换为 `Sigmoid`：

```python
from .layers import Linear, Sigmoid   # 改这里

class MLP:
    def __init__(self, input_dim=784, hidden_dim=256, num_classes=10):
        self.fc1 = Linear(input_dim, hidden_dim)
        self.act = Sigmoid()           # 改这里
        self.fc2 = Linear(hidden_dim, num_classes)

    def forward(self, x):
        h = self.fc1.forward(x)
        h = self.act.forward(h)        # 改这里
        return self.fc2.forward(h)

    def backward(self, dout):
        d = self.fc2.backward(dout)
        d = self.act.backward(d)       # 改这里
        self.fc1.backward(d)
```

### 预期结果

| 指标 | ReLU（原版） | Sigmoid |
|------|-------------|---------|
| 第 1 轮测试准确率 | ~96% | ~85-90% |
| 第 10 轮测试准确率 | ~98% | ~96-97% |
| 收敛速度 | 快 | 明显更慢 |
| 最终 loss | ~0.003 | ~0.05-0.1 |

**为什么 Sigmoid 也能到 96%+？** 因为这个网络只有 2 层，梯度消失还不严重。如果把网络加到 10 层，Sigmoid 会彻底训不动——这就是当年深度网络用 sigmoid 的困境。

### 你能学到什么

1. **激活函数的选择决定训练速度**。ReLU 不是"更好"，而是"不卡梯度"。
2. **初始化必须和激活函数配对**。Kaiming→ReLU，Xavier→Sigmoid/tanh，换一个不换另一个会放大问题。
3. **浅层网络对激活函数不敏感**。验证一个 trick 时，堆深网络才能真正暴露差异。

---

## 练习 2：增加第 3 层 Linear(256→128)

**难度**：⭐　　**改文件**：`model.py`

### 原理

两层 MLP 已经是通用逼近器了，为什么还要加层？答案是**表征效率**——同样的表达力，深层网络需要的参数更少，且每一层学到的特征更抽象。

加一层 `fc3(128, 10)`，同时把 `fc2(256, 128)`。这其实是个降维金字塔：784→256→128→10。

### 代码改动

修改 `model.py` 的 `MLP` 类：

```python
class MLP:
    """三层 MLP: 784 → 256 → 128 → 10"""

    def __init__(self, input_dim=784, hidden_dim=256, num_classes=10):
        self.fc1 = Linear(input_dim, hidden_dim)    # 784 → 256
        self.relu1 = ReLU()
        self.fc2 = Linear(hidden_dim, 128)           # 256 → 128  (新增)
        self.relu2 = ReLU()                          # (新增)
        self.fc3 = Linear(128, num_classes)          # 128 → 10   (新增)

    def forward(self, x):
        h = self.fc1.forward(x)
        h = self.relu1.forward(h)
        h = self.fc2.forward(h)       # 新增
        h = self.relu2.forward(h)     # 新增
        return self.fc3.forward(h)    # 由 fc2 改为 fc3

    def backward(self, dout):
        d = self.fc3.backward(dout)   # 由 fc2 改为 fc3
        d = self.relu2.backward(d)    # 新增
        d = self.fc2.backward(d)      # 新增
        d = self.relu1.backward(d)
        self.fc1.backward(d)

    @property
    def linear_layers(self):
        return [self.fc1, self.fc2, self.fc3]   # 新增 fc3

    @property
    def total_params(self):
        return sum(l.W.size + l.b.size for l in self.linear_layers)
```

参数量变化：

| 层 | 旧参数 | 新参数 |
|----|--------|--------|
| fc1 | 200,960 | 200,960 |
| fc2 | 2,570 | 256×128+128 = 32,896 |
| fc3 | — | 128×10+10 = 1,290 |
| **总计** | **203,530** | **235,146** |

只多了 3 万参数（+15%）。

### 预期结果

| 指标 | 2 层（原版） | 3 层 |
|------|------------|------|
| 最终准确率 | ~98.3% | ~98.0-98.5% |
| 过拟合程度 | 轻微（1.6% gap） | 可能稍大 |

**关键观察**：准确率提升可能很小甚至略降。这说明对于 MNIST 这种简单任务，2 层的表达能力已经足够——加层不会带来明显收益，只会增加训练成本。**不是越深越好**。

### 你能学到什么

1. **模型容量 vs 任务复杂度要匹配**。MNIST 不需要 3 层，加层可能适得其反。
2. **深层网络的价值在复杂任务上体现**。同样的 3 层结构放到 CIFAR-10 上，会比 2 层有明显提升。
3. 当你看到一个模型架构，先问"这个深度是任务需要，还是跟风的"。

---

## 练习 3：手写 Dropout 层

**难度**：⭐⭐　　**改文件**：`layers.py`、`model.py`、`train.py`

### 原理

Dropout 是 Hinton 在 2012 年提出的正则化技术。训练时随机"关掉"一部分神经元（输出置零），强制网络不能依赖任何单个神经元——每个神经元都得学会和任意子集协作。

**训练时**：每个神经元以概率 $p$ 保留，以 $1-p$ 关闭。
**测试时**：不做随机，但所有权重乘以 $p$ 以保证输出期望一致。

### 代码改动

**Step 1**：在 `layers.py` 末尾添加：

```python
class Dropout:
    """训练时随机丢弃神经元，测试时缩放输出"""

    def __init__(self, p: float = 0.5):
        """
        p: 保留概率（keep probability），而非丢弃概率。
           0.5 表示每个神经元有 50% 几率存活。
        """
        self.p = p
        self.mask: np.ndarray | None = None
        self.training: bool = True   # 由外部控制

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.training:
            self.mask = (np.random.rand(*x.shape) < self.p) / self.p
            return x * self.mask
        else:
            return x  # 测试时不操作（训练时已通过 /p 缩放）

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.training:
            return dout * self.mask
        return dout
```

**注意**：mask 里除以了 `self.p`（inverted dropout），这样测试时直接不做任何操作。这是现代框架的通用做法。

**Step 2**：修改 `model.py`，在激活之后插入 Dropout：

```python
from .layers import Linear, ReLU, Dropout

class MLP:
    def __init__(self, input_dim=784, hidden_dim=256, num_classes=10,
                 dropout_p=0.5):
        self.fc1 = Linear(input_dim, hidden_dim)
        self.relu1 = ReLU()
        self.dropout1 = Dropout(p=dropout_p)          # 新增
        self.fc2 = Linear(hidden_dim, num_classes)

    def forward(self, x):
        h = self.fc1.forward(x)
        h = self.relu1.forward(h)
        h = self.dropout1.forward(h)                  # 新增
        return self.fc2.forward(h)

    def backward(self, dout):
        d = self.fc2.backward(dout)
        d = self.dropout1.backward(d)                 # 新增
        d = self.relu1.backward(d)
        self.fc1.backward(d)

    def set_training(self, mode: bool):
        """切换训练/测试模式"""
        for attr in vars(self).values():
            if isinstance(attr, Dropout):
                attr.training = mode
```

**Step 3**：修改 `train.py`，训练时 `model.set_training(True)`，测试时 `model.set_training(False)`：

```python
def train(model, train_X, train_y, test_X, test_y, ...):
    ...
    for epoch in range(1, epochs + 1):
        model.set_training(True)          # 训练模式
        for X_batch, y_batch in get_batches(train_X, train_y, batch_size):
            ...

        model.set_training(False)         # 测试模式
        test_logits = model.forward(test_X)
        test_acc = compute_accuracy(test_logits, test_y)
        ...
```

### 预期结果

| 指标 | 无 Dropout | Dropout(p=0.5) |
|------|-----------|-----------------|
| 训练准确率 | 100% | ~98-99% |
| 测试准确率 | ~98.3% | ~98.0-98.5% |
| 过拟合 gap | ~1.7% | ~0.5-1.0% |
| 收敛速度 | 快 | 略慢 |

**关键观察**：Dropout 缩小了训练集和测试集之间的准确率差距（过拟合 gap），但在这个浅层网络上效果不显著。在更深的网络或更小的数据集上，Dropout 的效果会非常明显。

### 你能学到什么

1. **Dropout 是一种"廉价集成学习"**。每次前向传播训练的是不同的子网络，测试时相当于几百个子网络的投票平均。
2. **p=0.5 是经验值，不是定数**。输入层通常用 p=0.8（保留更多信息），隐藏层用 0.5。
3. **训练/测试模式切换**是正则化技术的通用模式——BatchNorm 也是同样的"训练一种行为，测试另一种行为"。

---

## 练习 4：实现 Adam 优化器

**难度**：⭐⭐　　**改文件**：`optim.py`、`train.py`

### 原理

Adam 融合了两个优化思想：
- **Momentum**：累积历史梯度方向，抑制震荡（一阶矩 $m_t$）
- **RMSProp**：自适应学习率，大梯度→小步，小梯度→大步（二阶矩 $v_t$）

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t} \quad \text{（偏差修正）}$$
$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

**偏差修正**是 Adam 的精髓。$m_t$ 和 $v_t$ 初始化为 0，前几步会因为 $\beta$ 接近 1 而严重偏小。除以 $1 - \beta^t$ 修正这个冷启动偏差。

### 代码改动

**Step 1**：在 `optim.py` 末尾添加：

```python
class Adam:
    """Adam 优化器: Momentum + RMSProp + 偏差修正"""

    def __init__(
        self,
        layers: list,
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.layers = layers
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0  # 时间步，用于偏差修正

        # 一阶矩（动量）
        self.m_W = [np.zeros_like(l.W) for l in layers]
        self.m_b = [np.zeros_like(l.b) for l in layers]
        # 二阶矩（自适应学习率）
        self.v_W = [np.zeros_like(l.W) for l in layers]
        self.v_b = [np.zeros_like(l.b) for l in layers]

    def step(self):
        self.t += 1
        for i, layer in enumerate(self.layers):
            # 更新一阶矩和二阶矩
            self.m_W[i] = self.beta1 * self.m_W[i] + (1 - self.beta1) * layer.dW
            self.v_W[i] = self.beta2 * self.v_W[i] + (1 - self.beta2) * (layer.dW ** 2)

            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * layer.db
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (layer.db ** 2)

            # 偏差修正
            m_hat_W = self.m_W[i] / (1 - self.beta1 ** self.t)
            v_hat_W = self.v_W[i] / (1 - self.beta2 ** self.t)

            m_hat_b = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_hat_b = self.v_b[i] / (1 - self.beta2 ** self.t)

            # 参数更新
            layer.W -= self.lr * m_hat_W / (np.sqrt(v_hat_W) + self.eps)
            layer.b -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.eps)
```

**Step 2**：修改 `train.py`，导入并创建 Adam：

```python
from .optim import SGD, Adam

def train(model, ..., optimizer_name="sgd"):
    if optimizer_name == "adam":
        optimizer = Adam(model.linear_layers, lr=lr)
    else:
        optimizer = SGD(model.linear_layers, lr=lr, momentum=momentum)
    ...
```

### 预期结果

| 指标 | SGD+Momentum | Adam |
|------|-------------|------|
| 第 1 轮测试准确率 | ~96% | ~97% |
| 第 5 轮测试准确率 | ~98% | ~98.3% |
| 收敛速度 | 基线 | 快 2-3 轮达到同等水平 |
| 最终准确率 | ~98.3% | ~98.4% |
| 对学习率敏感度 | 较敏感 | 不敏感（自适应） |

Adam 在这个任务上只快了几轮，但在 Transformer 训练中差距是决定性的——没有 Adam 的 warmup + 自适应学习率，Transformer 几乎训不动。

### 你能学到什么

1. **Adam 的默认 lr=0.001 比 SGD 的 0.1 小两个数量级**，因为它用自适应步长代替了大基础学习率。
2. **偏差修正只在前几十步有意义**。当 t=100 时，$1 - 0.9^{100} \approx 0.99997$，修正几乎无效果。
3. **$\epsilon=10^{-8}$ 不是装饰**。当梯度接近零时，防止除以零；太小会导致数值不稳定，太大削弱了自适应特性。

---

## 练习 5：纯 SGD（batch_size=1）

**难度**：⭐　　**改文件**：`train.py` 或 `main.py`

### 原理

"随机"梯度下降的极端——每次只看 1 个样本就更新参数。梯度方向极度噪声大，但理论上有更好的泛化性（因为不容易陷入尖锐的局部最优）。

### 代码改动

修改 `main.py`，把 `BATCH_SIZE` 设为 1：

```python
BATCH_SIZE = 1   # 原来 64
EPOCHS = 3        # batch_size=1 下每轮要跑 6 万次更新，建议减少轮数
```

### 预期结果

| 指标 | Mini-batch (64) | Pure SGD (1) |
|------|----------------|--------------|
| 每轮耗时 | ~1.7 秒 | ~30-40 秒 |
| Loss 曲线 | 平滑下降 | 剧烈震荡 |
| 第 1 轮准确率 | ~93% | ~85-90% |
| 收敛速度 | 快（按 epoch 算） | 慢 |
| 最终准确率 | ~98.3% | ~97%+ (需要更多 epoch) |

**每轮耗时对比**：

```
Mini-batch: 60000/64 ≈ 937 次参数更新
Pure SGD:   60000/1  = 60000 次参数更新 + 60,000 次独立的 backward

同等 epoch 下，Pure SGD 慢 30-60 倍。
```

### 你能学到什么

1. **Batch size 本质是"速度 vs 噪声"的权衡**。batch_size=1 噪声太大，batch_size=60000 收敛太慢。64-256 是经验最优区间。
2. **Loss 震荡不一定是坏事**。适度的噪声帮助跳出局部最优，这是 SGD 泛化性好的核心原因。
3. 工业界近年倾向"大批量 + 大学习率"（如 LLaMA 用 4M tokens 的 batch），靠 GPU 并行抵消大批次的计算劣势。

---

## 练习 6：打印梯度范数

**难度**：⭐　　**改文件**：`train.py`

### 原理

梯度范数（Gradient Norm）是诊断训练健康度的核心指标：

$$\|dW\|_2 = \sqrt{\sum_{i,j} dW_{ij}^2}$$

- **梯度消失**：$|dW| \rightarrow 0$，靠前的层几乎不更新 → 网络退化成了浅层网络
- **梯度爆炸**：$|dW| \rightarrow \infty$，参数一步飞出去 → loss 变成 NaN
- **健康训练**：各层梯度范数保持在同一数量级，随训练逐渐衰减

### 代码改动

在 `train.py` 的训练循环中，每隔 N 个 batch 打印一次：

```python
def train(model, ...):
    ...
    for epoch in range(1, epochs + 1):
        for batch_idx, (X_batch, y_batch) in enumerate(
            get_batches(train_X, train_y, batch_size)
        ):
            logits = model.forward(X_batch)
            loss = criterion.forward(logits, y_batch)
            dout = criterion.backward()
            model.backward(dout)

            # --- 每 200 个 batch 打印梯度范数 ---
            if batch_idx % 200 == 0:
                grad_norm1 = np.linalg.norm(model.fc1.dW)
                grad_norm2 = np.linalg.norm(model.fc2.dW)
                ratio = grad_norm1 / (grad_norm2 + 1e-12)
                print(f"  [epoch {epoch:2d}, batch {batch_idx:4d}] "
                      f"fc1.dW={grad_norm1:.6f}  fc2.dW={grad_norm2:.6f}  "
                      f"ratio={ratio:.2f}")

            optimizer.step()
            ...
```

### 预期结果

典型输出：

```
[epoch  1, batch    0] fc1.dW=0.031245  fc2.dW=0.008912  ratio=3.51
[epoch  1, batch  200] fc1.dW=0.018723  fc2.dW=0.006834  ratio=2.74
[epoch  1, batch  400] fc1.dW=0.012456  fc2.dW=0.005123  ratio=2.43
[epoch  2, batch    0] fc1.dW=0.005234  fc2.dW=0.002891  ratio=1.81
[epoch  5, batch    0] fc1.dW=0.001456  fc2.dW=0.000892  ratio=1.63
[epoch 10, batch    0] fc1.dW=0.000312  fc2.dW=0.000234  ratio=1.33
```

**解读**：

1. **fc1 的梯度始终大于 fc2**——因为输入维度更大（784 vs 256），且梯度在反向传播中经过 ReLU 会有一定衰减。
2. **梯度随训练逐渐减小**——这是正常的，模型接近最优解了。
3. **ratio 趋近于 1**——说明两层在同步收敛，没有哪层"掉队"。
4. **没有出现 NaN 或急剧变大**——梯度稳定，初始化合理。

**危险信号对照表**：

| 现象 | 原因 | 解法 |
|------|------|------|
| 梯度 < 1e-7 | 梯度消失 | 换 ReLU、减小网络深度、梯度裁剪 |
| 梯度 > 1e3 | 梯度爆炸 | 降低学习率、梯度裁剪 |
| fc1 梯度 >> fc2 | 深层梯度衰减 | 残差连接 |
| 梯度中途跳变 100x | 坏样本/bug | 检查数据 |
| 梯度 = NaN | 上溢 | 检查学习率、loss 中的 epsilon |

### 你能学到什么

1. **梯度范数是训练过程最重要的健康检查**。任何一个 debug 训练问题的场景，第一件事就是打印梯度范数。
2. **ratio 比绝对值更有信息量**。梯度会随 batch、数据分布变化，但各层之间的相对比例应该是稳定的。
3. **监控梯度是写训练代码的习惯，不是额外功能**。后续学到 RNN/LSTM 时，你会反复依赖这个诊断工具。

---

## 小结

| 练习 | 核心收获 |
|------|----------|
| 1. Sigmoid | 激活函数 + 初始化 = 配套使用，换一个必须换另一个 |
| 2. 3 层 MLP | 模型容量要匹配任务复杂度，不是越深越好 |
| 3. Dropout | 训练/测试行为分离是正则化的通用模式 |
| 4. Adam | 自适应学习率的本质是"大梯度小步，小梯度大步" |
| 5. Pure SGD | batch_size 是速度 vs 噪声的权衡，1 和 6 万是两个极端 |
| 6. 梯度范数 | 训练的第一行 debug 代码，学会看 ratio 而非绝对值 |
