# Stage 1 CNN-RNN 练习题答案

---

## 练习 1：把 ResNet-18 改成 ResNet-34

**难度**：⭐  
**要改的文件**：`resnet-cifar10/resnet.py`

### 分析

ResNet-18 和 ResNet-34 的唯一区别是每个 stage 的 BasicBlock 数量：

| 模型 | layer1 | layer2 | layer3 | layer4 | 总层数 |
|------|--------|--------|--------|--------|--------|
| ResNet-18 | 2 | 2 | 2 | 2 | 18 |
| ResNet-34 | 3 | 4 | 6 | 3 | 34 |

总层数计算：`1 (conv1) + 2×(3+4+6+3) (每个block 2层卷积) + 1 (fc) = 1 + 32 + 1 = 34`

### 答案代码

在 `resnet.py` 末尾添加工厂函数：

```python
def ResNet34(num_classes: int = 10) -> ResNet:
    """
    ResNet-34
    - 34 层 = 1（首层卷积）+ 4×2×(3+4+6+3)（4 个阶段 × 每层 2 卷积 × block 数）+ 1（全连接） = 34
    - [3, 4, 6, 3] 表示 4 个阶段分别包含 3、4、6、3 个 BasicBlock
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, use_skip=True)
```

然后在 `main.py` 中添加 `--resnet34` 参数：

```python
parser.add_argument("--resnet34", action="store_true",
                    help="使用 ResNet-34 替代 ResNet-18")
```

并在构建模型处：

```python
if args.resnet34:
    from resnet import ResNet34
    model = ResNet34()
elif args.plain:
    model = PlainNet18()
else:
    model = ResNet18()
```

### 预期效果

- 参数量：ResNet-18 约 11M → ResNet-34 约 21M
- 训练时间增加约 60-80%
- 准确率可能提升 1-3 个百分点（CIFAR-10 上 34 层和 18 层差距不大，因为图片分辨率低）
- 在 ImageNet 上 ResNet-34 比 ResNet-18 有更明显的提升

---

## 练习 2：打印 PlainNet vs ResNet 各层梯度范数

**难度**：⭐⭐  
**要改的文件**：`resnet-cifar10/resnet.py`, `resnet-cifar10/train.py`

### 分析

残差连接的核心收益是解决深层网络的梯度消失——梯度可以通过跳跃连接"无损"传播到前面的层。PlainNet 没有跳跃连接，梯度在反向传播经过 18 层卷积后会指数衰减。

我们需要：
1. 给每个 BasicBlock/PlainBlock 注册一个 hook，在反向传播后打印该 block 的梯度范数
2. 在 ResNet 的 forward 中为各层收集梯度信息
3. 修改 train.py，在特定 epoch 打印梯度统计

### 答案代码

**方案 A：在 train.py 的 train_epoch 中添加梯度范数统计（推荐，改动最小）**

修改 `train.py` 的 `train_epoch` 函数，在 `loss.backward()` 之后、`optimizer.step()` 之前插入梯度范数统计：

```python
def train_epoch(model, loader, criterion, optimizer, device, log_grads=False):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # --- 梯度范数统计（练习 2）---
        if log_grads and batch_idx == 0:  # 只打印第一个 batch，避免刷屏
            print("\n  Gradient norms by layer:")
            for name, param in model.named_parameters():
                if "weight" in name and param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    print(f"    {name:50s} | grad norm = {grad_norm:.6f}")
            print()

        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total
```

在 `train()` 函数中，第一个 epoch 调用时传入 `log_grads=True`：

```python
# 第一个 epoch 打印梯度范数
if epoch == 1:
    train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                        optimizer, device, log_grads=True)
else:
    train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                        optimizer, device)
```

### 预期观察

运行 `python main.py`（ResNet-18）和 `python main.py --plain`（PlainNet-18）后对比：

| 层 | ResNet-18 梯度范数 | PlainNet-18 梯度范数 | 分析 |
|----|-------------------|---------------------|------|
| layer1.conv1.weight | ~1e-3 | ~1e-5 或更小 | PlainNet 前端梯度几乎消失 |
| layer2.conv1.weight | ~1e-3 | ~1e-4 | 差距缩小但仍明显 |
| layer3.conv1.weight | ~1e-3 | ~1e-3 | 中间层差距不大 |
| layer4.conv1.weight | ~1e-3 | ~1e-2 | PlainNet 后端梯度反而大 |

关键发现：ResNet 各层梯度范数比较均匀（都在 ~1e-3 量级），而 PlainNet 的梯度从前到后呈指数级衰减——**前端的层几乎学不到任何东西**，这是 PlainNet 性能差的根本原因。

### 原理

反向传播中，梯度从 loss 出发，逐层向前传播。在 PlainNet 中：

$$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial h_{18}} \cdot \prod_{i=2}^{18} \frac{\partial h_i}{\partial h_{i-1}} \cdot \frac{\partial h_1}{\partial W_1}$$

连乘 18 个雅可比矩阵后，梯度指数衰减。

而在 ResNet 中，因为跳跃连接 $\text{out} = F(x) + x$：

$$\frac{\partial \text{out}}{\partial x} = \frac{\partial F(x)}{\partial x} + I$$

恒等矩阵 $I$ 提供了一条梯度高速公路，确保即使 $\frac{\partial F(x)}{\partial x}$ 很小，梯度 $I$ 仍然无损流过。

---

## 练习 3：把 LSTM 的 num_layers 从 2 改为 4

**难度**：⭐  
**要改的文件**：`lstm-text-gen/main.py`

### 分析

默认 `num_layers=2`，即堆叠 2 层 LSTM。改为 4 层构成"深度 LSTM"。每层 LSTM 的输出作为下一层的输入，层间用 Dropout 正则化。

### 答案

**方法 1：直接改命令行参数（最简单）**

```bash
python main.py --num_layers 4 --epochs 20
```

**方法 2：修改默认值**

在 `main.py` 中将 `--num_layers` 的 `default` 从 `2` 改为 `4`：

```python
parser.add_argument("--num_layers", type=int, default=4,
                    help="RNN 堆叠层数（默认 4）")
```

### 预期效果

| 指标 | num_layers=2 | num_layers=4 | 原因 |
|------|-------------|-------------|------|
| 参数量 | ~1.5M | ~2.7M | 多了 2 层 LSTM 的参数 |
| 每 epoch 时间 | ~15s | ~28s | 计算量翻倍 |
| Val Perplexity | ~1.6 | ~1.5-1.7 | 不一定更好 |
| 文本质量 | 良好 | 可能略好或持平 | 数据量有限，深度增加收益递减 |

**关键观察**：
- 4 层 LSTM 训练更慢（多了 2 层的参数和计算量）
- 在莎士比亚数据集上（~1M 字符），2 层已经足够，4 层提升不明显
- 过深的堆叠在数据量不足时反而容易过拟合
- Dropout（`dropout=0.3`）在多层 LSTM 中更重要——它只作用于层间，单层时不生效

---

## 练习 4：实现温度退火生成

**难度**：⭐⭐  
**要改的文件**：`lstm-text-gen/rnn_gen.py`

### 分析

当前 `generate()` 使用固定温度（默认 0.8）。温度退火指在生成过程中，温度从一个较高的值（1.5，鼓励多样性）线性降到较低的值（0.3，保证连贯性）。

核心思想：**生成开头时允许"探索"（高温→更多样），生成结尾时要求"精准"（低温→更确定）**。

### 答案代码

修改 `rnn_gen.py` 中 `generate()` 方法，添加 `temperature_start` 和 `temperature_end` 参数：

```python
def generate(
    self,
    start_str: str,
    char_to_idx: dict,
    idx_to_char: dict,
    length: int = 200,
    temperature: float = 0.8,          # 固定温度（温度退火时忽略）
    device: torch.device = torch.device("cpu"),
    anneal_temp: bool = False,          # 是否启用温度退火
    temp_start: float = 1.5,           # 退火起始温度
    temp_end: float = 0.3,             # 退火结束温度
) -> str:
    self.eval()
    with torch.no_grad():
        chars = [char_to_idx.get(c, 0) for c in start_str]
        inp = torch.tensor([chars], dtype=torch.long, device=device)

        hidden = self.init_hidden(1, device)
        _, hidden = self(inp, hidden)

        result = list(start_str)
        next_char_idx = chars[-1]

        for step in range(length):
            inp = torch.tensor([[next_char_idx]], dtype=torch.long, device=device)
            logits, hidden = self(inp, hidden)

            # --- 温度计算（核心改动）---
            if anneal_temp:
                # 线性退火：从 temp_start 线性降到 temp_end
                progress = step / max(length - 1, 1)  # 0.0 → 1.0
                current_temp = temp_start + (temp_end - temp_start) * progress
            else:
                current_temp = temperature

            logits = logits[0, -1] / max(current_temp, 1e-8)
            probs = torch.softmax(logits, dim=-1)
            next_char_idx = int(torch.multinomial(probs, 1).item())
            result.append(idx_to_char[next_char_idx])

    return "".join(result)
```

在 `train.py` 中调用时启用退火：

```python
# 训练结束后用温度退火生成一段长文本
if epoch == epochs:  # 最后一个 epoch
    sample = model.generate(
        prompt, char_to_idx, idx_to_char,
        length=300, anneal_temp=True,
        temp_start=1.5, temp_end=0.3, device=device,
    )
```

### 预期效果

| 策略 | temperature | 开头（step 0~50） | 结尾（step 150~200） | 整体感受 |
|------|-------------|-------------------|---------------------|---------|
| 固定温度 | 0.8 | 较为连贯 | 偶尔重复 | 平衡 |
| 固定高温 | 1.5 | 多样但跳跃 | 不知所云 | 创造性过强 |
| 固定低温 | 0.3 | 单调重复 | 循环输出 | 过于保守 |
| **温度退火** | 1.5→0.3 | **多样有趣** | **连贯合理** | **最佳** |

退火曲线的选择：
- **线性退火**（实现最简单）：`temp = t_start + (t_end - t_start) * step / length`
- **指数退火**：`temp = t_start * (t_end / t_start) ** (step / length)`
- **余弦退火**：`temp = t_end + 0.5 * (t_start - t_end) * (1 + cos(pi * step / length))`

---

## 练习 5：把莎士比亚换成唐诗数据集

**难度**：⭐⭐  
**要改的文件**：`lstm-text-gen/data.py`

### 分析

数据集切换的核心挑战：
1. **数据来源**：唐诗（全唐诗约 4.8 万首，约 2-3M 字符），可以从 GitHub 获取
2. **编码问题**：中文字符远超 65 个（GB2312 有 6763 个汉字，Unicode 中文更多），词表会从 65 膨胀到几千
3. **模型容量**：词表增大 → Embedding 和 FC 层参数急剧增加，需要调整模型或筛选高频字
4. **序列长度**：中文每字符的信息密度远高于英文，seq_len=100 可能对应 3-5 首五言绝句

### 答案代码

修改 `data.py`：

```python
import os
import urllib.request
import torch
from torch.utils.data import Dataset, DataLoader
import re

# 唐诗数据集（chinese-poetry 仓库的全唐诗 JSON）
TANG_POETRY_URL = (
    "https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/"
    "json/poet.tang.0.json"
)
CACHE_PATH = os.path.join(os.path.dirname(__file__), ".tang_poetry.txt")


def load_tang_poetry(min_len: int = 16, max_vocab: int = 3000) -> str:
    """
    下载并加载唐诗数据集，返回纯文本。

    参数：
      min_len:    最短诗文字符数（过滤过短的诗）
      max_vocab:  保留的最高频字符数（控制词表大小）
    """
    import json

    if not os.path.exists(CACHE_PATH):
        print("  下载唐诗数据集...")
        urllib.request.urlretrieve(TANG_POETRY_URL, ".tang_raw.json")
        with open(".tang_raw.json", "r") as f:
            poems = json.load(f)

        lines = []
        for poem in poems:
            paragraphs = poem.get("paragraphs", [])
            text = "".join(paragraphs)
            # 清洗：保留中文字符、标点和换行
            text = re.sub(r"[^一-鿿　-〿＀-￯\n]", "", text)
            if len(text) >= min_len:
                lines.append(text)

        with open(CACHE_PATH, "w") as f:
            f.write("\n".join(lines))

        if os.path.exists(".tang_raw.json"):
            os.remove(".tang_raw.json")

    with open(CACHE_PATH, "r") as f:
        text = f.read()

    # --- 控制词表大小：只保留高频字 ---
    from collections import Counter
    char_counts = Counter(text)
    top_chars = set(c for c, _ in char_counts.most_common(max_vocab))
    # 保留高频字 + 换行
    top_chars.add("\n")
    filtered = "".join(c if c in top_chars else "□" for c in text)

    print(f"  唐诗加载完成: {len(text):,} 字符, "
          f"词表限制 {max_vocab} 字, "
          f"替换为□: {text.count('□')} 处")
    return filtered


# 将原来的 load_shakespeare 替换为 load_tang_poetry
def load_shakespeare() -> str:
    """兼容旧接口，实际加载唐诗"""
    return load_tang_poetry()
```

在 `main.py` 中建议调整的默认参数：

```python
# 中文 LSTM 的推荐配置
parser.add_argument("--seq_len", type=int, default=64,     # 中文每字信息量大，稍短些
                    help="序列长度（默认 64，中文推荐）")
parser.add_argument("--embed_dim", type=int, default=512,  # 词表变大了，embedding 也要变大
                    help="嵌入维度（默认 512）")
parser.add_argument("--hidden_dim", type=int, default=1024,# 更大的隐藏层
                    help="隐藏维度（默认 1024）")
parser.add_argument("--epochs", type=int, default=30,      # 更多轮次
                    help="训练轮数（默认 30）")
```

运行：

```bash
python main.py --rnn lstm --epochs 30 --embed_dim 512 --hidden_dim 1024 --seq_len 64
```

### 注意事项

1. **词表膨胀**：英文 65 个字符 → 中文 3000+。模型参数从 ~1.5M 涨到 ~15M+
2. **OOV 处理**：低频字替换为 `□`（U+25A1），或使用 subword tokenization（如 SentencePiece）
3. **训练变慢**：`nn.Linear(hidden_dim, vocab_size)` 的矩阵乘法计算量随 vocab_size 线性增长
4. **Perplexity 不可比**：中文 PPL 和英文 PPL 不在一个量级，因为词表大小不同
5. **数据格式**：唐诗的分行很重要——建议保留 `\n` 作为词表字符，帮助模型学习五言/七言的韵律结构
6. **生成的评估**：五言绝句是 4×5 结构，七言绝句是 4×7 结构。好的模型应该能生成字数整齐、押韵的伪诗

---

## 练习 6：Gradient Clipping 消融实验

**难度**：⭐⭐  
**要改的文件**：`lstm-text-gen/train.py`

### 分析

梯度裁剪（Gradient Clipping）是 RNN 训练的标配。消融实验的目的：**量化"有裁剪"和"没有裁剪"的差距，证明这不是一个可有可无的操作**。

当前 `train_epoch()` 接收 `clip` 参数（默认 1.0），直接传给 `clip_grad_norm_`。我们需要做的是：
1. 让 `train()` 函数支持设置不同的 clip 值
2. 在不同 clip 设置下训练并记录 loss 曲线
3. 对比分析结果

### 答案代码

**修改 `train.py` 中 `train()` 函数，添加 `clip` 参数：**

```python
def train(
    model, train_loader, val_loader, char_to_idx, idx_to_char,
    epochs=20, lr=0.001, device=None, prompt="ROMEO:",
    clip=1.0,  # ★ 新增：梯度裁剪阈值，None 表示不裁剪
):
    # ... 前面的代码不变 ...

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device,
            clip=None if clip is None else float(clip),  # 传入裁剪阈值
        )
        val_loss = evaluate(model, val_loader, criterion, device)
        # ... 后面的代码不变 ...
```

**修改 `train_epoch()`，支持 `clip=None`（不裁剪）：**

```python
def train_epoch(model, loader, criterion, optimizer, device, clip=1.0):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        batch_size = x.size(0)
        hidden = model.init_hidden(batch_size, device)

        if isinstance(hidden, tuple):
            hidden = tuple(h.detach() for h in hidden)
        else:
            hidden = hidden.detach()

        optimizer.zero_grad()
        logits, hidden = model(x, hidden)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()

        # --- 条件梯度裁剪 ---
        if clip is not None and clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)
```

**消融实验运行脚本：**

```python
# 在 main.py 或独立脚本中
from data import load_shakespeare, build_vocab, get_dataloaders
from rnn_gen import CharRNN
from train import train
import torch
import json

def run_ablation():
    text = load_shakespeare()
    char_to_idx, idx_to_char, vocab_size = build_vocab(text)
    train_loader, val_loader = get_dataloaders(text, char_to_idx)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}
    for clip_label, clip_val in [("no_clip", None), ("clip_0.5", 0.5), ("clip_1.0", 1.0)]:
        print(f"\n{'='*50}")
        print(f"  实验: {clip_label}")
        print(f"{'='*50}")

        # 每次独立初始化，确保公平对比
        model = CharRNN(vocab_size, rnn_type="lstm")
        train(model, train_loader, val_loader, char_to_idx, idx_to_char,
              epochs=20, device=device, clip=clip_val,
              prompt="ROMEO:")

    return results

if __name__ == "__main__":
    run_ablation()
```

### 预期结果

| 实验设置 | 训练稳定性 | Val Perplexity | 说明 |
|---------|-----------|---------------|------|
| **clip=1.0 (默认)** | 稳定 | ~1.6 | 最佳平衡 |
| **clip=0.5** | 稳定 | ~1.8-2.0 | 过于激进，正常的梯度也被压缩，阻碍学习 |
| **无裁剪** | 可能在 epoch 3-8 出现 NaN | NaN 或 ~10+ | 罕见字符序列触发梯度爆炸 |

### 为什么无裁剪会爆炸？

RNN 训练中一个不常见的字符组合（如 "XZQW"）会让 loss 关于隐藏状态的梯度极大。由于 BPTT 沿时间展开，这个局部的爆炸梯度会传播到前面的所有时间步：

```
某个时间步 t 的罕见字符组合
  → ∂loss/∂h_t 异常大（如 10000）
    → ∂loss/∂W_hh = ... + ∂loss/∂h_t · h_{t-1}  ← 巨大
      → SGD 更新: W_hh += lr × 巨大梯度  ← 破坏已学到的权重
        → 下一轮: 输出全是 NaN
```

### clip 值的选择经验

- **1.0**：最常见，适用于大多数序列模型
- **5.0**：序列很长（>200）时放宽限制
- **0.5**：训练初期不稳定时使用（如 warmup 阶段）
- **动态 clip**：`clip = 1.0 / max(1.0, epoch/10)`——前 10 轮收紧，后面放宽

### 验证方法

除了 loss 曲线，还可以直接打印梯度范数来观察：

```python
# 在 loss.backward() 之后
total_norm = 0.0
for p in model.parameters():
    if p.grad is not None:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
total_norm = total_norm ** 0.5
print(f"Gradient norm (before clip): {total_norm:.2f}")
```

预期：无裁剪时偶尔会看到 `total_norm > 100`，而有裁剪时始终 ≤ clip 值。

---

## 总结：6 个练习的核心知识点

| 练习 | 核心知识点 |
|------|-----------|
| 1 | ResNet 架构的灵活性——改 `num_blocks` 即可切换不同深度，基础模块（BasicBlock）不变 |
| 2 | **梯度消失是 PlainNet 性能差的根本原因**，残差连接的梯度高速公路是 ResNet 成功的核心 |
| 3 | 深度 RNN 并非越深越好——数据量决定了合理的模型容量，过深会导致过拟合 |
| 4 | 温度退火是**控制生成质量和多样性平衡**的实用技巧，开头探索、结尾精准 |
| 5 | 中文 NLP 的挑战：词表膨胀、编码处理、数据格式。字符级模型在中英文间的迁移涉及模型架构调整 |
| 6 | 梯度裁剪是 RNN 训练的**必要性操作**而非可选优化——没有裁剪的训练不稳定，随时可能 NaN |
