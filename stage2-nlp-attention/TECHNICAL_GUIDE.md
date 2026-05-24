# 阶段 2 技术详解：NLP 基础与注意力机制

> 配套项目 `stage2-nlp-attention/`，包含两个 Mini 项目：
> - `word2vec/` — Word2Vec Skip-gram + 负采样，Text8 训练 + 词类比
> - `seq2seq-attention/` — Seq2Seq + Bahdanau/Luong Attention，英法翻译

---

## 目录

1. [从符号到向量：词嵌入的革命](#1-从符号到向量词嵌入的革命)
   - [1.1 One-Hot 的困境](#11-one-hot-的困境)
   - [1.2 分布假说：一个词的含义由它的上下文决定](#12-分布假说一个词的含义由它的上下文决定)
   - [1.3 Word2Vec 的两种架构：CBOW vs Skip-gram](#13-word2vec-的两种架构cbow-vs-skip-gram)
2. [Skip-gram + 负采样：工业级词向量训练](#2-skip-gram--负采样工业级词向量训练)
   - [2.1 Skip-gram 的数学定义](#21-skip-gram-的数学定义)
   - [2.2 负采样：把 V 分类变成 (1+K) 个二分类](#22-负采样把-v-分类变成-1k-个二分类)
   - [2.3 二次采样：让"的"和"the"不再霸屏](#23-二次采样让的和the不再霸屏)
   - [2.4 词向量空间的线性结构](#24-词向量空间的线性结构)
3. [Seq2Seq：序列到序列的通用框架](#3-seq2seq序列到序列的通用框架)
   - [3.1 编码器-解码器架构](#31-编码器-解码器架构)
   - [3.2 Teacher Forcing：训练和推断的裂缝](#32-teacher-forcing训练和推断的裂缝)
   - [3.3 双向编码器：被忽视的"未来信息"](#33-双向编码器被忽视的未来信息)
4. [注意力机制：Seq2Seq 的救世主](#4-注意力机制seq2seq-的救世主)
   - [4.1 信息瓶颈问题](#41-信息瓶颈问题)
   - [4.2 Bahdanau Attention：第一个可学习的对齐](#42-bahdanau-attention第一个可学习的对齐)
   - [4.3 Luong Attention：三种得分函数的工程设计](#43-luong-attention三种得分函数的工程设计)
   - [4.4 为什么注意力有效：梯度视角](#44-为什么注意力有效梯度视角)
5. [实验结果与对比](#5-实验结果与对比)
6. [动手练习](#6-动手练习)
7. [代码逐行讲解（小白友好版）](#7-代码逐行讲解小白友好版)
   - [7.1 Word2Vec 项目](#71-word2vec-项目)
   - [7.2 Seq2Seq + Attention 项目](#72-seq2seq--attention-项目)

---

## 1. 从符号到向量：词嵌入的革命

### 1.1 One-Hot 的困境

在 Word2Vec 之前，NLP 系统的标准操作是：把每个词表示为一个 one-hot 向量，维度和词汇量相同（几万到几十万维）。

```
"king"   = [0, 0, 1, 0, 0, ..., 0]   ← 100,000 维，只有 1 个位置是 1
"queen"  = [0, 0, 0, 1, 0, ..., 0]   ← 和 "king" 的内积 = 0
"monarch"= [0, 0, 0, 0, 1, ..., 0]   ← 1,000 维上，和上面的距离完全一样
```

**one-hot 的三个致命缺陷**：

| 缺陷 | 后果 |
|------|------|
| 维度过高 | V=50000 意味着每个词是一个 50000 维向量，矩阵运算和存储都吃不消 |
| 语义鸿沟 | "king" 和 "queen" 的内积 = 0，"king" 和 "microscope" 的内积也 = 0——无法表示语义相似度 |
| 泛化灾难 | 训练时见过 "cat sat on mat"，测试时遇到 "dog sat on rug" 完全无法借助 "cat↔dog" "mat↔rug" 的相似性 |

### 1.2 分布假说：一个词的含义由它的上下文决定

语言学家 J.R. Firth (1957)：**"You shall know a word by the company it keeps."**

这是所有词嵌入方法的理论基础。"king" 和 "queen" 不是同义词，但它们出现在相似的上下文中（"the ___ ruled the kingdom"、"the ___ wore a crown"），所以它们的向量应该接近。

### 1.3 Word2Vec 的两种架构：CBOW vs Skip-gram

| 特性 | CBOW | Skip-gram |
|------|------|-----------|
| 任务 | 上下文词 → 预测中心词 | 中心词 → 预测上下文词 |
| 训练速度 | 快（一个样本只预测 1 个词） | 慢（一个样本预测 2w 个词） |
| 稀有词质量 | 差（平均掉了） | **好**（每个词都要做中心词，都有充分训练） |
| 直觉 | "看到周围一群人，猜中间是谁" | "看到一个人，猜他周围有哪些朋友" |

**为什么本项目选 Skip-gram？** 对稀有词（如专业术语、古词）的词向量质量更好。在类比任务上，Skip-gram 通常优于 CBOW。

---

## 2. Skip-gram + 负采样：工业级词向量训练

### 2.1 Skip-gram 的数学定义

**原始 softmax 损失**（不可行的大词表版本）：

$$p(w_o | w_c) = \frac{\exp(u_c \cdot v_o)}{\sum_{w=1}^{V} \exp(u_c \cdot v_w)}$$

$$\mathcal{L} = -\frac{1}{T} \sum_{t=1}^{T} \sum_{-w \leq j \leq w, j \neq 0} \log p(w_{t+j} | w_t)$$

- $w_c$：中心词，$w_o$：上下文词
- $u_c$：中心词 $w_c$ 的输入向量（$u$ embedding）
- $v_o$：上下文词 $w_o$ 的输出向量（$v$ embedding）
- $V$：词表大小（50000+）
- 分母：要对整个词表的 50000 个词算点积再求和 = **每个样本 50000 次点积**

### 2.2 负采样：把 V 分类变成 (1+K) 个二分类

**核心洞察**：我们不需要精确知道每个词的概率，只需要让正确上下文比随机噪声得分更高。

**负采样损失**：

$$\mathcal{L}_{NEG} = -\log\sigma(u_c \cdot v_o) - \sum_{k=1}^{K} \mathbb{E}_{w_n \sim P_n}[\log\sigma(-u_c \cdot v_n)]$$

- 第一项：正样本的 log-sigmoid —— 希望中心词和上下文词的相似度高
- 第二项：K 个负样本的 log-sigmoid(-x) —— 希望中心词和噪声词的相似度低
- $P_n(w) \propto \text{count}(w)^{0.75}$：噪声分布，$^{0.75}$ 提升低频词的采样概率

**计算量对比**：

| 方法 | 每次更新的点积次数 | V=50000 |
|------|------------------|---------|
| Softmax | V 次 | 50,000 |
| 负采样 (K=5) | 1+K 次 | 6 |
| **加速比** | | **~8000x** |

**sigmoid 和 softmax 的区别**：

```
softmax: "这 V 个词中，哪个最可能是上下文？" → V 分类问题
sigmoid:  "这两个词是不是真实的上下文对？" → 二分类问题
```

负采样的美妙之处在于：**把 V 分类问题变成了 (1+K) 个独立的二分类问题**，计算量和词表大小解耦。

### 2.3 二次采样：让"的"和"the"不再霸屏

**为什么需要二次采样？**

"the" 占英语文本约 7% 的出现次数，即每 100 个样本中有 7 个的中心词是 "the"。这些样本的训练信号极弱——从 "the" 的上下文学不到任何有意义的语义关系。

**二次采样公式**：

$$P_{drop}(w) = 1 - \sqrt{\frac{t}{f(w)}}$$

- $t$：阈值（通常 1e-3 ~ 1e-5）
- $f(w)$：词 $w$ 的频率

| 词 | 频率 | t=1e-3 时 P_drop |
|----|------|------------------|
| "the" | 5% | $1 - \sqrt{0.001/0.05} = 0.86$ ← 86% 被丢弃 |
| "king" | 0.001% | $1 - \sqrt{0.001/0.00001} \approx 0$ ← 从不丢弃 |
| "dog" | 0.005% | $1 - \sqrt{0.001/0.00005} \approx 0$ ← 从不丢弃 |

**效果**：大幅减少高频功能词的训练样本，让模型把算力集中在有语义承载力的实词上。

### 2.4 词向量空间的线性结构

训练好的 Word2Vec 向量最令人惊叹的性质：**语义关系以向量算术的形式编码**。

$$\vec{v}(\text{king}) - \vec{v}(\text{man}) + \vec{v}(\text{woman}) \approx \vec{v}(\text{queen})$$
$$\vec{v}(\text{paris}) - \vec{v}(\text{france}) + \vec{v}(\text{italy}) \approx \vec{v}(\text{rome})$$

**为什么会有这种性质？**

Skip-gram 的训练目标迫使模型学习"在相同上下文中可互换的词"。如果一个词可以替换另一个词而不改变句子的分布，它们会有相似的向量。但更重要的是**差异向量**——"man" 和 "woman" 的差异恰好对应性别维度，而 "king" 和 "queen" 的差异也对应同样的维度。

> 严格来说，$\vec{v}(\text{king}) - \vec{v}(\text{man})$ 不精确等于"性别向量"，而是所有和 "king" 共现但和 "man" 不共现的上下文的加权和。实际中需要用 PCA/SVD 分析才能确认具体的语义维度。

**GloVe 的补充视角**：Word2Vec 只用局部上下文窗口，而 GloVe 额外利用了全局词共现矩阵。但本质上两者学到的向量空间具有相似的线性结构。

---

## 3. Seq2Seq：序列到序列的通用框架

### 3.1 编码器-解码器架构

Seq2Seq 由 Sutskever et al. (2014) 提出，是机器翻译、文本摘要、对话系统的"万能框架"。

```
Encoder（编码器）:      Decoder（解码器）:
  "I love you"             <SOS> → je → t'aime → <EOS>
      ↓                       ↑        ↑       ↑
  ┌─────────┐             ┌─────────┐
  │  GRU→   │             │  GRU→   │
  │  GRU→   │ ──context──→│  GRU→   │
  │  GRU→   │             │  GRU→   │
  └─────────┘             └─────────┘
   源语言句子               目标语言句子
```

**Encoder** 把变长源语言句子"压缩"成一个固定长度的向量（上下文向量 context vector）。

**Decoder** 从这个向量出发，逐词生成目标语言句子。

### 3.2 Teacher Forcing：训练和推断的裂缝

**问题**：训练时，Decoder 的输入是标准答案；推断时，Decoder 的输入是上一步的预测。这个不匹配叫做 **Exposure Bias**。

**Teacher Forcing** 的解决方案：以概率 $p$ 使用标准答案，以概率 $1-p$ 使用模型自己的预测。

```python
if random.random() < teacher_forcing_ratio:  # 0.5
    next_input = ground_truth_token           # 正确答案
else:
    next_input = model_prediction             # 模型自己的预测
```

| Teacher Forcing Ratio | 效果 |
|----------------------|------|
| 1.0（纯 teacher forcing） | 训练收敛快，但推断时质量差（没学过从自身错误中恢复） |
| 0.0（纯自回归训练） | 训练不稳定，很难收敛 |
| **0.5（混合）** | **本项目的选择**，平衡收敛速度和鲁棒性 |
| 0.5 → 0.0（课程学习） | 训练后期逐步降低 ratio，让模型适应推断场景 |

### 3.3 双向编码器：被忽视的"未来信息"

翻译一个词时，只看左边的上下文不够，还要看右边。

```
"the bank of the river"   → "la rive"     (河岸)
"the bank opened at 9"    → "la banque"    (银行)
```

单向 RNN 在读到 "bank" 时不知道后面是 "river" 还是 "opened"。**双向 RNN** 解决了这个问题——正向 RNN 读左边，反向 RNN 读右边，两个方向的隐藏状态拼接起来。

```python
# 双向 GRU 的输出
outputs = torch.cat([forward_hidden, backward_hidden], dim=-1)
# (batch, seq_len, hidden_dim) → (batch, seq_len, 2 * hidden_dim)
```

---

## 4. 注意力机制：Seq2Seq 的救世主

### 4.1 信息瓶颈问题

原始 Seq2Seq（无注意力）有一个根本问题：

> **整个源语言句子的信息被压缩成 Encoder 最后一个隐藏状态这一个固定维度的向量。**

- 翻译 50 词的长句时，encoder 最后一个 hidden state 需要"记住" 50 个词的全部语义——这不可能。
- 实际效果：短句翻译尚可，长句翻译质量断崖式下降。

注意力机制的解决方案：**Decoder 每生成一个词，都"回头看" Encoder 的所有隐藏状态，自己决定该关注哪些位置。**

### 4.2 Bahdanau Attention：第一个可学习的对齐

Bahdanau et al. (ICLR 2015) 提出了第一个端到端可训练的注意力机制。

**计算步骤**：

$$\text{score}(h_j, s_{t-1}) = v^T \cdot \tanh(W_h h_j + W_s s_{t-1})$$

$$\alpha_t = \text{softmax}(\text{score}_t)$$

$$c_t = \sum_{j=1}^{S} \alpha_{t,j} \cdot h_j$$

| 符号 | 含义 | 维度 |
|------|------|------|
| $h_j$ | Encoder 第 j 个位置的隐藏状态 | $2H_{enc}$ |
| $s_{t-1}$ | Decoder 上一步的隐藏状态 | $H_{dec}$ |
| $v$ | 可学习的投影向量 | $H_{dec}$ |
| $\alpha_{t,j}$ | 第 t 步对第 j 个源语言词的注意力权重 | 标量 |
| $c_t$ | 上下文向量（所有源语言词的加权和） | $2H_{enc}$ |

**白话理解**：
1. 对每个源语言位置 j，把 encoder 的 $h_j$ 和 decoder 当前状态 $s_{t-1}$ 拼在一起
2. 经过一个全连接层 + tanh，算出一个"匹配得分"
3. softmax 把这些得分变成概率分布（= 注意力权重）
4. 用这个分布对 encoder 输出做加权平均 = "当前应该关注哪些源语言词"

### 4.3 Luong Attention：三种得分函数的工程设计

Luong et al. (EMNLP 2015) 在 Bahdanau 基础上做了系统的工程设计：

| 得分函数 | 公式 | 参数 | 特点 |
|---------|------|------|------|
| **Dot** | $h_j^T \cdot s_t$ | 无 | 最简单，要求 encoder/decoder 维度相同 |
| **General** | $h_j^T \cdot W \cdot s_t$ | W 矩阵 | 最常用，$W$ 适配不同维度 |
| **Concat** | $v^T \cdot \tanh(W \cdot [h_j; s_t])$ | W, v | 类似 Bahdanau，表达能力最强但也最慢 |

**Bahdanau vs Luong 的关键区别**：

| | Bahdanau | Luong |
|------|---------|-------|
| 使用的 Decoder 状态 | $s_{t-1}$（上一步的 hidden） | $s_t$（当前步的 hidden） |
| 计算顺序 | 先算 context → 再算 s_t | 先算 s_t → 再算 context |
| 隐藏状态维度 | encoder 和 decoder 可以不同 | dot 要求相同 |
| 默认得分 | Additive | Multiplicative (dot/general) |

### 4.4 为什么注意力有效：梯度视角

没有注意力时，encoder 前端词的梯度需要穿过整个序列才能到达 decoder loss：

```
loss → decoder (T 步) → encoder 最后一个 hidden → ... → encoder 第一个 hidden
```

序列越长，路径越长，梯度衰减越严重。

有注意力时，每个 encoder hidden state $h_j$ 都有一条直达 decoder loss 的路径：

```
loss → context_vector → α_{t,j} · h_j
              ↑
         (权重这个系数一般 > 0)
```

注意力权重 $\alpha_{t,j}$ 在 0 到 1 之间，且和为 1——它天然提供了一个"软门控"，让梯度可以跳过时间步直接流动到任何 encoder 位置。**这就是后来 Transformer 论文说"Attention 提供了一条梯度高速公路"的原因。**

---

## 5. 实验结果与对比

### 5.1 Word2Vec：词类比准确率

运行：

```bash
cd word2vec
python main.py --epochs 10 --embed_dim 200
```

| 配置 | 期望类比准确率 | 训练时间 (CPU) |
|------|-------------|--------------|
| Text8, 50k 词表, 200 维, 10 epochs | 30-50% | ~30 分钟 |
| Text8, 50k 词表, 300 维, 15 epochs | 40-60% | ~1 小时 |

**类比准确率的衡量**：在 Google Analogy Test Set 上，模型能正确回答 "king - man + woman = ?" 这类问题的比例。

### 5.2 Seq2Seq + Attention：翻译质量

运行：

```bash
cd seq2seq-attention
python main.py --attn bahdanau           # Bahdanau Attention
python main.py --attn luong_general      # Luong General Attention
python main.py --attn luong_dot          # Luong Dot Attention
```

| 配置 | 期望 BLEU | 备注 |
|------|----------|------|
| Bahdanau Attention | 15-25 | 对齐直观 |
| Luong General Attention | 15-25 | 性能接近 Bahdanau |
| Luong Dot Attention | 12-22 | 无参数，对维度敏感 |
| Beam Search (size=3) | +1~3 BLEU | 贪心解码的改进 |
| 无注意力 (baseline) | 5-15 | 长句质量严重下降 |

**关于 BLEU 分数**：英法翻译在小数据集上（~10 万句对）的 BLEU 通常不超过 25 分。这不是模型的错，而是数据量限制。工业级 NMT 系统需要百万级句对 + subword tokenization 才能达到 30-40+ BLEU。

### 5.3 注意力权重可视化

运行后会生成 `attention_heatmap.png`。观察要点：

- **对角对齐**：英语的 "I love you" → 法语的 "je t'aime"，权重应该大致对角分布
- **词序差异**：英语 "red car" → 法语 "voiture rouge"（名词在前），注意力权重会在不同位置形成热点
- **软对齐**：不是硬的一对一映射，而是多个源语言词对一个目标语言词的"软"加权

---

## 6. 动手练习

| 序号 | 练习 | 难度 | 要改的文件 |
|------|------|------|-----------|
| 1 | 实现 CBOW 架构：用上下文词的平均向量预测中心词，对比 Skip-gram 的效果 | ⭐⭐ | `word2vec/word2vec.py`, `word2vec/data.py` |
| 2 | 训练中文 Word2Vec：用中文维基百科替换 Text8，观察类比结果 | ⭐⭐ | `word2vec/data.py`, `word2vec/main.py` |
| 3 | 消融实验：对比有无 Teacher Forcing、不同 TF ratio 的翻译质量 | ⭐ | `seq2seq-attention/train.py` |
| 4 | 实现注意力权重可视化：在翻译的同时保存注意力矩阵，画热力图 | ⭐ | `seq2seq-attention/translate.py` |
| 5 | 对比三种 Luong 注意力（dot/general/concat）的翻译质量和训练速度 | ⭐⭐ | `seq2seq-attention/attention.py` |
| 6 | 实现 BPE 子词切分：用 Byte-Pair Encoding 替代空格分词，解决 OOV 问题 | ⭐⭐⭐ | `seq2seq-attention/data.py` |

### 练习 1 提示

```python
# CBOW 的核心修改：输入是上下文词的 embedding 平均值
class CBOW(nn.Module):
    def forward(self, context, target, neg_samples):
        # context: (B, 2*window_size) 上下文词索引
        u_context = self.u_embed(context).mean(dim=1)  # 取平均
        v_target = self.v_embed(target)
        pos_loss = F.logsigmoid((u_context * v_target).sum(dim=1))
        # ... 负采样同 Skip-gram
```

### 练习 2 提示

- 中文维基百科下载：`https://dumps.wikimedia.org/zhwiki/`
- 需要分词（jieba），不能按空格分
- 词表设置 30000-50000（常用汉字 + 词汇）
- 可视化时检查："北京 - 中国 + 法国 ≈ 巴黎"

### 练习 6 提示

BPE 的核心操作：
1. 把每个词拆成字符序列
2. 统计所有相邻符号对的频率
3. 合并最高频的符号对（如 "e"+"r" → "er"）
4. 重复直到达到目标词表大小

---

## 7. 代码逐行讲解（小白友好版）

### 7.1 Word2Vec 项目

#### 7.1.1 负采样损失函数的维度追踪

```python
def forward(self, center, context, neg_samples):
    # center:      (B,)     如 [42, 7, 100, ...]
    # context:     (B,)     如 [8, 15, 200, ...]
    # neg_samples: (B, K)   如 [[1,5,9,2,3], ...]  K=5

    u_c = self.u_embed(center)           # (B,) → (B, D)  中心词向量
    v_o = self.v_embed(context)          # (B,) → (B, D)  正上下文向量
    pos_score = (u_c * v_o).sum(dim=1)   # (B,)  逐元素乘 → 求和 = 点积
    pos_loss = F.logsigmoid(pos_score)   # log σ(u·v)

    v_n = self.v_embed(neg_samples)      # (B, K) → (B, K, D)
    # Einstein summation: 对每个 batch 和每个负样本计算点积
    neg_score = torch.einsum("bd,bkd->bk", u_c, v_n)  # (B, K)
    neg_loss = F.logsigmoid(-neg_score)  # log σ(-u·v_n)

    loss = -(pos_loss + neg_loss.sum(dim=1)).mean()
    return loss
```

**为什么两个 Embedding 矩阵？**

训练时，每个词以两种身份出现：作为中心词（输入）和作为上下文词（输出）。直觉上，"dog" 作为被预测的目标和作为用来预测其他词的源，最优表示可能不同。实际工程中，最终词向量通常取 `u_embed`（中心词向量），或取 `(u_embed + v_embed) / 2` 来利用两个矩阵的信息。

**为什么用 Xavier 初始化？** 不用的话，初始词向量可能全在 0 附近——所有点积 ≈ 0 → sigmoid(0) = 0.5 → 模型一开始就觉得所有词对是 50% 概率的上下文关系 → 训练迟迟不收敛。

#### 7.1.2 二次采样的工程实现

```python
def make_subsample_table(word_counts, threshold=1e-3):
    total = sum(c for _, c in word_counts)
    freq = np.array([c / total for _, c in word_counts])
    drop_probs = 1.0 - np.sqrt(threshold / (freq + 1e-12))
    return np.clip(drop_probs, 0, 1)

def apply_subsample(tokens, word_to_idx, drop_probs):
    idxs = []
    for token in tokens:
        if token in word_to_idx:
            idx = word_to_idx[token]
            if random.random() > drop_probs[idx]:  # 以概率 (1-drop_prob) 保留
                idxs.append(idx)
    return idxs
```

**关键细节**：`random.random() > drop_probs[idx]` 而非 `<`。因为 `drop_probs` 是丢弃概率，所以大于它才保留。

**二次采样是随机的**：同一个词在文本中第一次出现可能被保留，第二次被丢弃。随机的保留策略确保了高频词仍然有部分出现机会，只是大幅降低了频率。

#### 7.1.3 词类比为什么有效

```python
def find_analogy(word_a, word_b, word_c, ...):
    query = vectors[idx_a] - vectors[idx_b] + vectors[idx_c]
    similarities = cosine_sim(query, all_vectors)
    # 返回和 query 最相似（但不包含 a, b, c 本身）的词
```

这不是魔法。Skip-gram 的训练目标最大化 $\log p(w_{t+j}|w_t)$，这等价于最大化 $u_{w_t} \cdot v_{w_{t+j}}$。如果一个词经常在 "X ruled the kingdom" 这类上下文中出现，它就会和 "king" 有相似的上下文分布。向量空间的线性结构是**上下文分布相似性的副产品**。

### 7.2 Seq2Seq + Attention 项目

#### 7.2.1 Encoder 的 pack_padded_sequence 详解

```python
def forward(self, src, src_lengths):
    embed = self.dropout(self.embedding(src))  # (B, S) → (B, S, E)

    # ★ 关键操作：pack → GRU → unpack
    packed = nn.utils.rnn.pack_padded_sequence(
        embed, src_lengths.cpu(), batch_first=True, enforce_sorted=False,
    )
    outputs, hidden = self.gru(packed)
    outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
```

| 操作 | 作用 |
|------|------|
| `pack_padded_sequence` | 把 `<PAD>` 对应的无效时间步跳过，GRU 不会在这些位置计算状态更新 |
| `enforce_sorted=False` | 不强制要求 batch 内句子按长度降序排列（PyTorch 1.1+ 支持） |
| `pad_packed_sequence` | 把压缩后的 GRU 输出恢复成正常的 padded tensor 形状 |

**不 pack 会怎样？** GRU 会在 `<PAD>` 位置上进行无意义的计算，浪费算力，而且 `<PAD>` 的 embedding（全零）会影响隐藏状态的统计分布。

#### 7.2.2 Decoder 的输入馈送机制

```python
# Luong 论文提出的 "input feeding" 方法
fc_input = torch.cat([
    gru_out.squeeze(1),    # decoder 当前步的隐藏状态  (B, H_dec)
    embed.squeeze(1),      # 当前输入词的 embedding   (B, E)
    context.squeeze(1),    # 注意力上下文向量         (B, H_enc)
], dim=1)
logits = self.fc(fc_input)  # (B, H_dec+E+H_enc) → (B, V)
```

为什么要把三个信号都拼在一起？纯粹的 "用 context 算 logits" 会丢失 decoder 自身状态的信息。把 hidden + embedding + context 三者都喂给 FC 层，让网络自己决定每个信号的重要性——这是深度学习中"让可学习参数自己选择"的哲学。

#### 7.2.3 Bahdanau Attention 的逐行实现

```python
class BahdanauAttention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        self.W_h = nn.Linear(enc_hidden_dim, dec_hidden_dim, bias=False)
        self.W_s = nn.Linear(dec_hidden_dim, dec_hidden_dim, bias=False)
        self.v   = nn.Linear(dec_hidden_dim, 1, bias=False)

    def forward(self, enc_outputs, dec_hidden, mask=None):
        # enc_outputs: (B, S, H_enc)  所有 encoder 位置的隐藏状态
        # dec_hidden:  (B, H_dec)     当前 decoder 隐藏状态

        # ① W_h(h_j): (B, S, H_enc) → (B, S, H_dec)
        energy_h = self.W_h(enc_outputs)

        # ② W_s(s_{t-1}): (B, H_dec) → (B, 1, H_dec)  广播到所有时间步
        energy_s = self.W_s(dec_hidden).unsqueeze(1)

        # ③ 相加 → tanh → 投影到标量
        # energy_h + energy_s: (B, S, H_dec) + (B, 1, H_dec) = (B, S, H_dec)
        # v(tanh(...)): (B, S, H_dec) → (B, S, 1) → squeeze → (B, S)
        energy = self.v(torch.tanh(energy_h + energy_s)).squeeze(-1)

        # ④ mask <PAD> 位置 → -inf
        if mask is not None:
            energy = energy.masked_fill(~mask, float("-inf"))

        # ⑤ softmax → 权重 + 加权求和
        attn_weights = F.softmax(energy, dim=1)  # (B, S)
        context = torch.bmm(attn_weights.unsqueeze(1), enc_outputs)  # (B, 1, H_enc)
        return context.squeeze(1), attn_weights
```

**为什么三个 Linear 的 bias=False？** Bahdanau 原始实现确实使用了 bias，但实践证明去掉 bias 基本不影响效果，反而减少参数、加速计算。这里的技巧和 ResNet 中 conv + BN 的 bias=False 是一回事——`v` 层的隐式偏置已经让模型的决策边界足够灵活。

#### 7.2.4 BLEU 的 n-gram 精度计算

```python
def compute_bleu(reference, candidate, max_n=4):
    bp = min(1.0, math.exp(1.0 - len(reference) / len(candidate)))

    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(tuple(ref[i:i+n]) for i in range(len(ref)-n+1))
        cand_ngrams = Counter(tuple(cand[i:i+n]) for i in range(len(cand)-n+1))
        overlap = sum(min(c, ref_ngrams.get(ng, 0)) for ng, c in cand_ngrams.items())
        precisions.append(overlap / max(len(cand) - n + 1, 1))

    log_avg = sum(math.log(max(p, 1e-10)) for p in precisions) / max_n
    return bp * math.exp(log_avg)
```

**为什么需要 brevity penalty？** 假设候选是 "the"，1-gram 精度 100%（"the" 出现在参考译文中）。如果不惩罚短句，这个 1 词的翻译就是满分——但它显然不是好翻译。BP 惩罚过短的候选（当候选比参考短时）。

**为什么 n 最多取 4？** 这是 BLEU 原始论文的经验值。5-gram 以上过于稀疏，统计意义不大。

#### 7.2.5 Teacher Forcing 的课程学习策略

```python
# 训练初期：TF ratio 高（多用正确答案），让模型先学会"看到正确答案时怎么翻译"
# 训练后期：TF ratio 低（多用自预测），让模型适应推断环境
current_tf = max(0.1, tf_ratio * (1.0 - epoch / epochs))
```

课程学习的直觉和"教小孩骑车"一样：开始时时扶着他（TF=1.0），慢慢松手（TF=0），直到他能自己骑（TF=0.0）。

---

## 从阶段 2 到阶段 3：Attention 如何演变成 Transformer

理解阶段 2 的注意力机制后，阶段 3 的 Transformer 就是自然的延伸：

| 概念（阶段 2） | Transformer 对应（阶段 3） |
|---------------|--------------------------|
| Bahdanau/Luong Attention（decoder→encoder） | Cross-Attention（decoder 层中的多头注意力） |
| Encoder 的 Bi-GRU 序列建模 | Self-Attention（encoder 层中的多头注意力） |
| 注意力权重 softmax | 完全相同的 softmax，只是加了 scaling（√d_k） |
| 单个注意力头 | Multi-Head Attention（8 个头并行） |
| RNN 的时序依赖 | Positional Encoding（位置信息注入） |

Transformer 本质上是用 **Self-Attention 完全替代 RNN**——把"时间步递推"变成"所有位置两两直接交互"。这解决了 RNN 的最后一个缺陷：无法并行训练（因为第 t 步必须等第 t-1 步的结果）。
