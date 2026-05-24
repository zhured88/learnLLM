# Stage 2 NLP 基础与注意力机制 — 练习题答案

---

## 练习 1：实现 CBOW 架构

**难度**：⭐⭐  
**要改的文件**：`word2vec/word2vec.py`, `word2vec/data.py`

### 分析

CBOW (Continuous Bag of Words) 和 Skip-gram 互为镜像：

| | CBOW | Skip-gram |
|------|------|-----------|
| 输入 | 2w 个上下文词 | 1 个中心词 |
| 输出 | 1 个中心词 | 2w 个上下文词 |
| 向量聚合 | 多个上下文的平均 | 单个向量 |

核心变化：CBOW 的输入 embedding 是上下文词向量的**平均值**（或求和）。

### 答案代码

**data.py：新增 CBOWDataset**

```python
class CBOWDataset(Dataset):
    """CBOW 数据集：上下文词 → 预测中心词"""

    def __init__(self, token_ids, window_size=5, num_negative=5,
                 noise_dist=None):
        self.token_ids = token_ids
        self.window_size = window_size
        self.num_negative = num_negative
        self.noise_dist = noise_dist
        self.vocab_size = len(noise_dist) if noise_dist is not None else 0

        self.pairs = []
        for i in range(window_size, len(token_ids) - window_size):
            center = token_ids[i]
            # 收集上下文词索引列表
            context = [token_ids[j] for j in range(i - window_size, i + window_size + 1)
                      if j != i]
            if len(context) == 2 * window_size:  # 确保完整窗口
                self.pairs.append((center, context))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        center, context = self.pairs[idx]
        neg_samples = np.random.choice(
            self.vocab_size, size=self.num_negative,
            replace=True, p=self.noise_dist,
        )
        return (
            torch.tensor(context, dtype=torch.long),        # (2w,)
            torch.tensor(center, dtype=torch.long),         # 标量
            torch.tensor(neg_samples, dtype=torch.long),    # (K,)
        )
```

**word2vec.py：新增 CBOW 模型**

```python
class CBOW(nn.Module):
    """CBOW + 负采样"""

    def __init__(self, vocab_size, embed_dim=200):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # CBOW 的两个 embedding 角色互换：
        # u_embed: 上下文词的 embedding（输入）
        # v_embed: 中心词的 embedding（输出）
        self.u_embed = nn.Embedding(vocab_size, embed_dim)
        self.v_embed = nn.Embedding(vocab_size, embed_dim)
        nn.init.xavier_uniform_(self.u_embed.weight)
        nn.init.xavier_uniform_(self.v_embed.weight)

    def forward(self, context, target, neg_samples):
        # context: (B, 2w)  输入是多个上下文词索引
        # target:  (B,)     输出是中心词索引

        # 取上下文词向量的平均值
        u_context = self.u_embed(context)       # (B, 2w, D)
        u_avg = u_context.mean(dim=1)           # (B, D) ← 关键：多个词取平均

        v_target = self.v_embed(target)         # (B, D)
        pos_score = (u_avg * v_target).sum(dim=1)
        pos_loss = F.logsigmoid(pos_score)

        v_neg = self.v_embed(neg_samples)       # (B, K, D)
        neg_score = torch.einsum("bd,bkd->bk", u_avg, v_neg)
        neg_loss = F.logsigmoid(-neg_score)

        return -(pos_loss + neg_loss.sum(dim=1)).mean()

    def get_vectors(self):
        # CBOW 通常取 (u + v) / 2 或仅取 u
        return ((self.u_embed.weight + self.v_embed.weight) / 2).detach().cpu()
```

### 预期对比

| 指标 | Skip-gram | CBOW |
|------|-----------|------|
| 训练速度 | 慢（每个中心词产生 2w 个样本） | **快**（2w 个词只产生 1 个样本） |
| 稀有词质量 | **好** | 一般 |
| 语义类比 | **较好** | 略差 |
| 语法类比 | 较好 | **好**（对功能词更好） |

---

## 练习 2：训练中文 Word2Vec

**难度**：⭐⭐  
**要改的文件**：`word2vec/data.py`, `word2vec/main.py`

### 答案

**data.py：中文维基百科加载器**

```python
import jieba
import re

ZHWIKI_URL = "https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2"

def load_zhwiki_sample() -> list:
    """
    加载中文维基百科语料

    由于完整 dump 约 2GB，实际教学中建议：
    1. 使用预处理好的中文语料（如 https://github.com/brightmart/nlp_chinese_corpus）
    2. 或使用 Wikipedia Extractor 提取文本后 jieba 分词
    """
    # 简化版本：用内置中文文本替代
    import urllib.request
    url = "https://raw.githubusercontent.com/ymcui/Chinese-Word-Vectors/master/data/zhwiki_sample.txt"
    path = os.path.join(CACHE_DIR, "zhwiki_sample.txt")
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)

    with open(path, "r") as f:
        text = f.read()

    # 分句 + 分词
    sentences = re.split(r"[。！？\n]+", text)
    tokens = []
    for sent in sentences:
        words = jieba.lcut(sent.strip())
        tokens.extend(words)

    return tokens
```

**main.py：中文类比测试集**

```python
CHINESE_ANALOGIES = [
    # 首都-国家
    ("北京", "中国", "巴黎", "法国"),
    ("东京", "日本", "伦敦", "英国"),
    ("首尔", "韩国", "柏林", "德国"),
    # 省份-省会
    ("广东", "广州", "江苏", "南京"),
    ("四川", "成都", "湖北", "武汉"),
    # 动词时态
    ("吃", "吃了", "看", "看了"),
    ("走", "走过", "跑", "跑过"),
    # 反义词
    ("大", "小", "高", "矮"),
    ("上", "下", "左", "右"),
]
```

### 中文 Word2Vec 的特殊考量

1. **分词质量决定向量质量**：jieba 的默认词表对专业术语切分可能不准确
2. **词表大小**：中文"词"远多于英文（常见词约 10-30 万），建议 max_vocab=30000
3. **单字词**：中文单字词也有独立语义（如"跑"、"打"），不应过滤
4. **繁体处理**：如果用繁体语料，建议先转简体（opencc）

---

## 练习 3：Teacher Forcing 消融实验

**难度**：⭐  
**要改的文件**：`seq2seq-attention/train.py`

### 答案

**修改 train() 函数，支持多个 TF ratio 对比：**

```python
def run_tf_ablation():
    """对比不同 Teacher Forcing ratio 的效果"""
    ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = {}

    for tf_ratio in ratios:
        print(f"\n{'='*50}")
        print(f"  Teacher Forcing Ratio: {tf_ratio}")
        print(f"{'='*50}")

        # 每次独立初始化模型
        model = Seq2Seq(
            Encoder(src_vocab, ...),
            Decoder(tgt_vocab, ...),
            tgt_pad_idx=pad_idx,
        )

        trained = train(model, train_loader, val_loader,
                       epochs=20, tf_ratio=tf_ratio)

        val_loss, _ = evaluate(trained, val_loader, device)
        results[tf_ratio] = val_loss

    # 打印对比表
    print(f"\n{'='*50}")
    print("  Teacher Forcing Ablation Results")
    print(f"{'='*50}")
    for ratio, loss in results.items():
        print(f"  TF={ratio:.2f}: val_loss={loss:.4f}")

    return results
```

### 预期结果

| TF Ratio | 训练收敛 | 推断 BLEU | 说明 |
|----------|---------|----------|------|
| 0.0 | 极慢/不收敛 | - | 模型自预测完全不准，误差累积 |
| 0.25 | 慢 | 较低 | 模型学到的信号不够清晰 |
| **0.5** | **快** | **最高** | 平衡最佳 |
| 0.75 | 快 | 略低 | 训练和推断的裂缝变大 |
| 1.0 | 最快 | 明显下降 | Exposure Bias 最严重 |
| 0.5→0.1 课程学习 | 中 | 可能最高 | 训练后期适应推断 |

**核心观察**：TF=1.0 训练 loss 最低，但推断 BLEU 最差。这证明了 Exposure Bias 是真实存在的——不要在验证 loss 上被 TF=1 欺骗了。

---

## 练习 4：实现注意力权重可视化

**难度**：⭐  
**要改的文件**：`seq2seq-attention/translate.py`

### 答案

本项目的 `translate.py` 已经包含 `plot_attention()` 函数。运行方式：

```bash
python main.py --attn bahdanau
# 训练完成后，会在第一句翻译示例时自动生成 attention_heatmap.png
```

**关键代码**（已在 translate.py 中实现）：

```python
def plot_attention(src_words, tgt_words, attn_matrix, save_path):
    plt.figure(figsize=(len(src_words) * 0.8, len(tgt_words) * 0.8))
    plt.imshow(attn_matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    plt.xticks(range(len(src_words)), src_words, rotation=45)
    plt.yticks(range(len(tgt_words)), tgt_words)
    plt.xlabel("Source (English)")
    plt.ylabel("Target (French)")
    plt.colorbar(label="Attention Weight")
    plt.savefig(save_path, dpi=150)
```

### 如何解读热力图

以 "I love you" → "je t'aime" 为例：

```
        I   love  you
  je   [0.1  0.3  0.6]  ← 法语 "je" 最关注英语 "you"（法语中 "je t'aime" 的第一个词对齐到英语最后一个词）
  t'   [0.2  0.6  0.2]  ← 关注 "love"
  aime [0.1  0.5  0.4]  ← 关注 "love"
```

这表明 "t'aime"（你+爱）对应英语的 "love you"，注意力学到了词序差异。

---

## 练习 5：对比三种 Luong 注意力

**难度**：⭐⭐  
**要改的文件**：`seq2seq-attention/attention.py`

### 分析

三种 Luong 注意力的核心差异仅在 `_score()` 方法中，可以直接运行对比实验：

```bash
python main.py --attn luong_dot       --epochs 20
python main.py --attn luong_general   --epochs 20
python main.py --attn luong_concat    --epochs 20
python main.py --attn bahdanau        --epochs 20  # 对照
```

### 预期结果

| 注意力 | 参数量 | 每 batch 时间 | BLEU | 特点 |
|--------|-------|-------------|------|------|
| Luong Dot | 0 | 最快 | 较低 | 对维度敏感，要求 enc/dec dim 相同 |
| Luong General | 少 (W 矩阵) | 快 | 较高 | **实践中最常用** |
| Luong Concat | 多 (W + v) | 中 | 高 | 和 Bahdanau 类似，但参数稍少 |
| Bahdanau | 多 (W_h, W_s, v) | 慢 | 高 | 第一个注意力机制，对齐最直观 |

**维度不匹配时的处理**：如果 encoder hidden dim ≠ decoder hidden dim，dot attention 直接报错（矩阵维度不兼容）。这是 dot attention 的主要限制——而 general 和 concat 通过可学习的 W 矩阵自然适配。

---

## 练习 6：实现 BPE 子词切分

**难度**：⭐⭐⭐  
**要改的文件**：`seq2seq-attention/data.py`

### 分析

BPE (Byte-Pair Encoding) 的核心思想：从字符开始，反复合并最频繁的符号对。最终词表由子词组成——常见词保持完整（"the"），罕见词拆成子词（"unbelievably" → "un" + "believe" + "ably"）。

### 答案代码

**data.py 新增 BPE 分词器：**

```python
from collections import Counter, defaultdict


def learn_bpe(sentences: list, num_merges: int = 8000) -> dict:
    """
    学习 BPE 合并规则

    参数：
      sentences:   训练句子列表
      num_merges:  BPE 合并次数（= 最终子词词表大小）

    返回：
      merges: {(a, b): ab} 合并规则表
    """
    # ① 初始化：每个词表示为字符序列 + 词尾标记 </w>
    word_freqs = Counter()
    for sent in sentences:
        for word in sent.strip().split():
            chars = " ".join(list(word)) + " </w>"
            word_freqs[chars] += 1

    # ② 统计所有相邻符号对
    def get_pair_counts(word_freqs):
        pairs = defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    merges = {}
    for i in range(num_merges):
        pair_counts = get_pair_counts(word_freqs)
        if not pair_counts:
            break

        # 取最高频的符号对
        best_pair = max(pair_counts, key=pair_counts.get)
        merges[best_pair] = "".join(best_pair)

        # ③ 合并：在所有词中替换该符号对
        new_word_freqs = {}
        bigram = " ".join(best_pair)
        replacement = "".join(best_pair)

        for word, freq in word_freqs.items():
            new_word = word.replace(bigram, replacement)
            new_word_freqs[new_word] = freq
        word_freqs = new_word_freqs

    return merges


def apply_bpe(text: str, merges: dict) -> list:
    """对文本应用已学好的 BPE 规则"""
    words = text.strip().split()
    result = []
    for word in words:
        chars = list(word) + ["</w>"]
        # 贪心匹配最长的合并规则
        while True:
            changed = False
            for i in range(len(chars) - 1):
                bigram = (chars[i], chars[i + 1])
                if bigram in merges:
                    chars[i] = merges[bigram]
                    chars.pop(i + 1)
                    changed = True
                    break
            if not changed:
                break
        result.extend([c for c in chars if c != "</w>"])
    return result
```

### BPE vs 传统分词的对比

| 特性 | 空格分词 | BPE 子词 |
|------|---------|---------|
| OOV 问题 | 大量 <UNK>（新词、拼写变体） | **不存在**（所有词都能拼出来） |
| 词表大小 | 10000 覆盖不到全部 | 8000 覆盖几乎全部 |
| 翻译质量 | 受限于词表 | **更好**（特别对稀有词） |
| 训练复杂度 | 简单 | 需要预学习 BPE 规则 |

### 生产环境建议

实际项目中使用 HuggingFace `tokenizers` 库或 `sentencepiece`：

```python
from tokenizers import Tokenizer, models, trainers

tokenizer = Tokenizer(models.BPE())
trainer = trainers.BpeTrainer(vocab_size=8000, special_tokens=["<PAD>", "<SOS>", "<EOS>", "<UNK>"])
tokenizer.train(["train.txt"], trainer)
```
