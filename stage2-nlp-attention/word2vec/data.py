"""
Word2Vec 数据加载 —— Text8 / 中文维基百科

核心概念：
  - 词表构建：统计词频 → 过滤低频词 → 建立 word↔idx 映射
  - 二次采样 (subsampling)：以概率 P(wi) = 1 - sqrt(t/f(wi)) 丢弃高频词
    原理：高频词（如"的"、"the"）信息量低，保留所有出现会浪费算力
  - 负采样表：构造一个按词频^0.75 加权的词表，快速采样"噪声词"
    原理：词频^0.75 平滑了分布——低频词被采样的概率稍微提高，高频词稍微降低
"""

import os
import urllib.request
import random
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


# ============================================================
# 数据集下载
# ============================================================

TEXT8_URL = "http://mattmahoney.net/dc/text8.zip"
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".data")
TEXT8_PATH = os.path.join(CACHE_DIR, "text8")


def maybe_download_text8():
    """下载 Text8 数据集（英文，100MB 维基百科清洗文本）"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    zip_path = os.path.join(CACHE_DIR, "text8.zip")

    if not os.path.exists(TEXT8_PATH):
        if not os.path.exists(zip_path):
            print("  下载 Text8 数据集 (~30MB)...")
            urllib.request.urlretrieve(TEXT8_URL, zip_path)

        print("  解压 Text8...")
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(CACHE_DIR)

    with open(TEXT8_PATH, "r") as f:
        return f.read().strip().split()


# ============================================================
# 词表构建
# ============================================================

def build_vocab(
    tokens: list,
    min_count: int = 5,
    max_vocab: int = 50000,
) -> tuple:
    """
    构建词表

    参数：
      tokens:    原始分词列表
      min_count: 最低词频（出现次数少于此值的词被映射为 <UNK>）
      max_vocab: 最大词表大小（按词频取 top-N）

    返回：
      word_to_idx:  {'the': 0, 'of': 1, ...}
      idx_to_word:  {0: 'the', 1: 'of', ...}
      word_counts:  [('the', 50000), ('of', 30000), ...]
    """
    # 统计词频
    counter = Counter(tokens)
    print(f"  原始词数: {len(counter):,}")

    # 过滤低频词
    counter = {w: c for w, c in counter.items() if c >= min_count}
    print(f"  过滤后词数 (min_count={min_count}): {len(counter):,}")

    # 按词频降序排列，取前 max_vocab 个
    word_counts = sorted(counter.items(), key=lambda x: -x[1])[:max_vocab]
    print(f"  最终词表大小: {len(word_counts):,}")

    # 构建映射：高频词索引小 = embedding 查表更快
    word_to_idx = {w: i for i, (w, _) in enumerate(word_counts)}
    idx_to_word = {i: w for w, i in word_to_idx.items()}

    return word_to_idx, idx_to_word, word_counts


# ============================================================
# 二次采样 (Subsampling)
# ============================================================

def make_subsample_table(
    word_counts: list,
    threshold: float = 1e-3,
) -> np.ndarray:
    """
    构建二次采样丢弃概率表

    公式：P_drop(w) = 1 - sqrt(t / f(w))
    其中 t=threshold, f(w)=词频/总词数

    直观理解：
      - "the" 词频=5% → P_drop = 1 - sqrt(0.001/0.05) = 0.86 → 86% 概率丢弃
      - "dog" 词频=0.001% → P_drop < 0 → 保留（低频词不丢弃）

    返回：
      drop_probs: 长度 vocab_size 的 numpy 数组，drop_probs[i] = 词 i 的丢弃概率
    """
    total_count = sum(c for _, c in word_counts)
    freq = np.array([c / total_count for _, c in word_counts], dtype=np.float32)
    drop_probs = 1.0 - np.sqrt(threshold / (freq + 1e-12))
    return np.clip(drop_probs, 0, 1)  # 裁剪到 [0, 1]


def apply_subsample(tokens: list, word_to_idx: dict, drop_probs: np.ndarray) -> list:
    """对 tokens 列表应用二次采样，返回过滤后的 token 索引列表"""
    idxs = []
    for token in tokens:
        if token in word_to_idx:
            idx = word_to_idx[token]
            if random.random() > drop_probs[idx]:
                idxs.append(idx)
    return idxs


# ============================================================
# 负采样表
# ============================================================

def make_noise_dist(word_counts: list, power: float = 0.75) -> np.ndarray:
    """
    构建负采样用噪声分布

    公式：P(w) = count(w)^power / Σ count(w')^power

    power=0.75 的效果：
      - 原词频比 100:1 → 噪声分布比 (100^0.75):(1^0.75) ≈ 31:1
      - 低频词被采样的概率相对提高了

    返回：
      noise_dist: 归一化概率数组
    """
    counts = np.array([c for _, c in word_counts], dtype=np.float64)
    counts_pow = counts ** power
    return counts_pow / counts_pow.sum()


# ============================================================
# Skip-gram 数据集
# ============================================================

class SkipGramDataset(Dataset):
    """
    Skip-gram 数据集：对每个中心词，采样其上下文窗口内的词作为正样本

    参数：
      window_size: 上下文窗口半径（每个中心词左右各取 window_size 个词）
      num_negative: 每个正样本配多少个负样本

    示例（文本 "the cat sat on the mat", window_size=2）：
      中心词="sat" (idx=2) → 上下文=[cat, on, the, mat] (idx=1,3,4,5)
      产生 4 个 (center, context, label=1) 样本
      label=1 表示正样本
    """

    def __init__(
        self,
        token_ids: list,
        window_size: int = 5,
        num_negative: int = 5,
        noise_dist: np.ndarray = None,
    ):
        self.token_ids = token_ids
        self.window_size = window_size
        self.num_negative = num_negative
        self.noise_dist = noise_dist
        self.vocab_size = len(noise_dist) if noise_dist is not None else 0

        # 预计算所有 (center, context) 对
        self.pairs = []
        for i in range(window_size, len(token_ids) - window_size):
            center = token_ids[i]
            for j in range(i - window_size, i + window_size + 1):
                if j != i:
                    self.pairs.append((center, token_ids[j]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        返回：
          center:      标量，中心词索引
          context:     标量，正样本上下文词索引
          neg_samples: (num_negative,) 负样本词索引
        """
        center, context = self.pairs[idx]

        # 负采样：从噪声分布中采 num_negative 个词
        neg_samples = np.random.choice(
            self.vocab_size, size=self.num_negative,
            replace=True, p=self.noise_dist,
        )

        return (
            torch.tensor(center, dtype=torch.long),
            torch.tensor(context, dtype=torch.long),
            torch.tensor(neg_samples, dtype=torch.long),
        )


def get_dataloader(
    token_ids: list,
    window_size: int = 5,
    num_negative: int = 5,
    noise_dist: np.ndarray = None,
    batch_size: int = 512,
) -> DataLoader:
    """创建 Skip-gram DataLoader"""
    dataset = SkipGramDataset(token_ids, window_size, num_negative, noise_dist)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      pin_memory=True, drop_last=True)
