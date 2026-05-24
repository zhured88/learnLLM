"""
GPT-2 预训练数据加载 —— WikiText-2 / 自定义文本

WikiText-2:
  - 维基百科精选文章，约 2M tokens
  - 比 PTB (Penn Treebank) 更大、更多样
  - 适合小规模语言模型预训练和验证
  - 下载 URL: https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/

核心操作：
  - 词表构建（字符级或子词级）
  - 滑动窗口切分长序列
  - 训练/验证/测试集按顺序切分（时序数据）
"""

import os
import urllib.request
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter


# ============================================================
# 特殊 token
# ============================================================

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN]

# ============================================================
# 数据加载
# ============================================================

WIKITEXT_URL = (
    "https://raw.githubusercontent.com/pytorch/examples/master/"
    "word_language_model/data/wikitext-2/"
)

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".data")


def load_wikitext() -> tuple:
    """
    加载 WikiText-2 数据集

    返回: (train_text, val_text, test_text)
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    datasets = {}
    for name in ["train", "valid", "test"]:
        fname = f"wiki.{name}.tokens"
        path = os.path.join(CACHE_DIR, fname)

        if not os.path.exists(path):
            url = WIKITEXT_URL + fname
            print(f"  下载 WikiText-2 {name}...")
            urllib.request.urlretrieve(url, path)

        with open(path, "r") as f:
            # WikiText 格式：每行一个句子，空行表示段落分隔
            text = f.read()
            # 将空行替换为段落标记（帮助模型学习段落结构）
            text = text.replace("\n\n", "\n=\n=\n")  # 段落分隔
            datasets[name] = text

    print(f"  训练集: {len(datasets['train']):,} 字符")
    print(f"  验证集: {len(datasets['valid']):,} 字符")
    print(f"  测试集: {len(datasets['test']):,} 字符")

    return datasets["train"], datasets["valid"], datasets["test"]


# ============================================================
# 词表构建（字符级 + 可选 BPE）
# ============================================================

def build_char_vocab(text: str) -> tuple:
    """
    构建字符级词表（阶段 1 LSTM 项目的相同方法）

    适合小模型快速实验。GPT-2 实际用的是 BPE (tiktoken)，
    但这里为了教学清晰度，先用字符级做 demo。
    """
    chars = sorted(list(set(text)))
    word_to_idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for ch in chars:
        if ch not in word_to_idx:
            word_to_idx[ch] = len(word_to_idx)

    idx_to_word = {i: w for w, i in word_to_idx.items()}
    return word_to_idx, idx_to_word, len(word_to_idx)


def build_word_vocab(text: str, min_count: int = 3,
                     max_vocab: int = 10000) -> tuple:
    """
    构建词级（空格分词）词表

    对于 WikiText-2 这样的小数据集，~10k 词表足够覆盖
    """
    tokens = text.split()
    counter = Counter(tokens)
    counter = {w: c for w, c in counter.items() if c >= min_count}
    sorted_words = sorted(counter.items(), key=lambda x: -x[1])[:max_vocab]

    word_to_idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for w, _ in sorted_words:
        if w not in word_to_idx:
            word_to_idx[w] = len(word_to_idx)

    idx_to_word = {i: w for w, i in word_to_idx.items()}
    return word_to_idx, idx_to_word, len(word_to_idx)


# ============================================================
# 语言模型数据集
# ============================================================

class LMDataset(Dataset):
    """
    语言模型数据集：用滑动窗口切分固定长度序列

    输入:  [tok_0, tok_1, ..., tok_{T-1}]
    标签:  [tok_1, tok_2, ..., tok_T]     ← 右移一位

    这和阶段 1 的 CharDataset 完全相同的逻辑
    """

    def __init__(self, data: torch.Tensor, seq_len: int = 256):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y


def get_dataloaders(
    train_text: str,
    val_text: str,
    word_to_idx: dict,
    batch_size: int = 32,
    seq_len: int = 256,
    level: str = "char",
):
    """
    返回 (train_loader, val_loader)

    level="char": 字符级 tokenization
    level="word": 词级（空格分词）
    """
    def encode(text):
        if level == "char":
            return torch.tensor(
                [word_to_idx.get(c, word_to_idx[UNK_TOKEN]) for c in text],
                dtype=torch.long,
            )
        else:
            tokens = text.split()
            return torch.tensor(
                [word_to_idx.get(t, word_to_idx[UNK_TOKEN]) for t in tokens],
                dtype=torch.long,
            )

    train_data = encode(train_text)
    val_data = encode(val_text)

    train_ds = LMDataset(train_data, seq_len)
    val_ds = LMDataset(val_data, seq_len)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        pin_memory=True, drop_last=True,
    )

    print(f"  训练 batch: {len(train_loader):,}")
    print(f"  验证 batch: {len(val_loader):,}")
    return train_loader, val_loader
