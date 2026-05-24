"""
莎士比亚文本数据加载 + 字符级数据集构建

核心概念：
  - 字符级词表：不是按"词"分词，而是按"字符"分词（A-Z, a-z, 标点, 空格 = 65 个类别）
  - 滑动窗口：111 万字符的文本切成 100 字符长的重叠片段，每个片段是一个训练样本
  - 标签右移：输入和标签完全重叠但差 1 位 → 模型学习"看到前面的字符，预测下一个字符"
"""

import os
import urllib.request
import torch
from torch.utils.data import Dataset, DataLoader


# 莎士比亚文集下载地址（Andrej Karpathy 的 char-rnn 仓库）
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
CACHE_PATH = os.path.join(os.path.dirname(__file__), ".shakespeare.txt")


def load_shakespeare() -> str:
    """
    下载（首次运行时）并返回莎士比亚文集全文

    数据集内容：莎士比亚所有戏剧的台词文本，约 111 万字符
    """
    if not os.path.exists(CACHE_PATH):
        print("  下载莎士比亚文集...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, CACHE_PATH)

    with open(CACHE_PATH, "r") as f:
        return f.read()


def build_vocab(text: str) -> tuple:
    """
    构建字符级词表

    原理：
      不做传统的"分词"（如把 "Hello" 分成 "He" "llo"），
      而是把每个字符当作一个独立的"词"。莎士比亚文集只有 65 个不同字符。

    返回：
      char_to_idx:  {'\n': 0, ' ': 1, 'A': 2, ...}  字符 → 索引
      idx_to_char:  {0: '\n', 1: ' ', 2: 'A', ...}  索引 → 字符
      vocab_size:   65
    """
    # sorted(set(text))：去重 → 排序 → 每个字符有了一个固定的索引
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    return char_to_idx, idx_to_char, len(chars)


class CharDataset(Dataset):
    """
    字符级数据集：用滑动窗口把长文本切成固定长度的训练样本

    示例（文本 "Hello World"，seq_len=4）：
      样本 0: x=[H, e, l, l], y=[e, l, l, o]   ← 看到 H e l l，预测 e l l o
      样本 1: x=[e, l, l, o], y=[l, l, o,  ]
      样本 2: x=[l, l, o,  ], y=[l, o,  , W]

    关键设计：
      - y 是 x 右移一位 → 每个位置都在预测"下一个字符是什么"
      - 重叠窗口 → 111 万字符产生 ~111 万个训练样本，充分利用数据
    """

    def __init__(self, text: str, char_to_idx: dict, seq_len: int = 100):
        """
        参数：
          text:        原始文本字符串
          char_to_idx: 字符→索引的映射字典
          seq_len:     每个样本的序列长度（= RNN 展开的时间步数）
        """
        self.seq_len = seq_len
        # 整段文本 → 一维长向量：111 万字符 → 111 万个整数
        self.data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)

    def __len__(self):
        """总样本数 = 文本长度 - 序列长度（因为每个样本需要 seq_len+1 个字符）"""
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        """
        取第 idx 个样本

        x: 从 idx 开始，取 100 个字符（输入）
        y: 从 idx+1 开始，取 100 个字符（标签 = 每个位置的下一个字符）
        """
        x = self.data[idx : idx + self.seq_len]           # 输入：[a, b, c, d, ...]
        y = self.data[idx + 1 : idx + self.seq_len + 1]   # 标签：[b, c, d, e, ...] 右移一位
        return x, y


def get_dataloaders(
    text: str, char_to_idx: dict, batch_size: int = 64, seq_len: int = 100,
    train_ratio: float = 0.9,  # 90% 数据用于训练，10% 用于验证
):
    """
    返回 (train_loader, val_loader)

    按 90/10 切分训练集和验证集（时序数据不能随机切，要按顺序切）
    """
    # --- 按顺序切分数据 ---
    # 不能随机切！文本是有时序的，随机切会让验证集和训练集的内容高度重叠
    n = int(len(text) * train_ratio)
    train_text = text[:n]      # 前 90% 训练
    val_text = text[n:]        # 后 10% 验证

    train_ds = CharDataset(train_text, char_to_idx, seq_len)
    val_ds = CharDataset(val_text, char_to_idx, seq_len)

    # drop_last=True：丢弃最后一个不完整的 batch（当数据集不能被 batch_size 整除时）
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            pin_memory=True, drop_last=True)
    return train_loader, val_loader
