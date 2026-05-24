"""莎士比亚文本数据加载"""

import os
import urllib.request
import torch
from torch.utils.data import Dataset, DataLoader


SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
CACHE_PATH = os.path.join(os.path.dirname(__file__), ".shakespeare.txt")


def load_shakespeare() -> str:
    """下载并返回莎士比亚文本"""
    if not os.path.exists(CACHE_PATH):
        print("  下载莎士比亚文集...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, CACHE_PATH)

    with open(CACHE_PATH, "r") as f:
        return f.read()


def build_vocab(text: str) -> tuple:
    """构建字符级词表，返回 (char_to_idx, idx_to_char, vocab_size)"""
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    return char_to_idx, idx_to_char, len(chars)


class CharDataset(Dataset):
    """字符级数据集：滑动窗口切成 (seq_len) 的片段"""

    def __init__(self, text: str, char_to_idx: dict, seq_len: int = 100):
        self.seq_len = seq_len
        self.data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y


def get_dataloaders(
    text: str, char_to_idx: dict, batch_size: int = 64, seq_len: int = 100,
    train_ratio: float = 0.9,
):
    """返回 train_loader, val_loader"""
    n = int(len(text) * train_ratio)
    train_text = text[:n]
    val_text = text[n:]

    train_ds = CharDataset(train_text, char_to_idx, seq_len)
    val_ds = CharDataset(val_text, char_to_idx, seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            pin_memory=True, drop_last=True)
    return train_loader, val_loader
