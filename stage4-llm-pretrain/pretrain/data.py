"""
LLM 预训练数据加载

Stage 4 升级点:
  - 使用 BPE tokenizer 替代字符/词级
  - 支持预 tokenize + 缓存（避免每 epoch 重新 tokenize）
  - 更大序列长度 (2048)
  - 支持 WikiText-103 和其他语料

数据流:
  原始文本 -> tokenizer.encode() -> .bin 缓存 -> LMDataset -> DataLoader
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


CACHE_DIR = os.path.join(os.path.dirname(__file__), ".data")


class LMDataset(Dataset):
    """语言模型数据集：滑动窗口取 (x, y) 对。

    x[i] = data[i : i+seq_len]
    y[i] = data[i+1 : i+seq_len+1]
    """

    def __init__(self, data: torch.Tensor, seq_len: int = 2048):
        self.data = data
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx) -> tuple:
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y


def download_wikitext103(cache_dir: str = None) -> dict:
    """下载 WikiText-103 数据集。

    WikiText-103: ~100M tokens，适合 ~100M 参数模型的预训练。

    Returns:
        {"train": text, "val": text, "test": text}
    """
    if cache_dir is None:
        cache_dir = CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    import urllib.request

    base_url = "https://s3.amazonaws.com/research.metamind.io/wikitext/"
    files = {
        "train": "wikitext-103-raw-v1.zip",
        "val":   "wikitext-103-raw-v1.zip",  # 同一个 zip 包含
        "test":  "wikitext-103-raw-v1.zip",
    }

    zip_path = os.path.join(cache_dir, "wikitext-103-raw-v1.zip")
    extract_dir = os.path.join(cache_dir, "wikitext-103")

    if not os.path.exists(extract_dir):
        if not os.path.exists(zip_path):
            print(f"下载 WikiText-103 到 {zip_path} ...")
            urllib.request.urlretrieve(f"{base_url}wikitext-103-raw-v1.zip", zip_path)

        import zipfile
        print(f"解压到 {extract_dir} ...")
        with zipfile.ZipFile(zip_path, "r") as f:
            f.extractall(extract_dir)

    # 读取各 split
    result = {}
    for split, fname in [("train", "wiki.train.raw"),
                          ("val", "wiki.valid.raw"),
                          ("test", "wiki.test.raw")]:
        path = os.path.join(extract_dir, fname)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            result[split] = text
            print(f"  {split}: {len(text):,} 字符")

    return result


def download_wikitext2(cache_dir: str = None) -> dict:
    """下载 WikiText-2（Demo 用，~2M tokens）。"""
    if cache_dir is None:
        cache_dir = CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    import urllib.request

    base_url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/"
    result = {}

    for split, fname in [("train", "wiki.train.tokens"),
                          ("val", "wiki.valid.tokens"),
                          ("test", "wiki.test.tokens")]:
        path = os.path.join(cache_dir, fname)
        if not os.path.exists(path):
            urllib.request.urlretrieve(f"{base_url}{fname}", path)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        result[split] = text
        print(f"  {split}: {len(text):,} 字符")

    return result


def tokenize_and_cache(text: str, tokenizer, cache_path: str) -> torch.Tensor:
    """将文本 tokenize 并缓存到 .bin 文件。

    Args:
        text: 原始文本
        tokenizer: BPETokenizer 实例
        cache_path: 缓存文件路径 (.bin)

    Returns:
        (N,) tensor of token ids
    """
    if os.path.exists(cache_path):
        print(f"  加载缓存: {cache_path}")
        return torch.from_numpy(np.fromfile(cache_path, dtype=np.uint16).astype(np.int64))

    print(f"  Tokenizing ({len(text):,} chars) ...")
    ids = tokenizer.encode(text)
    data = np.array(ids, dtype=np.uint16)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    data.tofile(cache_path)
    print(f"  缓存保存: {cache_path} ({len(ids):,} tokens)")

    return torch.from_numpy(data.astype(np.int64))


def get_dataloaders(tokenizer, dataset: str = "wikitext2",
                    batch_size: int = 16, seq_len: int = 2048,
                    num_workers: int = 2) -> tuple:
    """获取训练和验证 DataLoader。

    Args:
        tokenizer: BPETokenizer 实例
        dataset: "wikitext2" 或 "wikitext103"
        batch_size: batch size
        seq_len: 序列长度
        num_workers: DataLoader workers

    Returns:
        (train_loader, val_loader)
    """
    if dataset == "wikitext2":
        raw = download_wikitext2()
    elif dataset == "wikitext103":
        raw = download_wikitext103()
    else:
        raise ValueError(f"未知数据集: {dataset}")

    cache_dir = os.path.join(CACHE_DIR, dataset)
    os.makedirs(cache_dir, exist_ok=True)

    loaders = {}
    for split, text in [("train", raw["train"]), ("val", raw.get("val", raw.get("valid", "")))]:
        cache_path = os.path.join(cache_dir, f"{split}.bin")
        data = tokenize_and_cache(text, tokenizer, cache_path)

        ds = LMDataset(data, seq_len=seq_len)
        shuffle = (split == "train")
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            pin_memory=True, drop_last=True,
            num_workers=num_workers
        )

    return loaders.get("train"), loaders.get("val")
