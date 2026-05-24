"""
Transformer 翻译数据集 —— IWSLT / Multi30k 英德/英法平行语料

核心概念：
  - 和阶段 2 的 data.py 类似，但增加了 BPE tokenization 的支持
  - Transformer 比 RNN 更依赖 batch 内句子长度的一致性
    → 使用 BucketIterator 或按长度排序减少 padding 浪费
"""

import os
import urllib.request
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter


# ============================================================
# 特殊 token
# ============================================================

PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

# ============================================================
# 数据集下载
# ============================================================

ENG_DEU_URL = "https://raw.githubusercontent.com/hunkim/translation-ko2/refs/heads/master/deu.txt"
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".data")
ENG_DEU_PATH = os.path.join(CACHE_DIR, "eng-deu.txt")


def load_parallel_corpus(
    url: str = None,
    file_path: str = None,
    min_len: int = 2,
    max_len: int = 50,
    max_pairs: int = 100000,
) -> list:
    """
    加载平行语料，返回 [(src_sentence, tgt_sentence), ...]

    支持英语-德语（IWSLT 风格）
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    if url is None:
        url = ENG_DEU_URL
    if file_path is None:
        file_path = ENG_DEU_PATH

    if not os.path.exists(file_path):
        print(f"  下载平行语料...")
        urllib.request.urlretrieve(url, file_path)

    pairs = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if len(pairs) >= max_pairs:
                break
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                src = parts[0].lower().strip()
                tgt = parts[1].lower().strip()
                if min_len <= len(src.split()) <= max_len and \
                   min_len <= len(tgt.split()) <= max_len:
                    pairs.append((src, tgt))

    print(f"  加载 {len(pairs):,} 对平行句")
    return pairs


# ============================================================
# 词表构建
# ============================================================

def tokenize(text: str) -> list:
    return text.strip().split()


def build_vocab(sentences: list, min_count: int = 2,
                max_vocab: int = 10000) -> tuple:
    """构建词表，索引 0-3 分配给特殊 token"""
    counter = Counter()
    for sent in sentences:
        counter.update(tokenize(sent))

    counter = {w: c for w, c in counter.items() if c >= min_count}
    sorted_words = sorted(counter.items(), key=lambda x: -x[1])[:max_vocab]

    word_to_idx = {}
    for tok in SPECIAL_TOKENS:
        word_to_idx[tok] = len(word_to_idx)

    for w, _ in sorted_words:
        if w not in word_to_idx:
            word_to_idx[w] = len(word_to_idx)

    idx_to_word = {i: w for w, i in word_to_idx.items()}
    return word_to_idx, idx_to_word, len(word_to_idx)


# ============================================================
# 数据集
# ============================================================

class TranslationDataset(Dataset):
    def __init__(self, pairs: list, src_w2i: dict, tgt_w2i: dict):
        self.pairs = pairs
        self.src_w2i = src_w2i
        self.tgt_w2i = tgt_w2i

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_text, tgt_text = self.pairs[idx]

        src = [self.src_w2i.get(w, self.src_w2i[UNK_TOKEN])
               for w in tokenize(src_text)]
        src.append(self.src_w2i[EOS_TOKEN])

        tgt = [self.tgt_w2i[SOS_TOKEN]]
        tgt += [self.tgt_w2i.get(w, self.tgt_w2i[UNK_TOKEN])
                for w in tokenize(tgt_text)]
        tgt.append(self.tgt_w2i[EOS_TOKEN])

        return (torch.tensor(src, dtype=torch.long),
                torch.tensor(tgt, dtype=torch.long))


def collate_fn(batch: list, src_pad: int, tgt_pad: int) -> tuple:
    src_list, tgt_list = zip(*batch)
    src_padded = pad_sequence(src_list, batch_first=True, padding_value=src_pad)
    tgt_padded = pad_sequence(tgt_list, batch_first=True, padding_value=tgt_pad)
    return src_padded, tgt_padded


def get_dataloaders(pairs: list, src_w2i: dict, tgt_w2i: dict,
                    batch_size: int = 64, train_ratio: float = 0.95):
    random.seed(42)
    shuffled = pairs[:]
    random.shuffle(shuffled)

    n_train = int(len(shuffled) * train_ratio)
    train_pairs = shuffled[:n_train]
    val_pairs = shuffled[n_train:]

    src_pad = src_w2i[PAD_TOKEN]
    tgt_pad = tgt_w2i[PAD_TOKEN]

    train_ds = TranslationDataset(train_pairs, src_w2i, tgt_w2i)
    val_ds = TranslationDataset(val_pairs, src_w2i, tgt_w2i)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, src_pad, tgt_pad),
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, src_pad, tgt_pad),
        pin_memory=True, drop_last=True,
    )

    print(f"  训练 batch: {len(train_loader):,}, 验证 batch: {len(val_loader):,}")
    return train_loader, val_loader
