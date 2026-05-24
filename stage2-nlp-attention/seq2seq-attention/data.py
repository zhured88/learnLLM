"""
Seq2Seq 翻译数据集 —— 英法 / 英中平行语料

核心概念：
  - 平行语料 (parallel corpus)：每一行是 "源语言 \t 目标语言"
  - 词表构建：源语言和目标语言各建一个词表
  - 特殊 token：<PAD>=填充, <SOS>=句子开头, <EOS>=句子结尾, <UNK>=未知词
  - Bucketing：将相似长度的句子归入同一个 batch，减少无效 padding
"""

import os
import urllib.request
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter


# ============================================================
# 数据集下载
# ============================================================

# 英语-法语平行语料（来自 Tatoeba / manythings.org）
ENG_FRA_URL = "https://raw.githubusercontent.com/hunkim/translation-ko2/refs/heads/master/fra.txt"
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".data")
ENG_FRA_PATH = os.path.join(CACHE_DIR, "eng-fra.txt")

# 特殊 token
SOS_TOKEN = "<SOS>"   # Start Of Sentence
EOS_TOKEN = "<EOS>"   # End Of Sentence
UNK_TOKEN = "<UNK>"   # Unknown word
PAD_TOKEN = "<PAD>"   # Padding

SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


def maybe_download_eng_fra(min_len: int = 3, max_len: int = 50) -> list:
    """
    下载英语-法语平行语料，返回 [(eng_sentence, fra_sentence), ...]

    参数：
      min_len: 最短句子长度（过滤太短的）
      max_len: 最长句子长度（过滤太长的，减少显存占用）
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    if not os.path.exists(ENG_FRA_PATH):
        print("  下载英-法平行语料...")
        urllib.request.urlretrieve(ENG_FRA_URL, ENG_FRA_PATH)

    pairs = []
    with open(ENG_FRA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                eng = parts[0].lower().strip()
                fra = parts[1].lower().strip()
                # 过滤长度
                eng_words = eng.split()
                fra_words = fra.split()
                if min_len <= len(eng_words) <= max_len and min_len <= len(fra_words) <= max_len:
                    pairs.append((eng, fra))

    print(f"  加载 {len(pairs):,} 对平行句")
    return pairs


# ============================================================
# 分词 + 词表构建
# ============================================================

def tokenize(text: str) -> list:
    """简单空格分词（生产环境推荐用 Moses/Spacy）"""
    return text.strip().split()


def build_vocab(
    sentences: list,
    min_count: int = 2,
    max_vocab: int = 10000,
) -> tuple:
    """
    构建词表（包含特殊 token）

    索引分配：
      <PAD> = 0  （方便 ignore_index=0 跳过填充位置）
      <SOS> = 1
      <EOS> = 2
      <UNK> = 3
      普通词从 4 开始
    """
    counter = Counter()
    for sent in sentences:
        counter.update(tokenize(sent))

    # 过滤低频词
    counter = {w: c for w, c in counter.items() if c >= min_count}

    # 按词频排序，取 top-N
    sorted_words = sorted(counter.items(), key=lambda x: -x[1])[:max_vocab]

    word_to_idx = {}
    # 特殊 token
    for tok in SPECIAL_TOKENS:
        word_to_idx[tok] = len(word_to_idx)

    for w, _ in sorted_words:
        word_to_idx[w] = len(word_to_idx)

    idx_to_word = {i: w for w, i in word_to_idx.items()}

    return word_to_idx, idx_to_word, len(word_to_idx)


# ============================================================
# 翻译数据集
# ============================================================

class TranslationDataset(Dataset):
    """
    翻译数据集

    每个样本：(src_indices, tgt_indices)
    src: [w1, w2, ..., wn] + <EOS>
    tgt: [<SOS>, w1, w2, ..., wm, <EOS>]

    输入 decoder 时：
      decoder_input  = [<SOS>, w1, ..., wm, <EOS>]   ← 去掉最后一个
      decoder_target = [w1, w2, ..., <EOS>]           ← 去掉第一个（右移一位）
    """

    def __init__(self, pairs: list, src_word_to_idx: dict, tgt_word_to_idx: dict):
        self.pairs = pairs
        self.src_word_to_idx = src_word_to_idx
        self.tgt_word_to_idx = tgt_word_to_idx

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_text, tgt_text = self.pairs[idx]

        # 源语言：token → idx + <EOS>
        src_indices = [self.src_word_to_idx.get(w, self.src_word_to_idx[UNK_TOKEN])
                       for w in tokenize(src_text)]
        src_indices.append(self.src_word_to_idx[EOS_TOKEN])

        # 目标语言：<SOS> + token → idx + <EOS>
        tgt_indices = [self.tgt_word_to_idx[SOS_TOKEN]]
        tgt_indices += [self.tgt_word_to_idx.get(w, self.tgt_word_to_idx[UNK_TOKEN])
                        for w in tokenize(tgt_text)]
        tgt_indices.append(self.tgt_word_to_idx[EOS_TOKEN])

        return (torch.tensor(src_indices, dtype=torch.long),
                torch.tensor(tgt_indices, dtype=torch.long))


def collate_fn(batch: list, src_pad_idx: int, tgt_pad_idx: int) -> tuple:
    """
    自定义 batch 拼接函数：对变长句子做 padding

    输入：[(src_tensor, tgt_tensor), ...] 每个长度不同
    输出：(padded_src, padded_tgt, src_lengths, tgt_lengths)
    """
    src_list, tgt_list = zip(*batch)

    src_lengths = torch.tensor([len(s) for s in src_list], dtype=torch.long)
    tgt_lengths = torch.tensor([len(t) for t in tgt_list], dtype=torch.long)

    # pad_sequence：把不等长的序列补 <PAD> 到统一长度
    src_padded = pad_sequence(src_list, batch_first=True, padding_value=src_pad_idx)
    tgt_padded = pad_sequence(tgt_list, batch_first=True, padding_value=tgt_pad_idx)

    return src_padded, tgt_padded, src_lengths, tgt_lengths


def get_dataloaders(
    pairs: list,
    src_word_to_idx: dict,
    tgt_word_to_idx: dict,
    batch_size: int = 64,
    train_ratio: float = 0.95,
) -> tuple:
    """返回 (train_loader, val_loader)"""
    # 随机打散（翻译数据通常有来源顺序，不做随机的话验证集可能有偏差）
    random.seed(42)
    shuffled = pairs[:]
    random.shuffle(shuffled)

    n_train = int(len(shuffled) * train_ratio)
    train_pairs = shuffled[:n_train]
    val_pairs = shuffled[n_train:]

    src_pad = src_word_to_idx[PAD_TOKEN]
    tgt_pad = tgt_word_to_idx[PAD_TOKEN]

    train_ds = TranslationDataset(train_pairs, src_word_to_idx, tgt_word_to_idx)
    val_ds = TranslationDataset(val_pairs, src_word_to_idx, tgt_word_to_idx)

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

    print(f"  训练 batch 数: {len(train_loader):,}")
    print(f"  验证 batch 数: {len(val_loader):,}")

    return train_loader, val_loader
