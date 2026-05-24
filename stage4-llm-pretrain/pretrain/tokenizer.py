"""
BPE Tokenizer — 使用 sentencepiece 训练和加载

Stage 3 使用字符级/词级 tokenizer，Stage 4 升级到 BPE (Byte-Pair Encoding)。

优先级:
  1. 使用 sentencepiece 训练 BPE（如果安装了 sentencepiece 且有语料）
  2. 回退到 HuggingFace GPT-2 tokenizer（如果安装了 transformers）
  3. 回退到简单字符级 tokenizer（纯 Python 兜底）

特殊 token:
  <pad>: 0
  <unk>: 1
  <bos>: 2
  <eos>: 3
"""

import os
import json
import pickle


class BPETokenizer:
    """BPE Tokenizer 封装。

    统一接口:
      encode(text) -> list[int]
      decode(ids) -> str
    """

    def __init__(self, model_path: str = None, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3

        self._sp = None        # sentencepiece 模型
        self._huggingface = None  # HuggingFace GPT-2 tokenizer
        self._char_vocab = None   # 回退: 字符级

        if model_path and os.path.exists(model_path):
            self._load_sp(model_path)

    def _load_sp(self, model_path: str):
        """加载 sentencepiece 模型。"""
        try:
            import sentencepiece as spm
            self._sp = spm.SentencePieceProcessor()
            self._sp.load(model_path)
            self.vocab_size = self._sp.get_piece_size()
        except ImportError:
            print("[WARN] sentencepiece 未安装，尝试 HuggingFace tokenizer")
            self._init_huggingface()

    def _init_huggingface(self):
        """使用 HuggingFace GPT-2 tokenizer。"""
        try:
            from transformers import AutoTokenizer
            self._huggingface = AutoTokenizer.from_pretrained("gpt2")
            self._huggingface.pad_token = self._huggingface.eos_token
            self.vocab_size = self._huggingface.vocab_size
        except ImportError:
            print("[WARN] transformers 未安装，回退到字符级 tokenizer")

    def _init_char_vocab(self, text: str):
        """以字符级 tokenizer 兜底（最后手段）。"""
        chars = sorted(set(text))
        self._char_vocab = {ch: i + 4 for i, ch in enumerate(chars)}  # 0-3 留给 special tokens
        self.vocab_size = len(self._char_vocab) + 4

    def encode(self, text: str) -> list:
        if self._sp:
            ids = self._sp.encode(text, out_type=int)
            return [self.bos_id] + ids + [self.eos_id]
        elif self._huggingface:
            ids = self._huggingface.encode(text)
            return ids
        elif self._char_vocab:
            return [self.bos_id] + [
                self._char_vocab.get(ch, self.unk_id) for ch in text
            ] + [self.eos_id]
        else:
            # 最坏情况: 按字节编码
            return [self.bos_id] + [ord(ch) % 256 + 4 for ch in text] + [self.eos_id]

    def decode(self, ids: list) -> str:
        if self._sp:
            return self._sp.decode(ids)
        elif self._huggingface:
            return self._huggingface.decode(ids, skip_special_tokens=True)
        elif self._char_vocab:
            id_to_char = {v: k for k, v in self._char_vocab.items()}
            id_to_char[0] = "<PAD>"
            id_to_char[1] = "<UNK>"
            id_to_char[2] = "<BOS>"
            id_to_char[3] = "<EOS>"
            return "".join(id_to_char.get(i, "<UNK>") for i in ids)
        else:
            return "".join(chr(i - 4) if i >= 4 else "" for i in ids)

    def train(self, text_file: str, model_prefix: str = "tokenizer",
              vocab_size: int = 32000):
        """使用 sentencepiece 训练 BPE tokenizer。

        Args:
            text_file: 原始文本文件路径（一行一句）
            model_prefix: 输出模型前缀
            vocab_size: 词汇表大小
        """
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError("请安装 sentencepiece: pip install sentencepiece")

        spm.SentencePieceTrainer.train(
            input=text_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
            character_coverage=1.0,
            max_sentence_length=4096,
            num_threads=4,
        )

        # 加载刚训练的模型
        model_path = f"{model_prefix}.model"
        self._load_sp(model_path)
        print(f"Tokenizer 训练完成: {model_path} (vocab_size={self.vocab_size})")


def get_tokenizer(model_path: str = None, vocab_size: int = 32000,
                  text_file: str = None) -> BPETokenizer:
    """获取 tokenizer：优先加载已有模型，否则回退。

    Args:
        model_path: 已有 sentencepiece 模型路径
        vocab_size: 词汇量
        text_file: 用于训练 tokenizer 的文本文件

    Returns:
        BPETokenizer 实例
    """
    tokenizer = BPETokenizer(model_path=model_path, vocab_size=vocab_size)

    # 如果提供了 text_file 但没有 sentencepiece 模型，用 HuggingFace 或字符级
    if text_file and tokenizer._sp is None and tokenizer._huggingface is None:
        if os.path.exists(text_file):
            with open(text_file, "r", encoding="utf-8") as f:
                text = f.read()
            tokenizer._init_char_vocab(text)
            print(f"[INFO] 使用字符级 tokenizer (vocab_size={tokenizer.vocab_size})")

    if tokenizer._huggingface:
        print(f"[INFO] 使用 HuggingFace GPT-2 tokenizer (vocab_size={tokenizer.vocab_size})")

    return tokenizer
