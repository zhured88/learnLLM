#!/usr/bin/env python3
"""
阶段 3 · Mini 项目 6：GPT-2 自回归语言模型

Decoder-Only Transformer 预训练 + 文本生成

核心实验：
  python main.py                                    # 小型 GPT (4 层, 256 维)
  python main.py --n_layer 8 --n_embd 512           # GPT-Medium (8 层, 512 维)
  python main.py --temperature 1.2 --top_k 40       # 调整生成参数

GPT-2 配置参考：
  小 Demo:  4 layers, 256 dim, 4 heads   (~8M params)   ✅ 本项目默认
  Small:   12 layers, 768 dim, 12 heads  (~124M params)
  Medium:  24 layers, 1024 dim, 16 heads (~355M params)
"""

import argparse
import torch
from data import (load_wikitext, build_char_vocab, build_word_vocab,
                  get_dataloaders, PAD_TOKEN, UNK_TOKEN)
from gpt2 import GPT2
from train import train


def main():
    parser = argparse.ArgumentParser(description="GPT-2 预训练")
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--token_level", type=str, default="char",
                        choices=["char", "word"])
    parser.add_argument("--max_vocab", type=int, default=10000)
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="梯度累积步数")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # --- 1. 加载数据 ---
    print("[1/3] 加载 WikiText-2...")
    train_text, val_text, _ = load_wikitext()

    # --- 2. 构建词表 ---
    print("[2/3] 构建词表...")
    if args.token_level == "char":
        w2i, i2w, vocab_size = build_char_vocab(train_text)
    else:
        w2i, i2w, vocab_size = build_word_vocab(train_text, max_vocab=args.max_vocab)
    print(f"  词表大小: {vocab_size:,}")

    train_loader, val_loader = get_dataloaders(
        train_text, val_text, w2i,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        level=args.token_level,
    )

    # --- 3. 构建模型 + 训练 ---
    print(f"[3/3] 构建 GPT-2 (L={args.n_layer}, D={args.n_embd}, "
          f"H={args.n_head})...")
    model = GPT2(
        vocab_size=vocab_size,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        max_seq_len=args.seq_len,
        pad_idx=w2i[PAD_TOKEN],
    )
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    trained = train(
        model, train_loader, val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        grad_accum_steps=args.grad_accum,
    )

    # --- 4. 文本生成 ---
    print(f"\n{'='*60}")
    print(f"  文本生成 (T={args.temperature}, top_k={args.top_k})")
    print(f"{'='*60}")

    prompts = ["The ", "In recent years ", "Artificial intelligence "]
    for prompt in prompts:
        # 编码 prompt
        if args.token_level == "char":
            prompt_ids = [w2i.get(c, w2i[UNK_TOKEN]) for c in prompt]
        else:
            prompt_ids = [w2i.get(w, w2i[UNK_TOKEN]) for w in prompt.split()]

        idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        generated = model.generate(
            idx, max_new_tokens=100,
            temperature=args.temperature,
            top_k=args.top_k,
        )

        # 解码
        out_ids = generated[0].tolist()
        if args.token_level == "char":
            text = "".join(i2w.get(i, UNK_TOKEN) for i in out_ids)
        else:
            text = " ".join(i2w.get(i, UNK_TOKEN) for i in out_ids)

        print(f"\n  Prompt: {prompt}")
        print(f"  Output: {text[:300]}")
        print(f"  {'─' * 60}")


if __name__ == "__main__":
    main()
