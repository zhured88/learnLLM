#!/usr/bin/env python3
"""
阶段 3 · Mini 项目 5：从零实现 Transformer 机器翻译

基于 "Attention Is All You Need" (Vaswani et al., 2017)

核心实验：
  python main.py                          # 默认配置 (d_model=512, 6 layers, 8 heads)
  python main.py --epochs 30             # 训练 30 轮
  python main.py --d_model 256 --num_heads 4 --num_layers 4  # 小型 Transformer
  python main.py --beam_size 5           # Beam Search 解码

关键参数（原始论文 base 配置）：
  d_model=512, num_heads=8, num_layers=6, d_ff=2048
"""

import argparse
import torch
from data import (
    load_parallel_corpus, build_vocab, get_dataloaders,
    PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN,
)
from transformer import Transformer
from train import train
from translate import translate_sentence, compute_bleu


def main():
    parser = argparse.ArgumentParser(description="Transformer NMT")
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_vocab", type=int, default=10000)
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=4000)
    parser.add_argument("--smoothing", type=float, default=0.1)
    parser.add_argument("--beam_size", type=int, default=1)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # --- 1. 加载数据 ---
    print("[1/4] 加载平行语料...")
    pairs = load_parallel_corpus(max_len=args.max_len)

    src_sentences = [s for s, _ in pairs]
    tgt_sentences = [t for _, t in pairs]
    src_w2i, src_i2w, src_vocab = build_vocab(src_sentences, max_vocab=args.max_vocab)
    tgt_w2i, tgt_i2w, tgt_vocab = build_vocab(tgt_sentences, max_vocab=args.max_vocab)
    pad_idx = src_w2i[PAD_TOKEN]

    print(f"  源词表: {src_vocab:,}, 目标词表: {tgt_vocab:,}")

    # --- 2. DataLoader ---
    print("[2/4] 构建 DataLoader...")
    train_loader, val_loader = get_dataloaders(
        pairs, src_w2i, tgt_w2i, batch_size=args.batch_size,
    )

    # --- 3. 模型 ---
    print(f"[3/4] 构建 Transformer (d={args.d_model}, h={args.num_heads}, "
          f"L={args.num_layers}, ff={args.d_ff})...")
    model = Transformer(
        src_vocab, tgt_vocab,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        pad_idx=pad_idx,
    )
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    # --- 4. 训练 ---
    print("[4/4] 开始训练...")
    trained = train(
        model, train_loader, val_loader,
        epochs=args.epochs,
        d_model=args.d_model,
        warmup_steps=args.warmup,
        smoothing=args.smoothing,
        device=device,
    )

    # --- 5. 翻译示例 ---
    test_pairs = pairs[-50:]
    sos_idx = tgt_w2i[SOS_TOKEN]
    eos_idx = tgt_w2i[EOS_TOKEN]

    print(f"\n{'='*60}")
    print(f"  翻译示例 (Beam={args.beam_size})")
    print(f"{'='*60}")

    bleu_scores = []
    for i in range(min(5, len(test_pairs))):
        src_text, ref_text = test_pairs[i]
        translated, _ = translate_sentence(
            trained, src_text, src_w2i, tgt_i2w, device,
            beam_size=args.beam_size,
        )

        ref_tokens = tokenize(ref_text)
        bleu = compute_bleu(ref_tokens, translated)
        bleu_scores.append(bleu)

        print(f"\n  Source:     {src_text}")
        print(f"  Reference:  {ref_text}")
        print(f"  Translated: {' '.join(translated)}")
        print(f"  BLEU:       {bleu:.4f}")

    if bleu_scores:
        print(f"\n  平均 BLEU (5 句): {sum(bleu_scores)/len(bleu_scores):.4f}")


if __name__ == "__main__":
    main()
