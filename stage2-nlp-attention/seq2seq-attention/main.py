#!/usr/bin/env python3
"""
阶段 2 · Mini 项目 4：Seq2Seq + Attention 机器翻译

核心实验：
  python main.py                                       # 默认 Bahdanau Attention
  python main.py --attn luong_general                  # Luong General Attention
  python main.py --attn luong_dot                      # Luong Dot Attention
  python main.py --attn bahdanau                       # Bahdanau Attention

运行示例：
  python main.py                                       # 默认配置，20 epochs
  python main.py --epochs 30                           # 训练 30 轮
  python main.py --embed_dim 512 --hidden_dim 1024     # 更大的模型
  python main.py --attn luong_general                  # 切换注意力机制
  python main.py --beam_size 3                         # Beam Search (size=3)
"""

import argparse
import torch
from data import (
    maybe_download_eng_fra, build_vocab, get_dataloaders,
    SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, PAD_TOKEN,
)
from encoder import Encoder
from decoder import Decoder
from train import Seq2Seq, train
from translate import translate_sentence, evaluate_bleu, plot_attention


def main():
    parser = argparse.ArgumentParser(description="Seq2Seq + Attention 翻译")
    parser.add_argument("--attn", type=str, default="bahdanau",
                        choices=["bahdanau", "luong_dot", "luong_general",
                                 "luong_concat"],
                        help="注意力类型（默认 bahdanau）")
    parser.add_argument("--epochs", type=int, default=20,
                        help="训练轮数（默认 20）")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch 大小（默认 64）")
    parser.add_argument("--embed_dim", type=int, default=256,
                        help="词向量维度（默认 256）")
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="隐藏层维度（默认 512）")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="RNN 层数（默认 2）")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="学习率（默认 0.001）")
    parser.add_argument("--tf_ratio", type=float, default=0.5,
                        help="Teacher Forcing 比例（默认 0.5）")
    parser.add_argument("--max_vocab", type=int, default=10000,
                        help="词表最大大小（默认 10000）")
    parser.add_argument("--beam_size", type=int, default=1,
                        help="Beam Search 大小（默认 1=贪心解码）")
    parser.add_argument("--max_len", type=int, default=30,
                        help="句子最大长度（默认 30）")
    args = parser.parse_args()

    # --- 自动选择设备 ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # --- 1. 加载数据 ---
    print("[1/4] 加载英语-法语平行语料...")
    pairs = maybe_download_eng_fra(max_len=args.max_len)
    print(f"  样本数: {len(pairs):,}")

    # 源语言和目标语言各自构建词表
    src_sentences = [eng for eng, _ in pairs]
    tgt_sentences = [fra for _, fra in pairs]

    src_w2i, src_i2w, src_vocab = build_vocab(src_sentences, max_vocab=args.max_vocab)
    tgt_w2i, tgt_i2w, tgt_vocab = build_vocab(tgt_sentences, max_vocab=args.max_vocab)

    print(f"  源语言词表: {src_vocab:,}")
    print(f"  目标语言词表: {tgt_vocab:,}")

    # 取特殊 token 索引
    sos_idx = tgt_w2i[SOS_TOKEN]
    eos_idx = tgt_w2i[EOS_TOKEN]
    unk_idx = tgt_w2i[UNK_TOKEN]
    pad_idx = tgt_w2i[PAD_TOKEN]

    # --- 2. 构建 DataLoader ---
    print("[2/4] 构建 DataLoader...")
    train_loader, val_loader = get_dataloaders(
        pairs, src_w2i, tgt_w2i, batch_size=args.batch_size,
    )

    # --- 3. 构建模型 ---
    print(f"[3/4] 构建 Seq2Seq + {args.attn} 模型...")

    encoder = Encoder(
        vocab_size=src_vocab,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        pad_idx=src_w2i[PAD_TOKEN],
    )

    decoder = Decoder(
        vocab_size=tgt_vocab,
        embed_dim=args.embed_dim,
        enc_hidden_dim=args.hidden_dim * 2,  # 双向 → 2 倍
        dec_hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        attn_type=args.attn,
        pad_idx=tgt_w2i[PAD_TOKEN],
    )

    model = Seq2Seq(encoder, decoder, tgt_pad_idx=pad_idx)
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    # --- 4. 训练 ---
    print("[4/4] 开始训练...")
    trained_model = train(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, device=device,
        tf_ratio=args.tf_ratio,
    )

    # --- 5. 评估 ---
    # 从训练集中取几个句子做示例翻译
    test_pairs = [(eng, fra) for eng, fra in pairs[-200:]]

    print(f"\n{'='*60}")
    print("  翻译示例 (Beam Search)" if args.beam_size > 1 else "  翻译示例 (Greedy)")
    print(f"{'='*60}")

    for i in range(3):
        src_text, ref_text = test_pairs[i]
        translated, attn_matrix = translate_sentence(
            trained_model, src_text, src_w2i, tgt_i2w,
            sos_idx, eos_idx, unk_idx, device,
            beam_size=args.beam_size,
        )
        print(f"\n  Source:     {src_text}")
        print(f"  Reference:  {ref_text}")
        print(f"  Translated: {' '.join(translated)}")

        # 第一句画注意力热力图
        if i == 0 and attn_matrix is not None:
            from data import tokenize
            src_words = tokenize(src_text)
            plot_attention(src_words, translated,
                          attn_matrix[:len(translated), :len(src_words)],
                          save_path="attention_heatmap.png")

    # BLEU 评估
    print(f"\n{'='*60}")
    print("  BLEU 评估")
    print(f"{'='*60}")
    evaluate_bleu(
        trained_model, test_pairs, src_w2i, tgt_i2w,
        sos_idx, eos_idx, unk_idx, device,
        num_samples=100, beam_size=args.beam_size,
    )


if __name__ == "__main__":
    main()
