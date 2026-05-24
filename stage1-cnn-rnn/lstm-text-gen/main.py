#!/usr/bin/env python3
"""
阶段 1 · Mini 项目 2：字符级 RNN/LSTM/GRU 文本生成

核心目标：
  1. 理解 RNN 的循环结构和 BPTT（沿时间反向传播）
  2. 对比 RNN / LSTM / GRU 的长序列建模能力
  3. 掌握温度采样（temperature sampling）控制文本多样性

运行：
  python main.py              # LSTM（默认）
  python main.py --rnn rnn    # Vanilla RNN
  python main.py --rnn gru    # GRU
"""

import argparse
import torch
from data import load_shakespeare, build_vocab, get_dataloaders
from rnn_gen import CharRNN
from train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rnn", type=str, default="lstm",
                        choices=["rnn", "lstm", "gru"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=100)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 数据
    print("[1/3] 加载莎士比亚文集...")
    text = load_shakespeare()
    char_to_idx, idx_to_char, vocab_size = build_vocab(text)
    print(f"  文本长度: {len(text):,} 字符, 词表大小: {vocab_size}")

    train_loader, val_loader = get_dataloaders(
        text, char_to_idx, batch_size=args.batch_size, seq_len=args.seq_len,
    )
    print(f"  训练 batch 数: {len(train_loader)}, 验证 batch 数: {len(val_loader)}")

    # 模型
    print(f"[2/3] 构建 {args.rnn.upper()} 模型...")
    model = CharRNN(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        rnn_type=args.rnn,
    )
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练
    print(f"[3/3] 训练...")
    train(
        model, train_loader, val_loader,
        char_to_idx, idx_to_char,
        epochs=args.epochs, device=device,
    )


if __name__ == "__main__":
    main()
