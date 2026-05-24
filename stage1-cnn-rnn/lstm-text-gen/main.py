#!/usr/bin/env python3
"""
阶段 1 · Mini 项目 2：字符级 RNN/LSTM/GRU 文本生成

核心实验（对比三种 RNN 架构）：
  python main.py --rnn lstm    # LSTM：三道门控，长程记忆最强
  python main.py --rnn gru     # GRU：两道门控，参数更少，效果接近 LSTM
  python main.py --rnn rnn     # Vanilla RNN：没有门控，长序列训不动

运行示例：
  python main.py                              # LSTM，默认 20 轮
  python main.py --rnn lstm --epochs 30       # LSTM，30 轮
  python main.py --rnn gru  --epochs 20       # GRU
  python main.py --rnn rnn  --epochs 20       # Vanilla RNN（对比实验）
  python main.py --num_layers 4               # 4 层堆叠的深度 LSTM
"""

import argparse
import torch
from data import load_shakespeare, build_vocab, get_dataloaders
from rnn_gen import CharRNN
from train import train


def main():
    # --- 命令行参数 ---
    parser = argparse.ArgumentParser(description="字符级 RNN 莎士比亚文本生成")
    parser.add_argument("--rnn", type=str, default="lstm",
                        choices=["rnn", "lstm", "gru"],
                        help="RNN 架构类型（默认 lstm）")
    parser.add_argument("--epochs", type=int, default=20,
                        help="训练轮数（默认 20）")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch 大小（默认 64）")
    parser.add_argument("--seq_len", type=int, default=100,
                        help="序列长度 / RNN 展开的时间步数（默认 100）")
    parser.add_argument("--embed_dim", type=int, default=256,
                        help="嵌入向量维度（默认 256）")
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="RNN 隐藏状态维度（默认 512）")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="RNN 堆叠层数（默认 2）")
    args = parser.parse_args()

    # --- 自动选择设备（GPU 优先：CUDA > MPS > CPU）---
    if torch.cuda.is_available():
        device = torch.device("cuda")           # NVIDIA GPU
    elif torch.backends.mps.is_available():
        device = torch.device("mps")            # Apple Silicon GPU（M1/M2/M3/M4）
    else:
        device = torch.device("cpu")            # 纯 CPU 兜底
    print(f"Device: {device}")

    # --- 1. 加载数据 ---
    print("[1/3] 加载莎士比亚文集...")
    text = load_shakespeare()
    char_to_idx, idx_to_char, vocab_size = build_vocab(text)
    print(f"  文本长度: {len(text):,} 字符")
    print(f"  词表大小: {vocab_size} 个不同字符")

    train_loader, val_loader = get_dataloaders(
        text, char_to_idx, batch_size=args.batch_size, seq_len=args.seq_len,
    )
    print(f"  训练 batch 数: {len(train_loader):,}")
    print(f"  验证 batch 数: {len(val_loader):,}")

    # --- 2. 构建模型 ---
    print(f"[2/3] 构建 {args.rnn.upper()} 模型...")
    model = CharRNN(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        rnn_type=args.rnn,
    )
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    # --- 3. 训练 ---
    print(f"[3/3] 开始训练（每 5 轮生成一段样本文本）...")
    train(
        model, train_loader, val_loader,
        char_to_idx, idx_to_char,
        epochs=args.epochs, device=device,
    )


if __name__ == "__main__":
    main()
