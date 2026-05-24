"""
Stage 4 · LLaMA 预训练入口

从零开始训练一个 ~100M 参数的 LLaMA 语言模型。

预置配置:
  tiny:  25M 参数 (n_embd=384,  n_layer=8,  n_head=6,  n_kv_head=2) — CPU 可运行
  small: 55M 参数 (n_embd=576,  n_layer=12, n_head=9,  n_kv_head=3) — 需 GPU
  base: 105M 参数 (n_embd=768,  n_layer=12, n_head=12, n_kv_head=4) — 需 GPU 24GB+

快速开始:
  python main.py                                    # tiny 模型, WikiText-2, 3 epochs
  python main.py --model small --dataset wikitext103 # small 模型, WikiText-103
  python main.py --model base --batch_size 8 --grad_accum 8  # 模拟 batch=64
"""

import argparse
import torch

from model import LLaMA, create_model, MODEL_CONFIGS
from tokenizer import get_tokenizer
from data import get_dataloaders
from train import train


def main():
    parser = argparse.ArgumentParser(description="Stage 4 · LLaMA 预训练")

    # 模型参数
    parser.add_argument("--model", type=str, default="tiny",
                        choices=list(MODEL_CONFIGS.keys()),
                        help="模型大小预设")
    parser.add_argument("--vocab_size", type=int, default=32000,
                        help="词汇表大小")
    parser.add_argument("--max_seq_len", type=int, default=2048,
                        help="最大序列长度")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Dropout 率")

    # 数据参数
    parser.add_argument("--dataset", type=str, default="wikitext2",
                        choices=["wikitext2", "wikitext103"],
                        help="训练数据集")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="每步 batch size")
    parser.add_argument("--seq_len", type=int, default=2048,
                        help="序列长度")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="DataLoader workers")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=3,
                        help="训练轮数")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="峰值学习率")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                        help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="线性 warmup 步数")
    parser.add_argument("--min_lr", type=float, default=1e-6,
                        help="最低学习率")
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="梯度累积步数")
    parser.add_argument("--clip_grad", type=float, default=1.0,
                        help="梯度裁剪阈值")

    # 其他
    parser.add_argument("--no_amp", action="store_true",
                        help="禁用混合精度")
    parser.add_argument("--resume", type=str, default=None,
                        help="恢复检查点路径")
    parser.add_argument("--checkpoint_interval", type=int, default=1,
                        help="保存检查点的 epoch 间隔")

    args = parser.parse_args()

    # ── 设备 ───────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"设备: CUDA · {gpu_name} ({gpu_mem:.1f} GB)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("设备: MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("设备: CPU")

    # ── Tokenizer ─────────────────────────────────────────
    print("\n[1/4] 加载 Tokenizer ...")
    tokenizer = get_tokenizer(vocab_size=args.vocab_size)
    actual_vocab_size = tokenizer.vocab_size
    print(f"  Vocab size: {actual_vocab_size}")

    # ── 数据 ─────────────────────────────────────────────
    print(f"\n[2/4] 加载数据 ({args.dataset}) ...")
    train_loader, val_loader = get_dataloaders(
        tokenizer, dataset=args.dataset,
        batch_size=args.batch_size, seq_len=args.seq_len,
        num_workers=args.num_workers
    )
    effective_batch = args.batch_size * args.grad_accum * args.seq_len
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Effective batch size: {args.batch_size} × {args.grad_accum} = {effective_batch} tokens/step")

    # ── 模型 ─────────────────────────────────────────────
    print(f"\n[3/4] 创建模型 ({args.model}) ...")
    cfg = MODEL_CONFIGS[args.model]
    print(f"  配置: n_embd={cfg['n_embd']}, n_layer={cfg['n_layer']}, "
          f"n_head={cfg['n_head']}, n_kv_head={cfg['n_kv_head']}")

    model = LLaMA(
        vocab_size=actual_vocab_size,
        n_embd=cfg["n_embd"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_kv_head=cfg["n_kv_head"],
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
    )

    # 参数量
    params = model.count_parameters()
    print(f"  总参数:      {params['total']:>12,} ({params['total']/1e6:.1f}M)")
    print(f"  Embedding:   {params['embedding']:>12,}")
    print(f"  Non-Embed:   {params['non_embedding']:>12,}")

    # 模型结构摘要
    print(f"\n  模型结构:")
    print(f"  ├── Token Embedding: {actual_vocab_size} × {cfg['n_embd']}")
    print(f"  ├── LLaMA Blocks × {cfg['n_layer']}:")
    print(f"  │   ├── RMSNorm → GQA (n_head={cfg['n_head']}, n_kv={cfg['n_kv_head']}) + RoPE")
    print(f"  │   └── RMSNorm → SwiGLU")
    print(f"  ├── RMSNorm (final)")
    print(f"  └── LM Head: {cfg['n_embd']} × {actual_vocab_size}")

    # ── 训练 ─────────────────────────────────────────────
    print(f"\n[4/4] 开始训练 ({args.epochs} epochs) ...")
    print(f"  LR: {args.lr:.1e}, Warmup: {args.warmup_steps} steps, Min LR: {args.min_lr:.1e}")
    print(f"  {'AMP ON' if not args.no_amp else 'AMP OFF'}, "
          f"Grad Accum: {args.grad_accum}x, Clip: {args.clip_grad}")

    model = train(
        model, train_loader, val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        min_lr=args.min_lr,
        grad_accum_steps=args.grad_accum,
        clip_grad=args.clip_grad,
        use_amp=not args.no_amp,
        device=device,
        checkpoint_interval=args.checkpoint_interval,
        resume_from=args.resume,
        model_size=args.model,
    )

    print("\n训练完成!")


if __name__ == "__main__":
    main()
