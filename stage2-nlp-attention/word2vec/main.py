#!/usr/bin/env python3
"""
阶段 2 · Mini 项目 3：Word2Vec Skip-gram + 负采样

核心实验：
  python main.py                              # 默认参数（Text8 英文）
  python main.py --embed_dim 200 --epochs 10  # 200 维向量，10 轮

运行示例：
  python main.py                              # 默认：Text8, 50k 词表, 200 维, 10 epochs
  python main.py --embed_dim 300              # 300 维词向量
  python main.py --window_size 5              # 上下文窗口 ±5
  python main.py --num_negative 10            # 10 个负样本
  python main.py --epochs 15                  # 15 轮训练
"""

import argparse
import torch
from data import (
    maybe_download_text8, build_vocab, make_subsample_table,
    apply_subsample, make_noise_dist, get_dataloader,
)
from word2vec import SkipGramNeg, find_analogy, find_similar
from train import train
from visualize import plot_tsne, plot_analogy


# ============================================================
# 标准类比测试集（Google Analogy Test Set 子集）
# ============================================================

ENGLISH_ANALOGIES = [
    # 首都-国家
    ("athens", "greece", "berlin", "germany"),
    ("paris", "france", "rome", "italy"),
    ("beijing", "china", "tokyo", "japan"),
    ("london", "england", "madrid", "spain"),
    ("moscow", "russia", "ottawa", "canada"),
    # 家庭关系
    ("king", "man", "queen", "woman"),
    ("man", "woman", "boy", "girl"),
    ("brother", "sister", "uncle", "aunt"),
    ("father", "mother", "son", "daughter"),
    # 动词时态
    ("go", "went", "see", "saw"),
    ("take", "took", "give", "gave"),
    ("say", "said", "know", "knew"),
    # 比较级/最高级
    ("good", "better", "bad", "worse"),
    ("big", "bigger", "small", "smaller"),
    # 国家-形容词
    ("china", "chinese", "france", "french"),
    ("germany", "german", "england", "english"),
]

# 用于展示相似词的测试词
TEST_WORDS = ["king", "queen", "london", "computer", "music", "war", "love",
              "one", "good", "great"]


def main():
    parser = argparse.ArgumentParser(description="Word2Vec Skip-gram + 负采样")
    parser.add_argument("--embed_dim", type=int, default=200,
                        help="词向量维度（默认 200）")
    parser.add_argument("--window_size", type=int, default=5,
                        help="上下文窗口半径（默认 5）")
    parser.add_argument("--num_negative", type=int, default=5,
                        help="负采样数（默认 5）")
    parser.add_argument("--epochs", type=int, default=10,
                        help="训练轮数（默认 10）")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch 大小（默认 512）")
    parser.add_argument("--min_count", type=int, default=5,
                        help="最低词频（默认 5）")
    parser.add_argument("--max_vocab", type=int, default=50000,
                        help="最大词表大小（默认 50000）")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="学习率（默认 0.001）")
    parser.add_argument("--no_plot", action="store_true",
                        help="跳过可视化")
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
    print("[1/4] 加载 Text8 数据集...")
    tokens = maybe_download_text8()
    print(f"  总词数: {len(tokens):,}")

    # --- 2. 构建词表 ---
    print("[2/4] 构建词表...")
    word_to_idx, idx_to_word, word_counts = build_vocab(
        tokens, min_count=args.min_count, max_vocab=args.max_vocab,
    )

    # 二次采样 + 负采样分布
    drop_probs = make_subsample_table(word_counts)
    noise_dist = make_noise_dist(word_counts)

    print("  应用二次采样...")
    token_ids = apply_subsample(tokens, word_to_idx, drop_probs)
    drop_rate = 1 - len(token_ids) / len(tokens)
    print(f"  丢弃率: {drop_rate:.1%}, 剩余 tokens: {len(token_ids):,}")

    # 构建 DataLoader
    train_loader = get_dataloader(
        token_ids,
        window_size=args.window_size,
        num_negative=args.num_negative,
        noise_dist=noise_dist,
        batch_size=args.batch_size,
    )
    print(f"  训练 batch 数: {len(train_loader):,}")
    print(f"  负采样分布前 5 词: {[idx_to_word[i] for i in noise_dist.argsort()[-5:][::-1]]}")

    # --- 3. 构建模型 ---
    print("[3/4] 构建 Skip-gram 模型...")
    model = SkipGramNeg(
        vocab_size=len(word_to_idx),
        embed_dim=args.embed_dim,
    )
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    # --- 4. 训练 ---
    print("[4/4] 开始训练...")
    trained_model = train(
        model, train_loader, word_to_idx, idx_to_word,
        test_words=TEST_WORDS,
        analogies=ENGLISH_ANALOGIES,
        epochs=args.epochs, lr=args.lr, device=device,
    )

    # --- 可视化 ---
    if not args.no_plot:
        print("\n生成可视化...")
        vectors = trained_model.get_vectors()
        plot_tsne(vectors, idx_to_word, word_to_idx,
                  words_to_plot=TEST_WORDS + [w for a in ENGLISH_ANALOGIES[:5] for w in a])
        plot_analogy(vectors, word_to_idx,
                     ("king", "man", "woman", "queen"))


if __name__ == "__main__":
    main()
