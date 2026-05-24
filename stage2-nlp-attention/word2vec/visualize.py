"""
Word2Vec 词向量可视化

生成：
  1. t-SNE 降维散点图（选取若干个高频词标注）
  2. 词类比二维示意图
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch


def plot_tsne(
    vectors: torch.Tensor,
    idx_to_word: dict,
    word_to_idx: dict,
    words_to_plot: list = None,
    top_n: int = 200,
    save_path: str = "word2vec_tsne.png",
):
    """
    t-SNE 可视化词向量

    参数：
      words_to_plot: 指定要标注的词列表（如 ["king", "queen", "man", "woman"]）
      top_n:         选取最高频的前 N 个词做 t-SNE（减少计算量）
    """
    vectors_np = vectors.numpy()

    # 选取 top_n 高频词
    vocab_size = min(top_n, len(vectors_np))
    selected_vectors = vectors_np[:vocab_size]
    selected_words = [idx_to_word[i] for i in range(vocab_size)]

    # t-SNE 降维到 2D
    print(f"  正在进行 t-SNE 降维 ({vocab_size} 个词)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=500)
    vectors_2d = tsne.fit_transform(selected_vectors)

    # 绘图
    plt.figure(figsize=(14, 10))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], s=5, alpha=0.3, c="steelblue")

    # 标注指定词
    if words_to_plot is None:
        words_to_plot = ["king", "queen", "man", "woman", "city", "country",
                         "one", "two", "three", "good", "great"]

    for word in words_to_plot:
        if word in word_to_idx:
            idx = word_to_idx[word]
            if idx < vocab_size:
                x, y = vectors_2d[idx]
                plt.annotate(word, (x, y), fontsize=11, fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow",
                                      alpha=0.7))

    plt.title("Word2Vec t-SNE Visualization", fontsize=14)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  图片已保存: {save_path}")


def plot_analogy(
    vectors: torch.Tensor,
    word_to_idx: dict,
    analogy: tuple,  # (a, b, c, expected)
    save_path: str = "word2vec_analogy.png",
):
    """
    2D 可视化一个词类比（用 PCA 降到 2D 后绘制箭头）

    analogy 格式: ("king", "man", "woman", "queen")
    展示 vec(b)→vec(a) 和 vec(c)→vec(?) 的关系
    """
    from sklearn.decomposition import PCA

    a, b, c, expected = analogy

    # 收集相关向量
    words = [a, b, c]
    if expected in word_to_idx:
        words.append(expected)
    indices = [word_to_idx[w] for w in words if w in word_to_idx]
    vecs = vectors[indices].numpy()

    # PCA 降维
    pca = PCA(n_components=2, random_state=42)
    vecs_2d = pca.fit_transform(vecs)

    plt.figure(figsize=(8, 8))

    # 绘制点
    for i, w in enumerate(words):
        if w in word_to_idx:
            plt.scatter(vecs_2d[i, 0], vecs_2d[i, 1], s=100)
            plt.annotate(w, (vecs_2d[i, 0], vecs_2d[i, 1]),
                        fontsize=12, fontweight="bold",
                        xytext=(5, 5), textcoords="offset points")

    # 绘制类比箭头
    # vec(a) - vec(b) ≈ vec(expected) - vec(c)
    # 即 vec(b)→vec(a) 应该和 vec(c)→vec(result) 平行
    idx_map = {w: i for i, w in enumerate(words) if w in word_to_idx}
    if all(w in idx_map for w in [a, b]):
        plt.arrow(vecs_2d[idx_map[b], 0], vecs_2d[idx_map[b], 1],
                  vecs_2d[idx_map[a], 0] - vecs_2d[idx_map[b], 0],
                  vecs_2d[idx_map[a], 1] - vecs_2d[idx_map[b], 1],
                  head_width=0.3, head_length=0.5, fc='red', ec='red',
                  width=0.05)

    if all(w in idx_map for w in [c, expected]):
        plt.arrow(vecs_2d[idx_map[c], 0], vecs_2d[idx_map[c], 1],
                  vecs_2d[idx_map[expected], 0] - vecs_2d[idx_map[c], 0],
                  vecs_2d[idx_map[expected], 1] - vecs_2d[idx_map[c], 1],
                  head_width=0.3, head_length=0.5, fc='green', ec='green',
                  width=0.05)

    plt.title(f"Word Analogy: {a} - {b} + {c} ≈ {expected}", fontsize=14)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  图片已保存: {save_path}")
