"""
翻译 + BLEU 评估 + 注意力可视化

核心概念：
  - 贪心解码：每步选 argmax token，快但可能不是全局最优
  - Beam Search：每步保留 top-k 候选，再扩展，取整体最高分序列
  - BLEU：n-gram 精度 + 短句惩罚（brevity penalty）
"""

import torch
import torch.nn.functional as F
from collections import Counter
import math
import matplotlib.pyplot as plt
import numpy as np


def translate_sentence(
    model,
    src_sentence: str,
    src_word_to_idx: dict,
    tgt_idx_to_word: dict,
    sos_idx: int,
    eos_idx: int,
    unk_idx: int,
    device: torch.device,
    max_len: int = 50,
    beam_size: int = 1,
) -> tuple:
    """
    翻译一个句子

    返回：(translated_tokens, attention_matrix)
    """
    from data import tokenize, EOS_TOKEN

    model.eval()

    # 源语言分词 + 转索引
    src_indices = [src_word_to_idx.get(w, unk_idx)
                   for w in tokenize(src_sentence)]
    src_indices.append(src_word_to_idx[EOS_TOKEN])

    src_tensor = torch.tensor([src_indices], dtype=torch.long, device=device)
    src_len = torch.tensor([len(src_indices)], dtype=torch.long)

    with torch.no_grad():
        enc_outputs, enc_hidden = model.encoder(src_tensor, src_len)
        src_mask = (src_tensor != model.encoder.pad_idx)

        indices, attn_w = model.decoder.translate(
            enc_outputs, src_mask, enc_hidden,
            sos_idx, eos_idx, max_len, beam_size=beam_size,
        )

    # 解码：索引 → 词 → 去掉 <EOS> 之后的所有
    if attn_w is not None:
        attn_w = attn_w.squeeze(0).cpu().numpy()  # (T, S)
    indices = indices.squeeze(0).tolist()

    tokens = []
    for idx in indices:
        if idx == eos_idx:
            break
        tokens.append(tgt_idx_to_word.get(idx, "<UNK>"))

    return tokens, attn_w


# ============================================================
# BLEU 评估
# ============================================================

def ngram_counts(tokens: list, n: int) -> Counter:
    """统计 n-gram 频率"""
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def compute_bleu(reference: list, candidate: list, max_n: int = 4) -> float:
    """
    计算 BLEU 分数

    BLEU = BP × exp(Σ w_n log p_n)
    其中：
      p_n = (candidate 和 reference 中 n-gram 重叠数) / (candidate 的 n-gram 总数)
      BP = min(1, exp(1 - ref_len / cand_len))  ← 短句惩罚

    参数：
      reference: 参考译文（词列表）
      candidate: 模型译文（词列表）
    """
    cand_len = len(candidate)
    ref_len = len(reference)

    # 短句惩罚：候选译文太短时惩罚
    if cand_len == 0:
        return 0.0
    bp = min(1.0, math.exp(1.0 - ref_len / cand_len))

    # n-gram 精度
    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = ngram_counts(reference, n)
        cand_ngrams = ngram_counts(candidate, n)

        overlap = 0
        for ng, count in cand_ngrams.items():
            overlap += min(count, ref_ngrams.get(ng, 0))

        total = max(len(candidate) - n + 1, 1)
        precisions.append(overlap / total if total > 0 else 0.0)

    # 几何平均（避免 log(0)）
    log_avg = sum(math.log(max(p, 1e-10)) for p in precisions) / max_n

    return bp * math.exp(log_avg)


def evaluate_bleu(
    model,
    test_pairs: list,
    src_word_to_idx: dict,
    tgt_idx_to_word: dict,
    sos_idx: int,
    eos_idx: int,
    unk_idx: int,
    device: torch.device,
    num_samples: int = 100,
    beam_size: int = 1,
) -> float:
    """
    在测试集上评估 BLEU 分数

    测试 pairs 格式：[(eng, fra), ...]
    """
    from data import tokenize

    scores = []
    samples = test_pairs[:num_samples]

    for i, (src_text, ref_text) in enumerate(samples):
        translated_tokens, _ = translate_sentence(
            model, src_text, src_word_to_idx, tgt_idx_to_word,
            sos_idx, eos_idx, unk_idx, device,
            beam_size=beam_size,
        )

        ref_tokens = tokenize(ref_text)
        score = compute_bleu(ref_tokens, translated_tokens)
        scores.append(score)

        if i < 3:
            print(f"\n  Source:     {src_text}")
            print(f"  Reference:  {ref_text}")
            print(f"  Translated: {' '.join(translated_tokens)}")
            print(f"  BLEU:       {score:.4f}")

    avg_bleu = sum(scores) / len(scores)
    print(f"\n  平均 BLEU ({len(scores)} 句): {avg_bleu:.4f}")
    return avg_bleu


# ============================================================
# 注意力可视化
# ============================================================

def plot_attention(
    src_words: list,
    tgt_words: list,
    attn_matrix: np.ndarray,
    save_path: str = "attention_heatmap.png",
):
    """
    绘制注意力权重热力图

    参数：
      src_words: 源语言词列表
      tgt_words: 目标语言词列表
      attn_matrix: (tgt_len, src_len) 注意力权重
    """
    plt.figure(figsize=(max(len(src_words) * 0.8, 8),
                        max(len(tgt_words) * 0.8, 6)))

    plt.imshow(attn_matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    plt.xticks(range(len(src_words)), src_words, rotation=45, ha="right",
               fontsize=9)
    plt.yticks(range(len(tgt_words)), tgt_words, fontsize=9)

    plt.xlabel("Source (English)", fontsize=12)
    plt.ylabel("Target (French)", fontsize=12)
    plt.title("Attention Alignment", fontsize=14)
    plt.colorbar(label="Attention Weight")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  注意力热力图已保存: {save_path}")
