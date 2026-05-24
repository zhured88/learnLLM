"""
Word2Vec Skip-gram 模型 + 负采样损失

核心概念：
  - Skip-gram：给定中心词 w_c，预测其上下文词 w_o
  - 负采样 (NEG)：不用 softmax 遍历整个词表（V 太大），
    而是把问题转化为二分类——"w_c 和 w_o 是不是真实上下文对？"
  - 两个 Embedding 矩阵：
      u_embed (输入向量): 作为中心词时的表示
      v_embed (输出向量): 作为上下文词时的表示
    训练后取 u_embed 作为最终词向量（或取两者的平均/拼接）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGramNeg(nn.Module):
    """
    Skip-gram + 负采样

    和传统 softmax 的区别：
      softmax: loss = -log( exp(u_c·v_o) / Σ_w exp(u_c·v_w) )  ← 分母要对 V 个词求和
      NEG:     loss = -log σ(u_c·v_o) - Σ_{k~Pn} log σ(-u_c·v_k)  ← 只算 (1+K) 个点积

    其中 σ=sigmoid, Pn=噪声分布, K=负样本数
    """

    def __init__(self, vocab_size: int, embed_dim: int = 200):
        """
        参数：
          vocab_size: 词表大小（Text8 用 50000, 中文约 30000）
          embed_dim:  词向量维度（200-300 是经典选择）
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # --- 中心词 Embedding（输入向量）---
        # 形状: (vocab_size, embed_dim)
        # 每一行是一个词的向量表示。训练完成后，这一般就是我们要的词向量
        self.u_embed = nn.Embedding(vocab_size, embed_dim)
        # Xavier 初始化：让向量初始值的方差合适，训练更稳定
        nn.init.xavier_uniform_(self.u_embed.weight)

        # --- 上下文词 Embedding（输出向量）---
        # Skip-gram 有两个 embedding 矩阵是论文的原始设计
        # 数学上，每个词有两个角色：作为中心词（输入）和作为上下文（输出）
        # 实践中通常只用 u_embed 或取 (u+v)/2
        self.v_embed = nn.Embedding(vocab_size, embed_dim)
        nn.init.xavier_uniform_(self.v_embed.weight)

    def forward(self, center: torch.Tensor, context: torch.Tensor,
                neg_samples: torch.Tensor) -> torch.Tensor:
        """
        计算负采样损失

        参数：
          center:      (B,)    中心词索引, 如 [42, 7, 100, ...]
          context:     (B,)    真实上下文词索引（正样本）
          neg_samples: (B, K)  噪声词索引（负样本），K=num_negative

        返回：
          loss: 标量，平均负采样损失

        维度推导（以 B=512, D=200, K=5 为例）：
          u_c:         (512, 200)    中心词的输入向量
          v_o:         (512, 200)    上下文词的输出向量
          pos_score:   (512,)        正样本得分 = u_c · v_o
          v_n:         (512, 5, 200) 噪声词的输出向量
          neg_score:   (512, 5)      负样本得分 = u_c · v_n
        """
        # --- ① 正样本得分：中心词 · 上下文词 ---
        u_c = self.u_embed(center)           # (B,) → (B, D)
        v_o = self.v_embed(context)          # (B,) → (B, D)
        pos_score = (u_c * v_o).sum(dim=1)   # (B, D) → (B,)  逐元素乘 → 求和 = 点积
        pos_loss = F.logsigmoid(pos_score)   # (B,)  log σ(u·v)

        # --- ② 负样本得分：中心词 · 噪声词 ---
        v_n = self.v_embed(neg_samples)      # (B, K) → (B, K, D)
        # 批量点积：(B, D) × (B, K, D) → (B, K)
        # Einstein summation: bd,bkd→bk  = 对每个 batch 和每个负样本计算点积
        neg_score = torch.einsum("bd,bkd->bk", u_c, v_n)  # (B, K)
        neg_loss = F.logsigmoid(-neg_score)  # log σ(-u·v_n)，注意负号

        # --- ③ 总损失 ---
        # NEG loss = -Σ log σ(u·v_pos) - Σ log σ(-u·v_neg)
        # 等价于 -mean(pos_loss + neg_loss.sum(dim=1))
        loss = -(pos_loss + neg_loss.sum(dim=1)).mean()
        return loss

    def get_vectors(self) -> torch.Tensor:
        """
        获取训练好的词向量

        返回 u_embed.weight（中心词向量），这是最常见的做法。
        也可以返回 (u_embed + v_embed) / 2，效果通常略好。
        """
        return self.u_embed.weight.detach().cpu()


# ============================================================
# 类比推理工具
# ============================================================

def find_analogy(
    word_a: str,
    word_b: str,
    word_c: str,
    word_to_idx: dict,
    idx_to_word: dict,
    vectors: torch.Tensor,
    top_k: int = 5,
) -> list:
    """
    词类比：a - b + c = ?

    经典例子："king" - "man" + "woman" ≈ "queen"
    即 vec(king) - vec(man) + vec(woman) 最接近 vec(queen)

    中文例子："北京" - "中国" + "法国" ≈ "巴黎"
    """
    if any(w not in word_to_idx for w in [word_a, word_b, word_c]):
        return [("N/A", 0.0)]

    idx_a = word_to_idx[word_a]
    idx_b = word_to_idx[word_b]
    idx_c = word_to_idx[word_c]

    # 计算类比向量
    query = vectors[idx_a] - vectors[idx_b] + vectors[idx_c]  # (D,)

    # 余弦相似度 = a·b / (|a| × |b|)
    # 这里用 L2 归一化后点积 = 余弦相似度
    query_norm = query / (query.norm() + 1e-8)
    vectors_norm = vectors / (vectors.norm(dim=1, keepdim=True) + 1e-8)
    similarities = vectors_norm @ query_norm  # (V,) 每行和 query 的余弦相似度

    # 排序取 top_k（排除输入的三个词本身）
    exclude = {idx_a, idx_b, idx_c}
    top_indices = similarities.argsort(descending=True).tolist()
    results = []
    for idx in top_indices:
        if idx not in exclude:
            results.append((idx_to_word[idx], similarities[idx].item()))
        if len(results) >= top_k:
            break

    return results


def find_similar(
    word: str,
    word_to_idx: dict,
    idx_to_word: dict,
    vectors: torch.Tensor,
    top_k: int = 10,
) -> list:
    """找最相似的 top_k 个词（余弦相似度）"""
    if word not in word_to_idx:
        return [("N/A", 0.0)]

    idx = word_to_idx[word]
    query = vectors[idx]
    query_norm = query / (query.norm() + 1e-8)
    vectors_norm = vectors / (vectors.norm(dim=1, keepdim=True) + 1e-8)
    similarities = vectors_norm @ query_norm

    top_indices = similarities.argsort(descending=True).tolist()
    results = []
    for i in top_indices:
        if i != idx:
            results.append((idx_to_word[i], similarities[i].item()))
        if len(results) >= top_k:
            break

    return results
