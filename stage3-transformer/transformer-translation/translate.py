"""
Transformer 翻译 + Beam Search + BLEU 评估

核心概念：
  - 自回归解码：逐 token 生成，每次用已生成的 token + encoder 输出预测下一个
  - Beam Search: 维护 k 个最优候选序列，每步扩展 × V 个可能 → 取 top-k → 剪枝
  - 和 RNN Decoder 推断的区别：Transformer 可以并行处理已生成的整个序列
    （因为 Self-Attention 没有时序依赖），但每次解码新 token 仍需重跑整个 Decoder
"""

import torch
import torch.nn.functional as F
import math
from data import SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, tokenize
from attention import make_padding_mask, make_causal_mask


@torch.no_grad()
def translate_sentence(
    model,
    src_sentence: str,
    src_w2i: dict,
    tgt_i2w: dict,
    device: torch.device,
    max_len: int = 50,
    beam_size: int = 1,
) -> tuple:
    """
    翻译一个句子

    返回: (translated_tokens, attention_weights)
    """
    model.eval()

    # 编码源语言
    src_indices = [src_w2i.get(w, src_w2i[UNK_TOKEN])
                   for w in tokenize(src_sentence)]
    src_indices.append(src_w2i[EOS_TOKEN])
    src = torch.tensor([src_indices], dtype=torch.long, device=device)

    sos_idx = src_w2i[SOS_TOKEN]
    eos_idx = src_w2i[EOS_TOKEN]

    if beam_size == 1:
        return _greedy_decode(model, src, sos_idx, eos_idx, tgt_i2w,
                              device, max_len)
    else:
        return _beam_search_decode(model, src, sos_idx, eos_idx, tgt_i2w,
                                   device, max_len, beam_size)


def _greedy_decode(model, src, sos_idx, eos_idx, tgt_i2w, device, max_len):
    """贪心解码"""
    src_mask = make_padding_mask(src, model.pad_idx)  # (1, 1, 1, S)

    # Encoder 只需前向一次（和 RNN 不同，不需要每步重跑 encoder）
    enc_output = model.encoder(src, src_mask)

    # 初始 decoder 输入：<SOS>
    generated = [sos_idx]

    for _ in range(max_len):
        tgt = torch.tensor([generated], dtype=torch.long, device=device)
        tgt_mask = make_causal_mask(tgt.size(1), device)

        logits = model.decoder(tgt, enc_output, tgt_mask, src_mask)
        # logits: (1, T, V)
        next_token = logits[0, -1, :].argmax().item()
        generated.append(next_token)

        if next_token == eos_idx:
            break

    tokens = [tgt_i2w.get(idx, UNK_TOKEN) for idx in generated[1:]
              if idx != eos_idx]
    return tokens, None


def _beam_search_decode(model, src, sos_idx, eos_idx, tgt_i2w,
                        device, max_len, beam_size):
    """Beam Search 解码"""
    B = src.size(0)  # 假设 = 1
    src_mask = make_padding_mask(src, model.pad_idx)
    enc_output = model.encoder(src, src_mask)

    # 扩展 encoder 输出以匹配 beam_size 个候选
    enc_output = enc_output.expand(beam_size, -1, -1)
    src_mask = src_mask.expand(beam_size, -1, -1, -1)

    # 初始候选：(序列, log_prob, 是否结束)
    beams = [([sos_idx], 0.0, False)]

    for _ in range(max_len):
        new_beams = []

        # 收集当前所有活跃候选
        active_seqs = []
        for seq, score, done in beams:
            if done:
                new_beams.append((seq, score, done))
            else:
                active_seqs.append((seq, score))

        if not active_seqs:
            break

        # 批量前向：所有活跃序列一起跑 decoder
        for seq, score in active_seqs:
            tgt = torch.tensor([seq], dtype=torch.long, device=device)
            tgt_mask = make_causal_mask(tgt.size(1), device)

            logits = model.decoder(tgt, enc_output[:1], tgt_mask, src_mask[:1])
            last_logits = logits[0, -1, :]  # (V,)

            # 归一化 logits → log probabilities
            log_probs = F.log_softmax(last_logits, dim=-1)

            # 取 top-beam_size 个 next token
            topk = log_probs.topk(beam_size)
            for k in range(beam_size):
                token = topk.indices[k].item()
                new_score = score + topk.values[k].item()
                new_seq = seq + [token]
                is_done = (token == eos_idx)
                new_beams.append((new_seq, new_score, is_done))

        # 剪枝：按分数排序，只保留 beam_size 个
        new_beams.sort(key=lambda x: x[1] / max(len(x[0]), 1), reverse=True)
        beams = new_beams[:beam_size]

        if all(d for _, _, d in beams):
            break

    # 取最优序列
    best = beams[0][0]
    tokens = [tgt_i2w.get(idx, UNK_TOKEN) for idx in best[1:]
              if idx != eos_idx]
    return tokens, None


# ============================================================
# BLEU（和阶段 2 相同）
# ============================================================

from collections import Counter

def compute_bleu(reference: list, candidate: list, max_n: int = 4) -> float:
    cand_len = len(candidate)
    ref_len = len(reference)
    if cand_len == 0:
        return 0.0
    bp = min(1.0, math.exp(1.0 - ref_len / cand_len))

    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(tuple(reference[i:i+n])
                            for i in range(ref_len - n + 1))
        cand_ngrams = Counter(tuple(candidate[i:i+n])
                             for i in range(cand_len - n + 1))
        overlap = sum(min(c, ref_ngrams.get(ng, 0))
                     for ng, c in cand_ngrams.items())
        total = max(cand_len - n + 1, 1)
        precisions.append(overlap / total if total > 0 else 0.0)

    log_avg = sum(math.log(max(p, 1e-10)) for p in precisions) / max_n
    return bp * math.exp(log_avg)
