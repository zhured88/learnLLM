"""
Decoder：带注意力的 GRU 解码器

核心概念：
  - 注意力解码：每一步用当前状态去"查询"encoder 所有时间步，
    找到最相关的部分 → 加权求和 → 和当前输入拼接 → 喂入 RNN
  - Teacher Forcing：训练时用标准答案作为下一步输入，而非上一步的预测
  - 推断时：用上一步预测的词作为下一步输入（自回归）

维度流转（以 B=64, S_src=15, S_tgt=18, E=256, H_enc=1024, H_dec=512 为例）：

  ① 输入 y_{t-1}:   (64,)         ← 上一步的 token 索引
  ② embedding:       (64, 256)     ← 词向量
  ③ 拼接 context:    (64, 1280)    ← [embedding; context]
  ④ GRU:             (64, 512)     ← 更新隐藏状态
  ⑤ 生成 logits:     (64, V_tgt)   ← 映射回词表
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from attention import BahdanauAttention, LuongAttention


class Decoder(nn.Module):
    """
    带注意力的 GRU Decoder

    结构：
      Embedding → Attention → GRU → Linear → logits
               ↑ context vector 来自 encoder outputs
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        enc_hidden_dim: int = 1024,   # encoder 输出维度（2 * hidden_dim）
        dec_hidden_dim: int = 512,     # decoder 隐藏维度
        num_layers: int = 2,
        attn_type: str = "bahdanau",   # "bahdanau" | "luong_dot" | "luong_general"
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dec_hidden_dim = dec_hidden_dim
        self.num_layers = num_layers
        self.attn_type = attn_type
        self.pad_idx = pad_idx

        # Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)

        # 注意力模块
        if attn_type.startswith("luong"):
            self.attention = LuongAttention(
                enc_hidden_dim, dec_hidden_dim,
                method=attn_type.replace("luong_", ""),
            )
        else:
            self.attention = BahdanauAttention(enc_hidden_dim, dec_hidden_dim)

        # GRU：输入是 [embedding; context]
        # embedding (E) + context (enc_hidden_dim) → GRU input
        self.gru_input_dim = embed_dim + enc_hidden_dim
        self.gru = nn.GRU(
            self.gru_input_dim, dec_hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # 输出映射
        self.fc = nn.Linear(dec_hidden_dim + embed_dim + enc_hidden_dim, vocab_size)
        # ↑ Luong 论文提出"输入馈送"（input feeding）：
        #   输入 [dec_hidden; embed; context] 三个信号

    def _init_hidden(self, encoder_hidden: torch.Tensor) -> torch.Tensor:
        """
        将 encoder 的最终隐藏状态转换为 decoder 的初始隐藏状态

        encoder_hidden: (2 * num_layers, B, hidden_dim)  双向
        → 需要映射为 (num_layers, B, dec_hidden_dim)     单向

        方法：取正向和反向的最后状态 → 拼接 → 线性映射
        """
        # encoder_hidden 是双向的，层排列为 [fwd0, bwd0, fwd1, bwd1, ...]
        # 取每层的正向和反向 → 拼接 → 映射
        batch_size = encoder_hidden.size(1)

        # reshape: (2, num_layers, B, H) → 取正向和反向的最后层
        enc_h = encoder_hidden.view(2, self.num_layers, batch_size,
                                     encoder_hidden.size(-1))
        # 取最后一层的正向+反向
        fwd_last = enc_h[-2, -1]  # (B, H)
        bwd_last = enc_h[-1, -1]  # (B, H)
        dec_init = torch.cat([fwd_last, bwd_last], dim=1)  # (B, 2H)

        # 线性映射到 decoder hidden dim
        if not hasattr(self, "enc_to_dec"):
            self.enc_to_dec = nn.Linear(
                encoder_hidden.size(-1), self.dec_hidden_dim,
            ).to(encoder_hidden.device)

        dec_h0 = torch.tanh(self.enc_to_dec(dec_init))  # (B, H_dec)

        # 扩展到 num_layers 层
        return dec_h0.unsqueeze(0).repeat(self.num_layers, 1, 1)  # (L, B, H_dec)

    def forward(
        self,
        tgt: torch.Tensor,              # (B, tgt_len) decoder 输入
        enc_outputs: torch.Tensor,      # (B, src_len, 2*H_enc)
        src_mask: torch.Tensor,         # (B, src_len)
        enc_hidden: torch.Tensor,       # (2*L, B, H_enc)
        teacher_forcing_ratio: float = 0.5,
    ) -> tuple:
        """
        训练模式前向传播

        返回：
          logits: (B, tgt_len, vocab_size)
          attn_weights: (B, tgt_len, src_len) 注意力权重（仅 teacher forcing 时）
        """
        B, tgt_len = tgt.shape
        device = tgt.device

        # 初始化 decoder 隐藏状态
        hidden = self._init_hidden(enc_hidden)  # (L, B, H_dec)

        # 存储每步输出
        outputs = []
        all_attn_weights = []

        # decoder 第一个输入是 <SOS> token
        input_token = tgt[:, 0]  # (B,)

        for t in range(1, tgt_len):
            # --- ① 当前输入词的 embedding ---
            embed = self.embedding(input_token)        # (B, E)
            embed = self.dropout(embed).unsqueeze(1)    # (B, 1, E)

            # --- ② 注意力：决定关注 encoder 的哪些位置 ---
            # Bahdanau: 用 dec_hidden[-1]（上一步的隐藏状态）
            # Luong:    用 dec_hidden[-1]（也是上一步，但语义上对应 "先用 s_{t-1} 算 context"）
            context, attn_w = self.attention(
                enc_outputs, hidden[-1], src_mask,
            )  # context: (B, H_enc), attn_w: (B, S)
            context = context.unsqueeze(1)  # (B, 1, H_enc)

            # --- ③ GRU 前向 ---
            # 输入 = [embedding; context]  沿特征维度拼接
            gru_input = torch.cat([embed, context], dim=2)  # (B, 1, E+H_enc)
            gru_out, hidden = self.gru(gru_input, hidden)     # (B, 1, H_dec)

            # --- ④ 计算输出 logits ---
            # 输入馈送：[gru_out; embed; context]
            fc_input = torch.cat([
                gru_out.squeeze(1),       # (B, H_dec)
                embed.squeeze(1),         # (B, E)
                context.squeeze(1),       # (B, H_enc)
            ], dim=1)  # (B, H_dec+E+H_enc)

            logits = self.fc(fc_input)     # (B, V)
            outputs.append(logits.unsqueeze(1))
            all_attn_weights.append(attn_w)

            # --- ⑤ 决定下一步的输入（Teacher Forcing 的关键）---
            # teacher_forcing_ratio = 0.5 → 50% 概率用正确答案，50% 用模型预测
            if random.random() < teacher_forcing_ratio:
                input_token = tgt[:, t]  # 正确答案
            else:
                input_token = logits.argmax(dim=1)  # 模型预测

        outputs = torch.cat(outputs, dim=1)               # (B, tgt_len-1, V)
        all_attn_weights = torch.stack(all_attn_weights, dim=1)  # (B, tgt_len-1, S)

        return outputs, all_attn_weights

    def translate(
        self,
        enc_outputs: torch.Tensor,
        src_mask: torch.Tensor,
        enc_hidden: torch.Tensor,
        sos_idx: int,
        eos_idx: int,
        max_len: int = 50,
        beam_size: int = 1,
    ) -> list:
        """
        推断模式：逐词翻译（自回归生成 + 可选 beam search）

        参数：
          beam_size: 1 = 贪心解码, >1 = beam search
        """
        if beam_size == 1:
            return self._greedy_decode(
                enc_outputs, src_mask, enc_hidden, sos_idx, eos_idx, max_len,
            )
        else:
            return self._beam_search(
                enc_outputs, src_mask, enc_hidden, sos_idx, eos_idx,
                max_len, beam_size,
            )

    def _greedy_decode(
        self,
        enc_outputs: torch.Tensor,
        src_mask: torch.Tensor,
        enc_hidden: torch.Tensor,
        sos_idx: int,
        eos_idx: int,
        max_len: int,
    ) -> tuple:
        """
        贪心解码：每一步选概率最大的词

        返回：(predicted_indices, attention_weights)
        """
        B = enc_outputs.size(0)
        device = enc_outputs.device

        hidden = self._init_hidden(enc_hidden)
        input_token = torch.full((B,), sos_idx, dtype=torch.long, device=device)

        result = []
        all_attn_weights = []
        done = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len):
            embed = self.embedding(input_token).unsqueeze(1)  # (B, 1, E)
            context, attn_w = self.attention(enc_outputs, hidden[-1], src_mask)
            context = context.unsqueeze(1)

            gru_input = torch.cat([embed, context], dim=2)
            gru_out, hidden = self.gru(gru_input, hidden)

            fc_input = torch.cat([
                gru_out.squeeze(1), embed.squeeze(1), context.squeeze(1),
            ], dim=1)
            logits = self.fc(fc_input)

            input_token = logits.argmax(dim=1)  # (B,)
            input_token[done] = eos_idx        # 已结束的句子填 <EOS>

            done = done | (input_token == eos_idx)
            result.append(input_token.unsqueeze(1))
            all_attn_weights.append(attn_w)

            if done.all():
                break

        result = torch.cat(result, dim=1)  # (B, T)
        all_attn_weights = torch.stack(all_attn_weights, dim=1)  # (B, T, S)
        return result, all_attn_weights

    def _beam_search(
        self,
        enc_outputs: torch.Tensor,
        src_mask: torch.Tensor,
        enc_hidden: torch.Tensor,
        sos_idx: int,
        eos_idx: int,
        max_len: int,
        beam_size: int,
    ) -> tuple:
        """
        Beam Search：维护 beam_size 个最优候选，每步扩展并剪枝

        简化实现：batch_size=1 时生效（单句翻译）
        """
        assert enc_outputs.size(0) == 1, "Beam search 目前只支持 batch_size=1"

        device = enc_outputs.device

        # 初始候选：(序列, log_prob, hidden)
        hidden_init = self._init_hidden(enc_hidden)  # (L, 1, H)
        beams = [([sos_idx], 0.0, hidden_init)]

        finished = []

        for _ in range(max_len):
            new_beams = []

            for seq, score, hidden in beams:
                if seq[-1] == eos_idx:
                    finished.append((seq, score))
                    continue

                input_token = torch.tensor([[seq[-1]]], dtype=torch.long,
                                           device=device)
                embed = self.embedding(input_token)  # (1, 1, E)

                # 扩展 enc_outputs 以匹配
                context, _ = self.attention(enc_outputs, hidden[-1], src_mask)
                context = context.unsqueeze(1)

                gru_input = torch.cat([embed, context], dim=2)
                gru_out, new_hidden = self.gru(gru_input, hidden)

                fc_input = torch.cat([
                    gru_out.squeeze(1), embed.squeeze(1), context.squeeze(1),
                ], dim=1)
                logits = self.fc(fc_input)  # (1, V)

                # top-k logits
                log_probs = F.log_softmax(logits, dim=1).squeeze(0)
                top_k = log_probs.topk(beam_size)

                for k in range(beam_size):
                    token_idx = top_k.indices[k].item()
                    new_score = score + top_k.values[k].item()
                    new_seq = seq + [token_idx]
                    new_beams.append((new_seq, new_score, new_hidden))

            # 剪枝：只保留分数最高的 beam_size 个候选
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_size]

            if not beams:
                break

        finished.extend(beams)
        finished.sort(key=lambda x: x[1] / max(len(x[0]), 1), reverse=True)

        best_seq = finished[0][0]
        indices = torch.tensor([best_seq], dtype=torch.long, device=device)
        return indices, None  # 简化：不返回注意力权重
