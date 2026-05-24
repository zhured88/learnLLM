"""
Seq2Seq + Attention 训练循环

核心要点：
  - Teacher Forcing：训练时以一定概率用标准答案作为下一步输入
  - Masking：计算 loss 时忽略 <PAD> 位置
  - Gradient Clipping：防止 RNN 梯度爆炸
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import Encoder
from decoder import Decoder
from data import PAD_TOKEN


class Seq2Seq(nn.Module):
    """Encoder-Decoder 总包装"""

    def __init__(self, encoder: Encoder, decoder: Decoder, tgt_pad_idx: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tgt_pad_idx = tgt_pad_idx

    def forward(self, src, tgt, src_lengths, teacher_forcing_ratio=0.5):
        """训练模式一次前向"""
        enc_outputs, enc_hidden = self.encoder(src, src_lengths)

        # src mask: 标记哪些位置不是 <PAD>
        src_mask = (src != self.encoder.pad_idx)  # (B, S)
        # 扩展 mask 到双向隐藏维度（encoder 输出的有效位置）
        # 由于 pack_padded_sequence 后有输出的位置都是有效的
        # 但 padded 位置也会产生输出 → 用 mask 过滤

        logits, attn_weights = self.decoder(
            tgt, enc_outputs, src_mask, enc_hidden,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        return logits, attn_weights


def train_epoch(
    model: Seq2Seq,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    clip: float = 1.0,
    teacher_forcing_ratio: float = 0.5,
) -> float:
    """训练一轮"""
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for src, tgt, src_len, tgt_len in loader:
        src = src.to(device)
        tgt = tgt.to(device)
        src_len = src_len.to(device)

        optimizer.zero_grad()

        # decoder 输入：去掉最后一个 token
        # decoder 目标：去掉第一个 token（<SOS>）
        dec_input = tgt[:, :-1]   # (B, T-1)
        dec_target = tgt[:, 1:]   # (B, T-1)

        logits, _ = model(src, dec_input, src_len, teacher_forcing_ratio)
        # logits: (B, T-1, V), dec_target: (B, T-1)

        # 展平后算交叉熵，ignore_index 跳过 <PAD>
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            dec_target.view(-1),
            ignore_index=model.tgt_pad_idx,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        # 统计（非 PAD token 的 loss）
        n_tokens = (dec_target != model.tgt_pad_idx).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def evaluate(
    model: Seq2Seq,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """验证：只用 Teacher Forcing（ratio=1.0 即完全用正确答案）"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for src, tgt, src_len, tgt_len in loader:
        src = src.to(device)
        tgt = tgt.to(device)
        src_len = src_len.to(device)

        dec_input = tgt[:, :-1]
        dec_target = tgt[:, 1:]

        logits, _ = model(src, dec_input, src_len, teacher_forcing_ratio=1.0)

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            dec_target.view(-1),
            ignore_index=model.tgt_pad_idx,
        )

        n_tokens = (dec_target != model.tgt_pad_idx).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    return total_loss / max(total_tokens, 1), total_loss / max(total_tokens, 1)


def train(
    model: Seq2Seq,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int = 20,
    lr: float = 0.001,
    device: torch.device = None,
    tf_ratio: float = 0.5,
):
    """总控训练函数"""
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\n{'='*60}")
    print(f"  Seq2Seq + Attention (English → French)")
    print(f"  Encoder: Bidirectional GRU, Decoder: GRU + {model.decoder.attn_type}")
    print(f"  Device: {device}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Teacher Forcing ratio: {tf_ratio}")
    print(f"{'='*60}")
    print(f"{'Epoch':>5} {'Train Loss':>12} {'Val Loss':>12} "
          f"{'Val PPL':>10} {'TF Ratio':>10} {'Time':>8}")
    print("-" * 64)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # 训练时 teacher forcing ratio 可以逐步降低（课程学习）
        current_tf = tf_ratio

        train_loss = train_epoch(
            model, train_loader, optimizer, device,
            teacher_forcing_ratio=current_tf,
        )
        val_loss, _ = evaluate(model, val_loader, device)

        val_ppl = torch.exp(torch.tensor(val_loss)).item()
        elapsed = time.time() - t0

        print(f"{epoch:>5} {train_loss:>12.4f} {val_loss:>12.4f} "
              f"{val_ppl:>9.2f} {current_tf:>9.2f} {elapsed:>7.1f}s")

    return model
