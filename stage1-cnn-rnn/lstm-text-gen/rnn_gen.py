"""RNN / LSTM / GRU 字符级语言模型 — 从零实现"""

import torch
import torch.nn as nn


class CharRNN(nn.Module):
    """字符级 RNN 语言模型，支持 vanilla RNN / LSTM / GRU"""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        rnn_type: str = "lstm",
        dropout: float = 0.3,
    ):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[self.rnn_type]
        self.rnn = rnn_cls(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
        )

        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, hidden=None):
        """
        x: (batch, seq_len)  字符索引
        返回: logits (batch, seq_len, vocab_size), hidden
        """
        embed = self.dropout(self.embedding(x))     # (B, T, E)
        out, hidden = self.rnn(embed, hidden)        # (B, T, H)
        out = self.dropout(out)
        logits = self.fc(out)                        # (B, T, V)
        return logits, hidden

    def init_hidden(self, batch_size: int, device: torch.device):
        """初始化隐藏状态（LSTM 需要 h 和 c 两元组）"""
        if self.rnn_type == "lstm":
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
            return (h0, c0)
        else:
            return torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)

    def generate(
        self,
        start_str: str,
        char_to_idx: dict,
        idx_to_char: dict,
        length: int = 200,
        temperature: float = 0.8,
        device: torch.device = torch.device("cpu"),
    ) -> str:
        """从起始字符串生成文本"""
        self.eval()
        with torch.no_grad():
            # 编码起始字符串
            chars = [char_to_idx.get(c, 0) for c in start_str]
            inp = torch.tensor([chars], dtype=torch.long, device=device)

            hidden = self.init_hidden(1, device)
            # 先喂入起始序列，建立隐藏状态
            _, hidden = self(inp, hidden)

            result = list(start_str)
            next_char_idx = chars[-1]

            for _ in range(length):
                inp = torch.tensor([[next_char_idx]], dtype=torch.long, device=device)
                logits, hidden = self(inp, hidden)

                # 温度采样
                logits = logits[0, -1] / max(temperature, 1e-8)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

                # 偶尔选 top-k 避免极端分布
                next_char_idx = int(torch.multinomial(
                    torch.tensor(probs), 1
                ).item())
                result.append(idx_to_char[next_char_idx])

        return "".join(result)
