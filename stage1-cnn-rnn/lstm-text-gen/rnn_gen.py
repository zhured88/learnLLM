"""
RNN / LSTM / GRU 字符级语言模型 —— 从零实现

核心概念：
  - 字符级语言模型：输入一串字符，预测每个位置的下一个字符
  - 嵌入层（Embedding）：把 65 个离散字符映射到 256 维连续空间
  - RNN/LSTM/GRU：循环处理序列，维护"到目前为止看到了什么"的隐藏状态
  - 温度采样：控制生成文本的"创造性和连贯性之间的平衡"
"""

import torch
import torch.nn as nn


class CharRNN(nn.Module):
    """
    字符级 RNN 语言模型，支持 vanilla RNN / LSTM / GRU 三种架构

    模型结构：
      输入字符序列 → Embedding → Dropout → RNN/LSTM/GRU → Dropout → FC → 输出每个字符的得分

    维度流转（以默认参数为例）：
      (B, 100)        ← 100 个字符索引
      (B, 100, 256)   ← embedding 后每个字符变成 256 维向量
      (B, 100, 512)   ← RNN 处理后每个时间步输出 512 维隐藏状态
      (B, 100, 65)    ← FC 映射回词表大小，每个字符一个得分
    """

    def __init__(
        self,
        vocab_size: int,        # 词表大小（莎士比亚文集有 65 个不同字符）
        embed_dim: int = 256,   # 嵌入向量维度（每个字符用多长的向量表示）
        hidden_dim: int = 512,  # RNN 隐藏状态维度（模型的"记忆容量"）
        num_layers: int = 2,    # RNN 堆叠层数（>1 时构成"深度 RNN"）
        rnn_type: str = "lstm", # RNN 类型：["rnn", "lstm", "gru"]
        dropout: float = 0.3,   # Dropout 概率（30% 随机丢弃，防过拟合）
    ):
        super().__init__()
        self.rnn_type = rnn_type.lower()  # 统一小写，用户传 "LSTM" 也能匹配
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # --- 嵌入层：把离散字符索引映射到连续向量空间 ---
        # 65 个字符 → 65×256 的查找表。训练后相似字符（如 'a' 和 'A'）会有相近的向量
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # --- RNN 层：一行代码切换三种架构 ---
        # 字典映射是 Python 的策略模式：{"lstm": LSTM, "gru": GRU, "rnn": RNN}
        rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[self.rnn_type]
        self.rnn = rnn_cls(
            embed_dim, hidden_dim, num_layers,
            batch_first=True,  # 输入形状约定为 (batch, seq_len, features)，更直观
            dropout=dropout if num_layers > 1 else 0,  # 多层之间加 dropout，单层不加
        )

        # --- 输出层：把隐藏状态映射回词表大小 ---
        # 512 维 → 65 维，每个维度是"下一个字符是 c 的得分"
        self.fc = nn.Linear(hidden_dim, vocab_size)

        # --- 额外的 Dropout 层（放在 embedding 之后、RNN 输出之后）---
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, hidden=None):
        """
        前向传播

        参数：
          x:      (batch, seq_len) 字符索引张量，如 [64, 100]
          hidden: 隐藏状态，None 则自动初始化为零
        返回：
          logits: (batch, seq_len, vocab_size) 每个位置每个字符的得分
          hidden: 更新后的隐藏状态（用于下一段序列）
        """
        # ① 嵌入：每个字符索引 → 256 维向量
        embed = self.dropout(self.embedding(x))     # (B, T) → (B, T, 256)

        # ② RNN 处理：逐时间步更新隐藏状态
        out, hidden = self.rnn(embed, hidden)        # (B, T, 256) → (B, T, 512)

        # ③ Dropout 正则化 + 输出映射
        out = self.dropout(out)                      # 随机丢弃 30% 神经元
        logits = self.fc(out)                        # (B, T, 512) → (B, T, 65)

        return logits, hidden

    def init_hidden(self, batch_size: int, device: torch.device):
        """
        初始化隐藏状态（全零）

        LSTM 特殊处理：需要 (h0, c0) 两个状态
          - h0：短期记忆，暴露给外部输出
          - c0：长期记忆，内部细胞状态
        GRU 和 RNN 只需要一个状态 h0

        形状说明：
          (num_layers, batch_size, hidden_dim)
            ↑ 维度 0    ↑ 维度 1     ↑ 维度 2
              层数       每个样本       记忆容量
              独立状态   独立状态
        """
        if self.rnn_type == "lstm":
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
            return (h0, c0)  # LSTM 需要元组 (h, c)
        else:
            return torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)

    def generate(
        self,
        start_str: str,           # 起始字符串（如 "ROMEO:"），模型从这里"接龙"
        char_to_idx: dict,        # 字符→索引映射表
        idx_to_char: dict,        # 索引→字符映射表
        length: int = 200,        # 要生成多少个字符
        temperature: float = 0.8, # 温度：<1 保守，>1 冒险
        device: torch.device = torch.device("cpu"),
    ) -> str:
        """
        从起始字符串开始，逐字符生成文本

        生成流程：
          ① 用起始字符串"预热" RNN 的隐藏状态
          ② 取最后一个字符作为起点
          ③ 循环：喂入上一字符 → 得到 logits → 温度缩放 → softmax → 按概率采样 → 得到下一字符
          ④ 重复 length 次
        """
        self.eval()  # 切换到测试模式（Dropout 停止丢弃）
        with torch.no_grad():  # 关闭 autograd，生成时不需要反向传播
            # --- ① 编码起始字符串 ---
            # .get(c, 0)：如果遇到不认识的字，默认返回索引 0
            chars = [char_to_idx.get(c, 0) for c in start_str]
            inp = torch.tensor([chars], dtype=torch.long, device=device)  # (1, len)

            # --- ② 喂入起始序列，让 RNN 建立"语境" ---
            hidden = self.init_hidden(1, device)  # batch_size=1（只生成一段）
            _, hidden = self(inp, hidden)          # 丢弃输出的 logits，只要更新后的 hidden

            # --- ③ 逐字符生成 ---
            result = list(start_str)
            next_char_idx = chars[-1]  # 起点：起始字符串的最后一个字符

            for _ in range(length):
                # 喂入上一个字符（形状：1×1 的张量）
                inp = torch.tensor([[next_char_idx]], dtype=torch.long, device=device)
                logits, hidden = self(inp, hidden)

                # --- 温度采样 ---
                # 温度 < 1：放大高分和低分之间的差距 → 生成更确定、更保守
                # 温度 > 1：缩小差距 → 生成更多样、更有创造性
                # 温度 → 0：等价于 argmax（永远选概率最大的）
                # 温度 → ∞：等价于均匀随机（胡说八道）
                logits = logits[0, -1] / max(temperature, 1e-8)  # 取最后一个时间步，除以温度
                probs = torch.softmax(logits, dim=-1)  # 转成概率分布

                # torch.multinomial：按概率分布采样（而非永远取最大概率）
                # 如果是 argmax → 生成文本会单调重复，因为总是选同一个字符
                # 按概率采样 → 保留了多样性，但大概率字符仍然更常被选中
                next_char_idx = int(torch.multinomial(probs, 1).item())
                result.append(idx_to_char[next_char_idx])

        return "".join(result)
