"""损失函数"""

import numpy as np


class CrossEntropyLoss:
    """多分类交叉熵损失。与 Softmax 合并计算以保证数值稳定性。"""

    def __init__(self):
        self.probs: np.ndarray | None = None
        self.y: np.ndarray | None = None

    def forward(self, logits: np.ndarray, y: np.ndarray) -> float:
        """
        logits: (N, C)  未经 softmax 的原始输出
        y:      (N,)    真实标签（整数，非 one-hot）
        返回:   scalar loss
        """
        self.y = y
        batch_size = logits.shape[0]

        # 数值稳定：减去每行最大值再算 softmax
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_shifted = np.exp(shifted)
        self.probs = exp_shifted / exp_shifted.sum(axis=1, keepdims=True)

        # NLL loss
        correct_log_probs = -np.log(
            self.probs[np.arange(batch_size), y] + 1e-12
        )
        return correct_log_probs.mean()

    def backward(self) -> np.ndarray:
        """返回 loss 对 logits 的梯度: (probs - y_onehot) / N"""
        batch_size = self.probs.shape[0]
        num_classes = self.probs.shape[1]
        y_onehot = np.zeros((batch_size, num_classes))
        y_onehot[np.arange(batch_size), self.y] = 1
        return (self.probs - y_onehot) / batch_size
