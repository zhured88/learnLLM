"""核心层实现：Linear, ReLU, Softmax — 纯 numpy + 手写反向传播"""

import numpy as np


class Linear:
    """全连接层: y = x @ W + b"""

    def __init__(self, in_features: int, out_features: int):
        # Kaiming init 适配 ReLU
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros(out_features)

        self.x: np.ndarray | None = None
        self.dW: np.ndarray | None = None
        self.db: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout: np.ndarray) -> np.ndarray:
        # dout: (N, out_features)
        self.dW = self.x.T @ dout               # (in, out)
        self.db = dout.sum(axis=0)              # (out,)
        return dout @ self.W.T                  # (N, in)  传给上一层


class ReLU:
    """ReLU 激活: max(0, x)"""

    def __init__(self):
        self.mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * self.mask


class Softmax:
    """Softmax（与 CrossEntropyLoss 合并计算梯度以保持数值稳定）"""

    def __init__(self):
        self.probs: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # 数值稳定技巧：减去每行最大值
        shifted = x - x.max(axis=1, keepdims=True)
        exp_x = np.exp(shifted)
        self.probs = exp_x / exp_x.sum(axis=1, keepdims=True)
        return self.probs

    def backward(self, dout: np.ndarray) -> np.ndarray:
        # 仅用于非 CrossEntropy 场景。与 CE 合用时用 cross_entropy_grad
        batch_size = self.probs.shape[0]
        dx = np.empty_like(self.probs)
        for i in range(batch_size):
            p = self.probs[i].reshape(-1, 1)
            jacobian = np.diagflat(p) - p @ p.T  # softmax 雅可比矩阵
            dx[i] = dout[i] @ jacobian
        return dx


def cross_entropy_grad(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Softmax + CrossEntropy 合并梯度：probs - y_onehot，数值稳定且高效"""
    batch_size = probs.shape[0]
    num_classes = probs.shape[1]
    y_onehot = np.zeros((batch_size, num_classes))
    y_onehot[np.arange(batch_size), y] = 1
    return (probs - y_onehot) / batch_size
