"""2 层 MLP 模型组装"""

import numpy as np
from .layers import Linear, ReLU, cross_entropy_grad


class MLP:
    """两层 MLP: Linear(784→hidden) → ReLU → Linear(hidden→10) → Softmax → CE"""

    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, num_classes: int = 10):
        self.fc1 = Linear(input_dim, hidden_dim)
        self.relu = ReLU()
        self.fc2 = Linear(hidden_dim, num_classes)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """返回 logits（未经 softmax）"""
        h = self.fc1.forward(x)
        h = self.relu.forward(h)
        return self.fc2.forward(h)

    def backward(self, dout: np.ndarray):
        """从 loss 的梯度开始反向传播"""
        d = self.fc2.backward(dout)
        d = self.relu.backward(d)
        self.fc1.backward(d)

    @property
    def linear_layers(self) -> list:
        """返回有可训练参数的层列表，供优化器使用"""
        return [self.fc1, self.fc2]

    @property
    def total_params(self) -> int:
        """统计总参数量"""
        return sum(layer.W.size + layer.b.size for layer in self.linear_layers)
