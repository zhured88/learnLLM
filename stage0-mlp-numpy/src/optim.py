"""优化器"""

import numpy as np


class SGD:
    """带动量的随机梯度下降 — 直接操作 layer 对象，每次 step 动态读取梯度"""

    def __init__(self, layers: list, lr: float = 0.01, momentum: float = 0.9):
        """
        layers: 包含 .W, .b, .dW, .db 属性的层对象列表（如 Linear）
        """
        self.layers = layers
        self.lr = lr
        self.momentum = momentum

        self.v_W = [np.zeros_like(layer.W) for layer in layers]
        self.v_b = [np.zeros_like(layer.b) for layer in layers]

    def step(self):
        for i, layer in enumerate(self.layers):
            self.v_W[i] = self.momentum * self.v_W[i] - self.lr * layer.dW
            self.v_b[i] = self.momentum * self.v_b[i] - self.lr * layer.db
            layer.W += self.v_W[i]
            layer.b += self.v_b[i]
