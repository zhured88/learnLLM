"""训练循环"""

import numpy as np
import time
from .model import MLP
from .losses import CrossEntropyLoss
from .optim import SGD
from .data import get_batches


def compute_accuracy(logits: np.ndarray, y: np.ndarray) -> float:
    return (logits.argmax(axis=1) == y).mean()


def train(
    model: MLP,
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 0.1,
    momentum: float = 0.9,
    lr_decay: float = 0.95,
):
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.linear_layers, lr=lr, momentum=momentum)

    n_batches = train_X.shape[0] // batch_size

    print(f"{'Epoch':>5} {'Train Loss':>12} {'Train Acc':>10} {'Test Acc':>10} {'Time':>8}")
    print("-" * 52)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        total_loss = 0.0
        total_acc = 0.0
        n_batch = 0

        for X_batch, y_batch in get_batches(train_X, train_y, batch_size):
            # forward
            logits = model.forward(X_batch)
            loss = criterion.forward(logits, y_batch)

            # backward
            dout = criterion.backward()
            model.backward(dout)

            # update
            optimizer.step()

            total_loss += loss
            total_acc += compute_accuracy(logits, y_batch)
            n_batch += 1

        # 学习率衰减
        optimizer.lr *= lr_decay

        # 测试集评估
        test_logits = model.forward(test_X)
        test_acc = compute_accuracy(test_logits, test_y)

        avg_loss = total_loss / n_batch
        avg_acc = total_acc / n_batch
        elapsed = time.time() - t0

        print(
            f"{epoch:>5} {avg_loss:>12.4f} {avg_acc:>9.2%} {test_acc:>9.2%} {elapsed:>7.1f}s"
        )

    return model
