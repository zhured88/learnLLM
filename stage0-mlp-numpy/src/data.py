"""MNIST 数据加载 — 从 OpenML 下载，纯 numpy"""

import numpy as np
import os
import gzip
import struct
import urllib.request

MNIST_URLS = {
    "train_images": "https://github.com/golbin/TensorFlow-MNIST/raw/master/mnist/data/train-images-idx3-ubyte.gz",
    "train_labels": "https://github.com/golbin/TensorFlow-MNIST/raw/master/mnist/data/train-labels-idx1-ubyte.gz",
    "test_images": "https://github.com/golbin/TensorFlow-MNIST/raw/master/mnist/data/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://github.com/golbin/TensorFlow-MNIST/raw/master/mnist/data/t10k-labels-idx1-ubyte.gz",
}

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".mnist_cache")


def _download(url: str, dest: str):
    """下载文件到目标路径"""
    if os.path.exists(dest):
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  下载 {os.path.basename(dest)} ...")
    urllib.request.urlretrieve(url, dest)


def _parse_images(path: str) -> np.ndarray:
    """解析 IDX3 图片文件 -> (N, 784) float32 [0, 1]"""
    with gzip.open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
    return data.astype(np.float32) / 255.0


def _parse_labels(path: str) -> np.ndarray:
    """解析 IDX1 标签文件 -> (N,) int64"""
    with gzip.open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.astype(np.int64)


def load_mnist() -> tuple:
    """
    返回 (train_X, train_y, test_X, test_y)

    train_X: (60000, 784) float32
    train_y: (60000,)      int64
    test_X:  (10000, 784) float32
    test_y:  (10000,)      int64
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    files = {}
    for key, url in MNIST_URLS.items():
        filename = os.path.basename(url)
        dest = os.path.join(CACHE_DIR, filename)
        _download(url, dest)
        files[key] = dest

    train_X = _parse_images(files["train_images"])
    train_y = _parse_labels(files["train_labels"])
    test_X  = _parse_images(files["test_images"])
    test_y  = _parse_labels(files["test_labels"])

    return train_X, train_y, test_X, test_y


def get_batches(X: np.ndarray, y: np.ndarray, batch_size: int):
    """生成 mini-batch"""
    n = X.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)
    for start in range(0, n, batch_size):
        batch_idx = indices[start : start + batch_size]
        yield X[batch_idx], y[batch_idx]
