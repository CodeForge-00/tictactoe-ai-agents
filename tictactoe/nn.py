"""MLP mínimo en numpy (forward, backward, Adam). Evita depender de PyTorch."""
from __future__ import annotations

import numpy as np


class MLP:
    """Capas densas con ReLU; la última es lineal."""

    def __init__(self, sizes: list[int], seed: int = 0):
        rng = np.random.default_rng(seed)
        self.sizes = list(sizes)
        self.W = [rng.normal(0, np.sqrt(2.0 / a), (a, b)).astype(np.float32)
                  for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b, dtype=np.float32) for b in sizes[1:]]
        self._cache: list = []

    # ------------------------------------------------------------ inferencia
    def forward(self, x: np.ndarray, train: bool = False) -> np.ndarray:
        h = x
        if train:
            self._cache = []
        last = len(self.W) - 1
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            if train:
                self._cache.append(h)
            h = h @ W + b
            if i < last:
                h = np.maximum(h, 0.0)
        return h

    # ------------------------------------------------------------- gradientes
    def backward(self, dout: np.ndarray) -> list[np.ndarray]:
        """Gradientes [dW0, db0, dW1, db1, ...] dada dL/dsalida de `forward(train=True)`."""
        grads: list = [None] * (2 * len(self.W))
        d = dout
        for i in reversed(range(len(self.W))):
            h_in = self._cache[i]
            grads[2 * i] = h_in.T @ d
            grads[2 * i + 1] = d.sum(axis=0)
            if i > 0:
                d = (d @ self.W[i].T) * (h_in > 0)  # h_in ya es la salida de ReLU
        return grads

    def params(self) -> list[np.ndarray]:
        return [p for pair in zip(self.W, self.b) for p in pair]

    def copy_from(self, other: "MLP") -> None:
        for p, q in zip(self.params(), other.params()):
            p[...] = q

    # ------------------------------------------------------------ persistencia
    def save(self, path: str, **meta) -> None:
        data = {f"p{i}": p for i, p in enumerate(self.params())}
        np.savez(path, sizes=np.array(self.sizes), **data, **{k: np.array(v) for k, v in meta.items()})

    @classmethod
    def load(cls, path: str) -> "MLP":
        z = np.load(path)
        net = cls([int(s) for s in z["sizes"]])
        for i, p in enumerate(net.params()):
            p[...] = z[f"p{i}"]
        return net


class Adam:
    def __init__(self, params: list[np.ndarray], lr: float = 1e-3, b1: float = 0.9, b2: float = 0.999):
        self.params, self.lr, self.b1, self.b2 = params, lr, b1, b2
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads: list[np.ndarray], clip: float | None = 5.0) -> None:
        self.t += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            if clip is not None:
                np.clip(g, -clip, clip, out=g)
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            mh = m / (1 - self.b1 ** self.t)
            vh = v / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + 1e-8)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
