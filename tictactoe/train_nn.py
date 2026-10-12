"""Entrena la red neuronal para que juegue de forma óptima.

    python -m tictactoe.train_nn              # modo fuerza (el que se distribuye)
    python -m tictactoe.train_nn --holdout    # experimento de generalización

Como el juego está resuelto, se etiquetan las posiciones alcanzables con el conjunto de
jugadas óptimas (solver exacto) y se entrena un clasificador. Hay dos preguntas distintas:

* MODO FUERZA (por defecto): "¿cuál es el mejor jugador que puede ser esta red?". Se entrena
  con TODAS las posiciones (las 4 520 pueden darse en una partida, no hay posiciones "no vistas")
  y el modelo se elige por la medida que de verdad importa: la probabilidad EXACTA de perder
  contra un rival perfecto (`match.loss_vs_perfect`). Elegir por la jugada óptima media sería un
  error: casi no ve las pocas posiciones críticas (ver docs/03).

* MODO --holdout: "¿generaliza la red a posiciones que no ha visto?". Las posiciones se reparten
  por clase de simetría en entrenamiento (60 %), validación (20 %, para parar) y test (20 %).
  Si se repartieran posiciones sueltas, una versión girada de una posición de entrenamiento se
  "filtraría" a validación.
"""
from __future__ import annotations

import argparse
import os
import random
import time
from typing import Callable

import numpy as np

from . import match, solver
from .agents.neural import MODELS_DIR, NeuralAgent
from .game import SYMMETRIES, canonical, features, to_move, transform_board, transform_move
from .nn import MLP, Adam, softmax


def build_dataset(augment: bool, states: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """X (n,18) rasgos, T (n,9) objetivo (uniforme entre jugadas óptimas), M (n,9) casillas libres."""
    X, T, M = [], [], []
    for b in states:
        p = to_move(b)
        opt = solver.optimal_moves(b)
        variants = SYMMETRIES if augment else SYMMETRIES[:1]
        seen = set()
        for perm in variants:
            tb = tuple(transform_board(b, perm))
            if tb in seen:                       # posiciones simétricas consigo mismas
                continue
            seen.add(tb)
            t = np.zeros(9, np.float32)
            for m in opt:
                t[transform_move(m, perm)] = 1.0 / len(opt)
            X.append(features(tb, p))
            T.append(t)
            M.append(np.array([v == 0 for v in tb]))
    return np.array(X), np.array(T), np.array(M)


def split3(states: list, seed: int, val_frac: float = 0.2, test_frac: float = 0.2):
    """Reparte las posiciones por clase de simetría: (entrenamiento, validación, test)."""
    classes = sorted({canonical(b) for b in states})
    random.Random(seed).shuffle(classes)
    n_val, n_test = int(len(classes) * val_frac), int(len(classes) * test_frac)
    cls = {"test": set(classes[:n_test]), "val": set(classes[n_test:n_test + n_val])}
    parts = {"train": [], "val": [], "test": []}
    for b in states:
        c = canonical(b)
        parts["test" if c in cls["test"] else "val" if c in cls["val"] else "train"].append(b)
    return parts["train"], parts["val"], parts["test"]


class Evaluator:
    """Mide, vectorizado, en qué fracción de posiciones la red elige una jugada óptima."""

    def __init__(self, states: list):
        self.F = np.array([features(b, to_move(b)) for b in states])
        self.free = np.array([[v == 0 for v in b] for b in states])
        self.opt = np.zeros((len(states), 9), bool)
        for i, b in enumerate(states):
            self.opt[i, solver.optimal_moves(b)] = True

    def rate(self, net: MLP) -> float:
        logits = np.where(self.free, net.forward(self.F), -np.inf)
        return float(self.opt[np.arange(len(self.F)), logits.argmax(axis=1)].mean())


def loss_vs_perfect(net: MLP) -> float:
    return match.loss_vs_perfect(NeuralAgent(net))


def train(net: MLP, X, T, M, epochs: int, lr: float, seed: int, l2: float = 1e-3,
          monitor: Callable[[MLP], float] | None = None, higher_is_better: bool = True,
          every: int = 1, patience: int | None = None, batch: int = 128, log=None) -> int:
    """Entrena. Con `monitor`, se queda con los pesos de la mejor época (a igualdad, la más tardía).

    Devuelve la época elegida.
    """
    opt = Adam(net.params(), lr=lr)
    rng = np.random.default_rng(seed)
    n = len(X)
    sign = 1.0 if higher_is_better else -1.0
    best, best_ep, best_params = -np.inf, 0, None
    for ep in range(1, epochs + 1):
        order = rng.permutation(n)
        total = 0.0
        for i in range(0, n, batch):
            idx = order[i:i + batch]
            logits = np.where(M[idx], net.forward(X[idx], train=True), -1e9)   # enmascarar casillas ocupadas
            p = softmax(logits)
            total += float(-(T[idx] * np.log(p + 1e-9)).sum())
            grads = net.backward((p - T[idx]) / len(idx))
            for k in range(0, len(grads), 2):                                  # L2 sólo sobre pesos
                grads[k] += l2 * net.W[k // 2]
            opt.step(grads)
        if monitor is not None and ep % every == 0:
            r = monitor(net)
            if sign * r >= best:                   # >= : a igualdad gana la época más entrenada
                best, best_ep = sign * r, ep
                best_params = [q.copy() for q in net.params()]
            if log and (ep % (5 * every) == 0 or ep == every):
                log(f"época {ep:>4}  pérdida {total / n:.4f}  métrica {r:.4f}")
            if patience and ep - best_ep >= patience:
                break
    if best_params is not None:
        for q, b in zip(net.params(), best_params):
            q[...] = b
        return best_ep
    return epochs


def label_entropy(T: np.ndarray) -> float:
    """Suelo de la pérdida: la entropía media de las etiquetas blandas (varias jugadas óptimas)."""
    return float(-(T * np.log(T + 1e-12)).sum(axis=1).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holdout", action="store_true", help="experimento de generalización (train/val/test)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--l2", type=float, default=None, help="regularización L2 de los pesos")
    ap.add_argument("--patience", type=int, default=40, help="(sólo --holdout) épocas sin mejora para parar")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="dónde guardar (por defecto models/nn.npz; en --holdout no se guarda)")
    args = ap.parse_args()

    # Hiperparámetros por modo (se pueden sobrescribir desde la línea de órdenes)
    d = dict(epochs=150, hidden=64, lr=1e-3, l2=1e-3) if args.holdout else dict(epochs=100, hidden=128, lr=3e-3, l2=1e-4)
    epochs, hidden, lr, l2 = (getattr(args, k) if getattr(args, k) is not None else d[k]
                              for k in ("epochs", "hidden", "lr", "l2"))

    states = solver.playable_states()
    t0 = time.time()
    if args.holdout:
        train_s, val_s, test_s = split3(states, args.seed)
        X, T, M = build_dataset(True, train_s)
        print(f"posiciones: {len(states)} | entrenamiento {len(train_s)} ({len(X)} con simetrías) "
              f"| validación {len(val_s)} | test {len(test_s)}")
        print(f"suelo de la pérdida (entropía de las etiquetas): {label_entropy(T):.3f}")
        net = MLP([18, hidden, hidden, 9], seed=args.seed)
        val, test, every = Evaluator(val_s), Evaluator(test_s), Evaluator(states)
        best_ep = train(net, X, T, M, epochs, lr, args.seed, l2=l2, monitor=val.rate, higher_is_better=True,
                        patience=args.patience, log=lambda s: print(s, flush=True))
        print(f"entrenamiento: {time.time() - t0:.0f}s | mejor época (por validación): {best_ep}")
        print(f"jugada óptima  validación {val.rate(net):.1%} | TEST {test.rate(net):.1%} | todas las posiciones {every.rate(net):.1%}")
        print(f"probabilidad de perder contra un rival perfecto: {loss_vs_perfect(net):.1%}")
    else:
        X, T, M = build_dataset(True, states)
        print(f"posiciones: {len(states)} ({len(X)} muestras con simetrías) | red 18→{hidden}→{hidden}→9")
        net = MLP([18, hidden, hidden, 9], seed=args.seed)
        best_ep = train(net, X, T, M, epochs, lr, args.seed, l2=l2, monitor=loss_vs_perfect,
                        higher_is_better=False, every=5, log=lambda s: print(s, flush=True))
        print(f"entrenamiento: {time.time() - t0:.0f}s | época elegida (menor pérdida vs perfecto): {best_ep}")
        print(f"probabilidad de perder contra un rival perfecto: {loss_vs_perfect(net):.1%}")
        print(f"jugada óptima en todas las posiciones: {Evaluator(states).rate(net):.1%}")

    out = args.out or (None if args.holdout else os.path.join(MODELS_DIR, "nn.npz"))
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        net.save(out)
        print("guardado", out)


if __name__ == "__main__":
    main()
