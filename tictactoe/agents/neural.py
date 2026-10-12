"""Agentes basados en un modelo entrenado: red neuronal (supervisada) y Q-learning."""
from __future__ import annotations

import os
import random

import numpy as np

from ..game import features, legal_moves, state_index
from ..nn import MLP, softmax
from .base import Agent

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")


class NeuralAgent(Agent):
    """MLP 18 → 64 → 64 → 9 entrenado para predecir la jugada óptima (ver train_nn.py)."""

    name = "Red neuronal"
    score_kind = "prob"
    filename = "nn.npz"

    def __init__(self, net: MLP | None = None, path: str | None = None):
        if net is None:
            path = path or os.path.join(MODELS_DIR, self.filename)
            if not os.path.exists(path):
                raise FileNotFoundError(f"No hay modelo en {path}. Entrénalo con: python -m tictactoe.train_nn")
            net = MLP.load(path)
        self.net = net

    def scores(self, board, player):
        logits = self.net.forward(features(board, player))
        legal = legal_moves(board)
        s = np.full(9, np.nan)
        s[legal] = softmax(logits[legal])
        return s

    def act(self, board, player):
        s = self.scores(board, player)
        return int(np.nanargmax(s))


class RLAgent(Agent):
    """Política voraz sobre una tabla Q entrenada por Q-learning con auto-juego."""

    name = "Refuerzo (Q-learning)"
    score_kind = "q"
    filename = "rl.npz"

    def __init__(self, q: np.ndarray | None = None, path: str | None = None, seed: int | None = None):
        if q is None:
            path = path or os.path.join(MODELS_DIR, self.filename)
            if not os.path.exists(path):
                raise FileNotFoundError(f"No hay modelo en {path}. Entrénalo con: python -m tictactoe.train_rl")
            q = np.load(path)["q"]
        self.q = q
        self.rng = random.Random(seed)

    def scores(self, board, player):
        row = self.q[state_index(board, player)]
        s = np.full(9, np.nan)
        legal = legal_moves(board)
        s[legal] = row[legal]
        return s

    def act(self, board, player):
        s = self.scores(board, player)
        best = np.nanmax(s)
        return self.rng.choice([m for m in legal_moves(board) if s[m] >= best - 1e-9])
