from __future__ import annotations

import random

import numpy as np

from ..game import legal_moves
from .base import Agent


class RandomAgent(Agent):
    """Juega al azar entre las casillas libres. Línea base y sparring."""

    name = "Aleatorio"

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def scores(self, board, player):
        s = np.full(9, np.nan)
        s[legal_moves(board)] = 0.0
        return s

    def act(self, board, player):
        return self.rng.choice(legal_moves(board))
