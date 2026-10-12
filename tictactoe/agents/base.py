"""Interfaz común de los agentes."""
from __future__ import annotations

import numpy as np


class Agent:
    """Un agente elige una casilla (0-8) para el jugador `player` (1 = X, 2 = O).

    `scores` devuelve una puntuación por casilla (NaN si es ilegal) que la interfaz
    usa para mostrar el análisis; `score_kind` indica cómo formatearla:
    "outcome" (>0 gana, 0 tablas, <0 pierde), "prob" (probabilidad) o "q" (valor Q).
    """

    name = "agent"
    score_kind = "outcome"

    def reset(self) -> None:
        pass

    def scores(self, board, player: int) -> np.ndarray:
        raise NotImplementedError

    def act(self, board, player: int) -> int:
        raise NotImplementedError
