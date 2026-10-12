"""Agente de backtracking: minimax (negamax) con deshacer y poda alfa-beta.

Backtracking = DFS que elige una jugada, la APLICA al tablero, recurre y la
DESHACE al volver. En un juego de dos jugadores de suma cero, "la mejor
jugada" se define con minimax: yo maximizo, el rival minimiza (negamax lo
escribe con un único caso: el valor para mí es el opuesto al valor del rival).

Como el tres en raya tiene sólo 5 478 posiciones, el árbol se recorre ENTERO y
el agente juega de forma perfecta. Es el caso límite en el que la búsqueda
exhaustiva es viable (en Breakout no lo era: ahí hubo que limitar la profundidad).

Valor de una hoja: ganar cuenta `10 − profundidad` (mejor ganar pronto) y perder
`profundidad − 10` (mejor perder tarde). Las tablas valen 0.
"""
from __future__ import annotations

import random

import numpy as np

from ..game import EMPTY, legal_moves, other, winner
from .base import Agent

INF = 100


class MinimaxAgent(Agent):
    name = "Backtracking"
    score_kind = "outcome"

    def __init__(self, alpha_beta: bool = True, seed: int | None = None):
        self.alpha_beta = alpha_beta
        self.rng = random.Random(seed)
        self.nodes = 0                       # nodos visitados en la última búsqueda

    # ------------------------------------------------------------- búsqueda
    def _negamax(self, b: list, p: int, depth: int, alpha: int, beta: int) -> int:
        """Valor para el jugador `p` que mueve en `b`. Deja `b` como lo encontró."""
        self.nodes += 1
        best = -INF
        for m in range(9):
            if b[m] != EMPTY:
                continue
            b[m] = p                                       # APLICAR
            if winner(b):
                v = 10 - depth
            elif EMPTY not in b:
                v = 0
            else:
                v = -self._negamax(b, other(p), depth + 1, -beta, -alpha)
            b[m] = EMPTY                                   # DESHACER
            if v > best:
                best = v
            if self.alpha_beta:
                if best > alpha:
                    alpha = best
                if alpha >= beta:                          # PODA: el rival no permitirá esta rama
                    break
        return best

    def _move_value(self, b: list, p: int, m: int) -> int:
        """Valor exacto de jugar `m` (ventana completa: sin poda entre hermanos)."""
        b[m] = p
        if winner(b):
            v = 10
        elif EMPTY not in b:
            v = 0
        else:
            v = -self._negamax(b, other(p), 1, -INF, INF)
        b[m] = EMPTY
        return v

    # --------------------------------------------------------------- interfaz
    def scores(self, board, player):
        b = list(board)
        self.nodes = 0
        s = np.full(9, np.nan)
        for m in legal_moves(b):
            s[m] = self._move_value(b, player, m)
        return s

    def act(self, board, player):
        s = self.scores(board, player)
        best = np.nanmax(s)
        return self.rng.choice([m for m in legal_moves(board) if s[m] == best])
