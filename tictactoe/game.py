"""Reglas del tres en raya, representaciones del tablero y simetrías.

Tablero: lista de 9 enteros (fila a fila). 0 = vacío, 1 = X (empieza), 2 = O.

    0 | 1 | 2
    ---------
    3 | 4 | 5
    ---------
    6 | 7 | 8
"""
from __future__ import annotations

import numpy as np

EMPTY, X, O = 0, 1, 2
LINES = ((0, 1, 2), (3, 4, 5), (6, 7, 8),      # filas
         (0, 3, 6), (1, 4, 7), (2, 5, 8),      # columnas
         (0, 4, 8), (2, 4, 6))                 # diagonales


def other(player: int) -> int:
    return 3 - player


def winning_line(board) -> tuple | None:
    for line in LINES:
        a, b, c = line
        if board[a] and board[a] == board[b] == board[c]:
            return line
    return None


def winner(board) -> int:
    """0 si nadie ha ganado, 1 ó 2 si ganó ese jugador."""
    line = winning_line(board)
    return board[line[0]] if line else 0


def legal_moves(board) -> list[int]:
    return [i for i in range(9) if board[i] == EMPTY]


def is_full(board) -> bool:
    return EMPTY not in board


def is_over(board) -> bool:
    return winner(board) != 0 or is_full(board)


def to_move(board) -> int:
    """Quién juega: X si hay igual número de fichas, si no O."""
    return X if list(board).count(X) == list(board).count(O) else O


class Game:
    """Una partida: `move(i)` coloca la ficha del jugador en turno."""

    def __init__(self, first: int = X):
        self.first = first
        self.reset()

    def reset(self, first: int | None = None) -> None:
        if first is not None:
            self.first = first
        self.board = [EMPTY] * 9
        self.player = self.first
        self.winner = 0
        self.line: tuple | None = None
        self.over = False

    def move(self, i: int) -> None:
        if self.over or self.board[i] != EMPTY:
            raise ValueError(f"movimiento ilegal: {i}")
        self.board[i] = self.player
        self.line = winning_line(self.board)
        if self.line:
            self.winner, self.over = self.player, True
        elif is_full(self.board):
            self.over = True
        else:
            self.player = other(self.player)


# --------------------------------------------------------------------------
# Representaciones desde la perspectiva del jugador que mueve ("propias" vs
# "del rival"). Así un mismo modelo juega con X y con O.
# --------------------------------------------------------------------------
def features(board, player: int) -> np.ndarray:
    """18 rasgos: 9 casillas propias (1/0) + 9 casillas del rival (1/0)."""
    b = np.asarray(board)
    return np.concatenate([(b == player), (b == other(player))]).astype(np.float32)


_POW3 = [3 ** i for i in range(9)]


def state_index(board, player: int) -> int:
    """Índice en base 3 (0 vacío, 1 propia, 2 rival): 0 … 19 682."""
    return sum(_POW3[i] * (0 if v == EMPTY else 1 if v == player else 2) for i, v in enumerate(board))


# --------------------------------------------------------------------------
# Simetrías del cuadrado (4 giros × 2 reflejos = 8)
# --------------------------------------------------------------------------
def _build_symmetries() -> list[tuple]:
    rot = tuple((2 - c) * 3 + r for r in range(3) for c in range(3))     # giro 90°
    flip = tuple(r * 3 + (2 - c) for r in range(3) for c in range(3))    # reflejo horizontal
    ident = tuple(range(9))
    perms = {ident}
    frontier = [ident]
    while frontier:
        p = frontier.pop()
        for g in (rot, flip):
            q = tuple(p[g[i]] for i in range(9))
            if q not in perms:
                perms.add(q)
                frontier.append(q)
    return sorted(perms)


SYMMETRIES = _build_symmetries()   # perm[i] = casilla de origen de la casilla i


def transform_board(board, perm) -> list:
    return [board[perm[i]] for i in range(9)]


def transform_move(move: int, perm) -> int:
    """Dónde cae la jugada `move` del tablero original tras aplicar `perm`."""
    return perm.index(move)


def canonical(board) -> tuple:
    """Representante de la clase de simetría (el menor de las 8 variantes)."""
    return min(tuple(transform_board(board, p)) for p in SYMMETRIES)
