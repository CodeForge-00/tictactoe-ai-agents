"""Solver exacto con memoización (programación dinámica sobre el espacio de estados).

NO es el agente de backtracking del juego (ése es `agents/minimax.py`, DFS con
deshacer y poda). Este módulo resuelve el juego entero una vez y sirve para:
etiquetar datos de la red neuronal, medir la calidad de cualquier agente y
verificar de forma independiente el backtracking.
"""
from __future__ import annotations

from functools import lru_cache

from .game import EMPTY, legal_moves, other, to_move, winner


@lru_cache(maxsize=None)
def outcome(board: tuple) -> int:
    """Resultado con juego perfecto para quien mueve: +1 gana, 0 tablas, −1 pierde."""
    p = to_move(board)
    best = -1
    for m in legal_moves(board):
        nxt = board[:m] + (p,) + board[m + 1:]
        if winner(nxt):
            return 1
        if EMPTY not in nxt:
            v = 0
        else:
            v = -outcome(nxt)
        best = max(best, v)
        if best == 1:
            break
    return best


def move_outcomes(board) -> dict[int, int]:
    """Resultado perfecto de cada jugada legal (desde el punto de vista de quien mueve)."""
    board = tuple(board)
    p = to_move(board)
    res = {}
    for m in legal_moves(board):
        nxt = board[:m] + (p,) + board[m + 1:]
        if winner(nxt):
            res[m] = 1
        elif EMPTY not in nxt:
            res[m] = 0
        else:
            res[m] = -outcome(nxt)
    return res


def optimal_moves(board) -> list[int]:
    res = move_outcomes(board)
    best = max(res.values())
    return [m for m, v in res.items() if v == best]


def reachable_states() -> list[tuple]:
    """Todas las posiciones alcanzables jugando por turnos desde el tablero vacío (X empieza)."""
    start = (EMPTY,) * 9
    seen = {start}
    stack = [start]
    while stack:
        b = stack.pop()
        if winner(b) or EMPTY not in b:
            continue
        p = to_move(b)
        for m in legal_moves(b):
            nb = b[:m] + (p,) + b[m + 1:]
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return sorted(seen)


def playable_states() -> list[tuple]:
    """Posiciones alcanzables en las que aún hay que mover (no terminales)."""
    return [b for b in reachable_states() if not winner(b) and EMPTY in b]
