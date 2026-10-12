"""Partidas headless entre agentes y métricas de calidad."""
from __future__ import annotations

import itertools

import numpy as np

from . import solver
from .game import EMPTY, O, X, Game, to_move


def play_game(agent_x, agent_o, first: int = X) -> int:
    """Juega una partida. Devuelve el ganador (1 = X, 2 = O) o 0 si hay tablas."""
    g = Game(first)
    agents = {X: agent_x, O: agent_o}
    while not g.over:
        g.move(agents[g.player].act(g.board, g.player))
    return g.winner


def duel(a, b, games: int = 200) -> tuple[int, int, int]:
    """`a` contra `b`, alternando quién empieza. Devuelve (victorias de a, tablas, victorias de b)."""
    wa = d = wb = 0
    for i in range(games):
        a_is_x = i % 2 == 0
        w = play_game(a, b) if a_is_x else play_game(b, a)
        if w == 0:
            d += 1
        elif (w == X) == a_is_x:
            wa += 1
        else:
            wb += 1
    return wa, d, wb


def optimal_rates(agent) -> tuple[float, float]:
    """Calidad de la política contra el solver exacto, sobre todas las posiciones alcanzables.

    Devuelve (tasa de jugada óptima, tasa en las posiciones con UNA sola jugada óptima).
    Si el agente desempata al azar, se promedia sobre las jugadas empatadas.
    """
    total = single = hits = hits_single = 0.0
    for b in solver.playable_states():
        s = agent.scores(list(b), to_move(b))
        best = np.nanmax(s)
        top = {m for m in range(9) if not np.isnan(s[m]) and s[m] >= best - 1e-9}
        opt = set(solver.optimal_moves(b))
        p = len(top & opt) / len(top)
        total += 1
        hits += p
        if len(opt) == 1:
            single += 1
            hits_single += p
    return hits / total, hits_single / single


def tournament(makers: dict, games: int = 200) -> None:
    print(f"{'':>24} vs {'':<24} {'gana A':>7} {'tablas':>7} {'gana B':>7}")
    for (na, ma), (nb, mb) in itertools.combinations(makers.items(), 2):
        wa, d, wb = duel(ma(), mb(), games)
        print(f"{na:>24} vs {nb:<24} {wa:>7} {d:>7} {wb:>7}")


def loss_vs_perfect(agent) -> float:
    """Probabilidad EXACTA de que `agent` pierda contra un rival perfecto (sin simular partidas).

    El rival juega una jugada óptima elegida al azar entre las óptimas; el agente, su política
    (si desempata al azar, se promedia sobre las jugadas empatadas). Se recorre el árbol con
    memoización y se promedia entre jugar de primero y de segundo. 0.0 = no pierde nunca.
    """
    from .game import O, X, is_full, winner

    def prob_loss(agent_sym: int) -> float:
        memo: dict = {}

        def rec(b: tuple) -> float:
            if b in memo:
                return memo[b]
            w = winner(b)
            if w:
                v = 1.0 if w != agent_sym else 0.0
            elif EMPTY not in b:
                v = 0.0
            else:
                p = to_move(b)
                if p == agent_sym:
                    s = agent.scores(list(b), p)
                    best = np.nanmax(s)
                    moves = [m for m in range(9) if not np.isnan(s[m]) and s[m] >= best - 1e-9]
                else:
                    moves = solver.optimal_moves(b)
                v = sum(rec(b[:m] + (p,) + b[m + 1:]) for m in moves) / len(moves)
            memo[b] = v
            return v

        return rec((EMPTY,) * 9)

    return (prob_loss(X) + prob_loss(O)) / 2
