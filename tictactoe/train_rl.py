"""Entrena una tabla Q por Q-learning con auto-juego.

    python -m tictactoe.train_rl [--episodes 200000]

* Estado = tablero visto por quien mueve ("propias" / "del rival"): una sola tabla
  sirve para X y para O. 3^9 = 19 683 índices posibles (5 478 alcanzables).
* Recompensa sólo al final: +1 ganar, 0 tablas. Perder es el +1 del rival.
* Actualización NEGAMAX (juego de suma cero): tras mi jugada `a` en `s`, el rival
  queda en `s'` con valor `max Q(s', ·)` PARA ÉL, así que para mí vale su opuesto:

      Q(s, a) ← Q(s, a) + α · ( −γ · max_a' Q(s', a')  −  Q(s, a) )     (γ = 1)

* ε-greedy con ε decreciente; ambos lados juegan con (y actualizan) la misma tabla.
"""
from __future__ import annotations

import argparse
import os
import random
import time

import numpy as np

from . import match
from .agents import MinimaxAgent, RandomAgent
from .agents.neural import MODELS_DIR, RLAgent
from .game import EMPTY, X, is_full, legal_moves, other, state_index, winner


def choose(q: np.ndarray, board: list, player: int, eps: float, rng: random.Random) -> int:
    moves = legal_moves(board)
    if rng.random() < eps:
        return rng.choice(moves)
    row = q[state_index(board, player)]
    best = max(row[m] for m in moves)
    return rng.choice([m for m in moves if row[m] >= best - 1e-9])


def train(episodes: int, alpha: float, eps_start: float, eps_end: float, decay_frac: float,
          seed: int, log=None) -> np.ndarray:
    rng = random.Random(seed)
    q = np.zeros((3 ** 9, 9), dtype=np.float32)
    for ep in range(episodes):
        eps = max(eps_end, eps_start - (eps_start - eps_end) * ep / (decay_frac * episodes))
        board, player = [EMPTY] * 9, X
        while True:
            s = state_index(board, player)
            a = choose(q, board, player, eps, rng)
            board[a] = player
            done = bool(winner(board)) or is_full(board)
            if done:
                target = 1.0 if winner(board) else 0.0
            else:
                s2 = state_index(board, other(player))
                target = -max(q[s2][m] for m in legal_moves(board))     # negamax
            q[s, a] += alpha * (target - q[s, a])
            if done:
                break
            player = other(player)
        if log and (ep + 1) % (episodes // 10) == 0:
            agent = RLAgent(q, seed=0)
            vs_min = match.duel(agent, MinimaxAgent(seed=0), games=100)
            vs_rnd = match.duel(agent, RandomAgent(seed=0), games=200)
            log(f"episodio {ep + 1:>8}  ε={eps:.2f}  vs Backtracking (G/T/P): {vs_min[0]}/{vs_min[1]}/{vs_min[2]}"
                f"  | vs Aleatorio: {vs_rnd[0]}/{vs_rnd[1]}/{vs_rnd[2]}")
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200_000)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--eps-start", type=float, default=0.5)
    ap.add_argument("--eps-end", type=float, default=0.05)
    ap.add_argument("--decay-frac", type=float, default=0.8, help="fracción del entrenamiento en que ε decae")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    q = train(args.episodes, args.alpha, args.eps_start, args.eps_end, args.decay_frac, args.seed,
              log=lambda s: print(s, flush=True))
    visited = int((np.abs(q).sum(axis=1) > 0).sum())
    print(f"entrenamiento: {time.time() - t0:.0f}s | estados con algún valor aprendido: {visited}")
    rate, hard = match.optimal_rates(RLAgent(q))
    print(f"jugada óptima en todas las posiciones: {rate:.1%} | en las de jugada única: {hard:.1%}")
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, "rl.npz")
    np.savez_compressed(path, q=q)
    print("guardado", path, f"({os.path.getsize(path) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
