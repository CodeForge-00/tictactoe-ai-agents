"""Evalúa a los agentes: calidad frente al solver exacto y torneo todos contra todos.

    python -m tictactoe.evaluate [--games 400]
"""
from __future__ import annotations

import argparse

from . import match
from .agents import MinimaxAgent, RandomAgent
from .agents.neural import NeuralAgent, RLAgent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=400)
    args = ap.parse_args()
    makers = {
        "Aleatorio": lambda: RandomAgent(seed=1),
        "Backtracking": lambda: MinimaxAgent(seed=1),
        "Red neuronal": lambda: NeuralAgent(),
        "Refuerzo": lambda: RLAgent(seed=1),
    }
    print("Calidad de la política frente al solver exacto (4 520 posiciones alcanzables)")
    print(f"{'agente':>14} {'jugada óptima':>14} {'(jugada única)':>15} {'P(perder vs perfecto)':>22}")
    for name, mk in makers.items():
        rate, hard = match.optimal_rates(mk())
        lose = match.loss_vs_perfect(mk())
        print(f"{name:>14} {rate:>14.1%} {hard:>15.1%} {lose:>22.1%}")
    print(f"\nTorneo ({args.games} partidas por cruce, alternando quién empieza)")
    match.tournament(makers, args.games)


if __name__ == "__main__":
    main()
