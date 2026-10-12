import os

import numpy as np
import pytest

from tictactoe import match, solver
from tictactoe.agents import MinimaxAgent, RandomAgent
from tictactoe.agents.neural import MODELS_DIR, NeuralAgent, RLAgent
from tictactoe.game import EMPTY, O, X, to_move
from tictactoe.train_rl import train as train_q

HAS_NN = os.path.exists(os.path.join(MODELS_DIR, "nn.npz"))
HAS_RL = os.path.exists(os.path.join(MODELS_DIR, "rl.npz"))


def test_minimax_coincide_con_el_solver_en_todas_las_posiciones():
    ag = MinimaxAgent(seed=0)
    for b in solver.playable_states():
        s = ag.scores(list(b), to_move(b))
        for m, v in solver.move_outcomes(b).items():
            assert np.sign(s[m]) == v, (b, m)


def test_minimax_deshace_el_tablero_y_la_poda_ahorra_nodos():
    b = [X, EMPTY, EMPTY, EMPTY, O, EMPTY, EMPTY, EMPTY, EMPTY]
    original = list(b)
    pruned, full = MinimaxAgent(alpha_beta=True), MinimaxAgent(alpha_beta=False)
    s1, s2 = pruned.scores(b, X), full.scores(b, X)
    assert b == original
    assert np.array_equal(np.nan_to_num(s1, nan=99), np.nan_to_num(s2, nan=99))   # misma decisión
    assert pruned.nodes < full.nodes / 3          # en el tablero vacío el ahorro es ~15x


def test_minimax_gana_si_puede_y_bloquea_si_debe():
    ag = MinimaxAgent(seed=0)
    assert ag.act([X, X, EMPTY, O, O, EMPTY, EMPTY, EMPTY, EMPTY], X) == 2          # gana
    assert ag.act([X, X, EMPTY, EMPTY, O, EMPTY, EMPTY, EMPTY, EMPTY], O) == 2      # bloquea


def test_minimax_prefiere_ganar_pronto():
    s = MinimaxAgent(seed=0).scores([X, X, EMPTY, O, O, EMPTY, EMPTY, EMPTY, EMPTY], X)
    assert s[2] == 10                                   # victoria inmediata
    assert all(s[m] < 10 for m in (5, 6, 7, 8))         # cualquier otra va más lenta o peor


def test_minimax_nunca_pierde():
    for first in (X, O):
        assert match.play_game(MinimaxAgent(seed=1), MinimaxAgent(seed=2), first=first) == 0
    rnd = RandomAgent(seed=3)
    wa, d, wb = match.duel(MinimaxAgent(seed=4), rnd, games=100)
    assert wb == 0 and wa > 80


@pytest.mark.skipif(not HAS_RL, reason="sin modelo")
def test_refuerzo_juega_perfecto():
    rate, hard = match.optimal_rates(RLAgent(seed=0))
    assert rate > 0.99 and hard > 0.99
    wa, d, wb = match.duel(RLAgent(seed=0), MinimaxAgent(seed=0), games=40)
    assert wb == 0 and wa == 0                       # siempre tablas


@pytest.mark.skipif(not HAS_NN, reason="sin modelo")
def test_red_neuronal_juega_perfecto():
    """El modelo distribuido se eligió por su pérdida exacta contra un rival perfecto: debe ser 0."""
    ag = NeuralAgent()
    assert match.loss_vs_perfect(ag) == 0.0
    rate, hard = match.optimal_rates(ag)
    assert rate > 0.99 and hard > 0.99
    wa, d, wb = match.duel(NeuralAgent(), MinimaxAgent(seed=0), games=40)
    assert wb == 0 and wa == 0                       # siempre tablas


def test_perdida_exacta_contra_perfecto():
    assert match.loss_vs_perfect(MinimaxAgent(seed=0)) == 0.0          # el backtracking no pierde
    assert match.loss_vs_perfect(RandomAgent(seed=0)) > 0.8            # el azar casi siempre pierde


def test_perdida_exacta_coincide_con_partidas_simuladas():
    """Contraste de la métrica exacta con simulación: agente débil (azar) contra minimax."""
    wa, d, wb = match.duel(RandomAgent(seed=5), MinimaxAgent(seed=5), games=300)
    simulated = wb / 300
    assert abs(simulated - match.loss_vs_perfect(RandomAgent(seed=0))) < 0.07


def test_q_learning_aprende_algo_con_pocos_episodios():
    q = train_q(episodes=3000, alpha=0.2, eps_start=0.5, eps_end=0.05, decay_frac=0.8, seed=0)
    assert q.shape == (3 ** 9, 9) and np.abs(q).sum() > 0
    wa, d, wb = match.duel(RLAgent(q, seed=0), RandomAgent(seed=0), games=100)
    assert wa > wb
