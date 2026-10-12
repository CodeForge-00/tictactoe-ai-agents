import pytest

from tictactoe import game, solver
from tictactoe.game import EMPTY, O, X, Game


def test_ganador_en_cada_linea():
    for line in game.LINES:
        b = [EMPTY] * 9
        for i in line:
            b[i] = O
        assert game.winner(b) == O and game.winning_line(b) == line
    assert game.winner([EMPTY] * 9) == 0


def test_turno_segun_fichas():
    assert game.to_move([EMPTY] * 9) == X
    assert game.to_move([X, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY]) == O


def test_partida_y_movimientos_ilegales():
    g = Game()
    g.move(0)
    with pytest.raises(ValueError):
        g.move(0)                      # casilla ocupada
    for m in (3, 1, 4, 2):             # X: 0,1,2  O: 3,4  -> X completa la fila superior
        g.move(m)
    assert g.over and g.winner == X and g.line == (0, 1, 2)
    with pytest.raises(ValueError):
        g.move(5)                      # partida terminada


def test_empate():
    g = Game()
    for m in (0, 1, 2, 4, 3, 5, 7, 6, 8):    # tablero sin líneas
        g.move(m)
    assert g.over and g.winner == 0


def test_espacio_de_estados():
    rs = solver.reachable_states()
    assert len(rs) == 5478
    assert len(solver.playable_states()) == 4520
    assert len({game.canonical(b) for b in rs}) == 765


def test_simetrias_consistentes():
    assert len(game.SYMMETRIES) == 8
    b = [X, O, EMPTY, EMPTY, X, EMPTY, EMPTY, EMPTY, O]
    for perm in game.SYMMETRIES:
        tb = game.transform_board(b, perm)
        for i in range(9):
            assert tb[game.transform_move(i, perm)] == b[i]


def test_indice_de_estado_inyectivo_y_en_rango():
    seen = {}
    for b in solver.playable_states():
        i = game.state_index(b, game.to_move(b))
        assert 0 <= i < 3 ** 9
        assert seen.setdefault(i, b) == b


def test_perspectiva_independiente_del_simbolo():
    """Intercambiar X<->O y el jugador no cambia lo que ve el modelo."""
    b = [X, EMPTY, O, EMPTY, X, EMPTY, EMPTY, EMPTY, EMPTY]
    swapped = [{X: O, O: X}.get(v, v) for v in b]
    assert game.state_index(b, X) == game.state_index(swapped, O)
    assert (game.features(b, X) == game.features(swapped, O)).all()
