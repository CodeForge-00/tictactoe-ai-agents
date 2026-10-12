from tictactoe.agents import MinimaxAgent, RandomAgent
from tictactoe.game import O, X
from tictactoe.session import Session


def test_dos_jugadores_alternan_turnos_y_simbolos():
    s = Session(None)
    assert s.symbol(0) == X and s.turn_seat == 0
    assert s.click(0) and s.turn_seat == 1
    assert not s.click(0)                      # ocupada
    for c in (3, 1, 4, 2):
        s.click(c)
    assert s.game.over and s.score == [1, 0, 0] and "Jugador 1" in s.status
    s.new_round()
    assert s.symbol(0) == O and s.turn_seat == 1       # empieza el otro


def test_la_ia_juega_cuando_le_toca_y_no_deja_mover_al_humano():
    s = Session(RandomAgent(seed=0))
    s.new_round()                                       # ronda 1: la IA es X y empieza
    assert s.ai_to_move and not s.click(0)
    cell = s.ai_move()
    assert cell is not None and s.game.board[cell] == X and not s.ai_to_move
    assert s.ai_move() is None                          # no es su turno


def test_marcador_cuenta_una_sola_vez_y_empate():
    s = Session(MinimaxAgent(seed=0))
    while not s.game.over:
        if s.ai_to_move:
            s.ai_move()
        else:
            s.click(s.free_cells()[0])
    total = sum(s.score)
    s.click(0), s.ai_move()
    assert sum(s.score) == total == 1
    assert s.score[0] == 0                              # al backtracking no se le gana


def test_analisis_disponible_en_dos_jugadores():
    s = Session(None)
    sc = s.analysis()
    assert sc is not None and len(sc) == 9 and not any(v != v for v in sc)
