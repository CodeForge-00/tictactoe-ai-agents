"""Lógica de una sesión de juego (sin pygame): turnos, roles que se alternan y marcador.

Hay dos "asientos": el 0 (Tú / Jugador 1) y el 1 (la IA o el Jugador 2). En cada
ronda se intercambian los símbolos para que empiece alguien distinto: en las
rondas pares el asiento 0 juega con X (y empieza); en las impares, con O.
"""
from __future__ import annotations

from .agents import Agent, MinimaxAgent
from .game import O, X, Game, legal_moves


class Session:
    def __init__(self, agent: Agent | None):
        self.agent = agent
        self.analyst: Agent = agent or MinimaxAgent()   # en 2 jugadores, el análisis lo da el backtracking
        self.names = ("Tú", agent.name) if agent else ("Jugador 1", "Jugador 2")
        self.score = [0, 0, 0]                          # asiento 0, tablas, asiento 1
        self.round = -1
        self.game = Game()
        self.new_round()

    # ------------------------------------------------------------------ rondas
    def new_round(self) -> None:
        self.round += 1
        self.game.reset()
        if self.agent:
            self.agent.reset()
        self._counted = False

    def symbol(self, seat: int) -> int:
        return X if (seat == 0) == (self.round % 2 == 0) else O

    def seat_of(self, player: int) -> int:
        return 0 if self.symbol(0) == player else 1

    # ------------------------------------------------------------------- turnos
    @property
    def turn_seat(self) -> int:
        return self.seat_of(self.game.player)

    @property
    def ai_to_move(self) -> bool:
        return self.agent is not None and not self.game.over and self.turn_seat == 1

    def click(self, cell: int) -> bool:
        """Jugada humana. Devuelve True si se aplicó."""
        g = self.game
        if g.over or self.ai_to_move or not (0 <= cell < 9) or g.board[cell] != 0:
            return False
        g.move(cell)
        self._tally()
        return True

    def ai_move(self) -> int | None:
        if not self.ai_to_move:
            return None
        g = self.game
        cell = self.agent.act(list(g.board), g.player)
        g.move(cell)
        self._tally()
        return cell

    def _tally(self) -> None:
        g = self.game
        if g.over and not self._counted:
            self._counted = True
            self.score[1 if g.winner == 0 else (0 if self.seat_of(g.winner) == 0 else 2)] += 1

    # ---------------------------------------------------------------- análisis
    def analysis(self):
        """Puntuaciones del analista para quien mueve ahora (None si la partida terminó)."""
        g = self.game
        if g.over:
            return None
        return self.analyst.scores(list(g.board), g.player)

    @property
    def status(self) -> str:
        g = self.game
        if g.over:
            if g.winner == 0:
                return "Empate"
            seat = self.seat_of(g.winner)
            if self.agent:
                return "¡Ganaste!" if seat == 0 else f"Gana {self.names[1]}"
            return f"¡Gana {self.names[seat]}!"
        mark = "X" if g.player == X else "O"
        if self.ai_to_move:
            return f"{self.names[1]} piensa… ({mark})"
        if self.agent and self.turn_seat == 0:
            return f"Tu turno ({mark})"
        return f"Turno de {self.names[self.turn_seat]} ({mark})"

    def free_cells(self) -> list[int]:
        return legal_moves(self.game.board)
