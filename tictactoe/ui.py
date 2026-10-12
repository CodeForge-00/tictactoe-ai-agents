"""Interfaz pygame: menú de modos y tablero con análisis opcional de la IA."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pygame

from .agents import Agent, MinimaxAgent
from .agents.neural import MODELS_DIR, NeuralAgent, RLAgent
from .game import O, X
from .session import Session

W, H = 540, 740
CELL, BOARD_X, BOARD_Y = 150, 45, 150
BG, FG, DIM = (14, 18, 32), (232, 236, 245), (120, 128, 150)
PANEL, GRID = (28, 36, 64), (60, 72, 110)
C_X, C_O, C_WIN = (86, 204, 242), (255, 138, 101), (129, 199, 132)
C_GOOD, C_MID, C_BAD = (129, 199, 132), (255, 238, 88), (239, 83, 80)
AI_DELAY_MS = 450


@dataclass
class Mode:
    title: str
    blurb: str
    factory: Optional[Callable[[], Agent]] = None
    model_file: Optional[str] = None

    @property
    def available(self) -> bool:
        return self.model_file is None or os.path.exists(os.path.join(MODELS_DIR, self.model_file))


MODES = [
    Mode("2 jugadores", "Dos personas en el mismo equipo."),
    Mode("Contra Backtracking", "Minimax con poda: juega perfecto, sólo puedes empatar.", MinimaxAgent),
    Mode("Contra Red neuronal", "MLP que imita al solver: muy bueno, pero falla a veces.", NeuralAgent, "nn.npz"),
    Mode("Contra Refuerzo", "Q-learning por auto-juego: aprendió solo y juega perfecto.", RLAgent, "rl.npz"),
]


class App:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Tres en raya")
        self.screen = pygame.display.set_mode((W, H))
        self.clock = pygame.time.Clock()
        mk = lambda size, bold=False: pygame.font.SysFont("helveticaneue,arial", size, bold=bold)
        self.f_big, self.f_mid, self.f_small, self.f_tiny = mk(44, True), mk(26), mk(17), mk(14)
        self.sel = 0

    # ------------------------------------------------------------------ texto
    def text(self, s, font, color, center=None, topleft=None):
        img = font.render(s, True, color)
        r = img.get_rect()
        if center:
            r.center = center
        else:
            r.topleft = topleft
        self.screen.blit(img, r)

    # ------------------------------------------------------------------- menú
    def draw_menu(self):
        self.screen.fill(BG)
        self.text("TRES EN RAYA", self.f_big, FG, center=(W // 2, 110))
        self.text("elige cómo jugar", self.f_small, DIM, center=(W // 2, 155))
        for i, m in enumerate(MODES):
            y = 250 + i * 88
            if i == self.sel:
                pygame.draw.rect(self.screen, PANEL, (50, y - 8, W - 100, 72), border_radius=10)
                pygame.draw.rect(self.screen, C_X, (50, y - 8, 4, 72), border_radius=2)
            col = FG if m.available else DIM
            self.text(f"{i + 1}  {m.title}", self.f_mid, col, topleft=(74, y))
            blurb = m.blurb if m.available else f"Falta models/{m.model_file} (ver README)"
            self.text(blurb, self.f_tiny, DIM, topleft=(74, y + 34))
        self.text("Arriba/Abajo elegir · Enter o clic jugar · Esc salir", self.f_tiny, DIM, center=(W // 2, H - 40))

    def menu_event(self, e) -> Optional[Mode]:
        if e.type == pygame.MOUSEMOTION:
            for i in range(len(MODES)):
                if 50 <= e.pos[0] <= W - 50 and 242 + i * 88 <= e.pos[1] <= 314 + i * 88:
                    self.sel = i
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            for i in range(len(MODES)):
                if 50 <= e.pos[0] <= W - 50 and 242 + i * 88 <= e.pos[1] <= 314 + i * 88:
                    self.sel = i
                    return MODES[i] if MODES[i].available else None
        elif e.type == pygame.KEYDOWN:
            if e.key in (pygame.K_UP, pygame.K_w):
                self.sel = (self.sel - 1) % len(MODES)
            elif e.key in (pygame.K_DOWN, pygame.K_s):
                self.sel = (self.sel + 1) % len(MODES)
            elif pygame.K_1 <= e.key < pygame.K_1 + len(MODES):
                self.sel = e.key - pygame.K_1
                return MODES[self.sel] if MODES[self.sel].available else None
            elif e.key in (pygame.K_RETURN, pygame.K_SPACE):
                return MODES[self.sel] if MODES[self.sel].available else None
            elif e.key == pygame.K_ESCAPE:
                raise SystemExit
        return None

    # ----------------------------------------------------------------- tablero
    @staticmethod
    def cell_rect(i: int) -> pygame.Rect:
        r, c = divmod(i, 3)
        return pygame.Rect(BOARD_X + c * CELL, BOARD_Y + r * CELL, CELL, CELL)

    def cell_at(self, pos) -> Optional[int]:
        for i in range(9):
            if self.cell_rect(i).collidepoint(pos):
                return i
        return None

    def draw_mark(self, rect: pygame.Rect, mark: int, color=None):
        pad = 36
        if mark == X:
            c = color or C_X
            pygame.draw.line(self.screen, c, (rect.left + pad, rect.top + pad), (rect.right - pad, rect.bottom - pad), 12)
            pygame.draw.line(self.screen, c, (rect.right - pad, rect.top + pad), (rect.left + pad, rect.bottom - pad), 12)
        else:
            pygame.draw.circle(self.screen, color or C_O, rect.center, CELL // 2 - pad + 4, 11)

    def fmt_score(self, kind: str, v: float):
        if kind == "outcome":
            return ("gana", C_GOOD) if v > 0 else (("tablas", C_MID) if v == 0 else ("pierde", C_BAD))
        if kind == "prob":
            return f"{v:.0%}", (C_GOOD if v >= 0.5 else C_MID if v >= 0.15 else DIM)
        return f"{v:+.2f}", (C_GOOD if v > 0.05 else C_BAD if v < -0.05 else C_MID)

    def draw_session(self, s: Session, show_analysis: bool, hover: Optional[int]):
        self.screen.fill(BG)
        g = s.game
        self.text(s.status, self.f_mid, FG, center=(W // 2, 70))
        sx, so = s.symbol(0), s.symbol(1)
        self.text(f"{s.names[0]}: {'X' if sx == X else 'O'}   ·   {s.names[1]}: {'X' if so == X else 'O'}",
                  self.f_tiny, DIM, center=(W // 2, 105))
        for k in (1, 2):                                   # rejilla
            x = BOARD_X + k * CELL
            y = BOARD_Y + k * CELL
            pygame.draw.line(self.screen, GRID, (x, BOARD_Y + 8), (x, BOARD_Y + 3 * CELL - 8), 5)
            pygame.draw.line(self.screen, GRID, (BOARD_X + 8, y), (BOARD_X + 3 * CELL - 8, y), 5)

        scores = s.analysis() if show_analysis else None
        for i in range(9):
            r = self.cell_rect(i)
            if g.board[i] == 0:
                if hover == i and not g.over and not s.ai_to_move:
                    pygame.draw.rect(self.screen, PANEL, r.inflate(-14, -14), border_radius=10)
                if scores is not None and not np.isnan(scores[i]):
                    label, col = self.fmt_score(s.analyst.score_kind, scores[i])
                    self.text(label, self.f_small, col, center=r.center)
                else:
                    self.text(str(i + 1), self.f_tiny, (50, 58, 84), center=(r.left + 14, r.top + 14))
            else:
                win = g.line is not None and i in g.line
                self.draw_mark(r, g.board[i], C_WIN if win else None)
        if g.line:
            a, b = self.cell_rect(g.line[0]).center, self.cell_rect(g.line[2]).center
            pygame.draw.line(self.screen, C_WIN, a, b, 6)

        y0 = BOARD_Y + 3 * CELL + 30
        self.text(f"{s.names[0]} {s.score[0]}   ·   Tablas {s.score[1]}   ·   {s.names[1]} {s.score[2]}",
                  self.f_small, FG, center=(W // 2, y0))
        if show_analysis:
            self.text(f"Análisis de {s.analyst.name} para quien mueve", self.f_tiny, C_MID, center=(W // 2, y0 + 30))
        hint = "Clic o teclas 1-9 · H análisis · R / Enter nueva partida · Esc menú"
        self.text(hint, self.f_tiny, DIM, center=(W // 2, H - 30))

    # ------------------------------------------------------------------- juego
    def play(self, mode: Mode) -> None:
        s = Session(mode.factory() if mode.factory else None)
        show_analysis = False
        ai_due: Optional[int] = None
        while True:
            now = pygame.time.get_ticks()
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    raise SystemExit
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        return
                    if e.key == pygame.K_h:
                        show_analysis = not show_analysis
                    elif e.key in (pygame.K_r, pygame.K_RETURN) and s.game.over:
                        s.new_round()
                        ai_due = None
                    elif pygame.K_1 <= e.key <= pygame.K_9:
                        s.click(e.key - pygame.K_1)
                elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    cell = self.cell_at(e.pos)
                    if cell is not None:
                        s.click(cell)
                    elif s.game.over:
                        s.new_round()
                        ai_due = None
            if s.ai_to_move:
                if ai_due is None:
                    ai_due = now + AI_DELAY_MS
                elif now >= ai_due:
                    s.ai_move()
                    ai_due = None
            hover = self.cell_at(pygame.mouse.get_pos())
            self.draw_session(s, show_analysis, hover)
            pygame.display.flip()
            self.clock.tick(60)

    def run(self) -> None:
        try:
            while True:
                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        raise SystemExit
                    mode = self.menu_event(e)
                    if mode:
                        self.play(mode)
                self.draw_menu()
                pygame.display.flip()
                self.clock.tick(30)
        except SystemExit:
            pygame.quit()


def main():
    App().run()
