# Documentación

Orden de lectura recomendado. Para la teoría general de redes neuronales y de DQN, estos documentos
enlazan con la [documentación de Breakout](https://github.com/CodeForge-00/breakout-ai-agents/tree/main/docs)
en lugar de repetirla.

| # | Documento | Qué cubre | Código |
|---|---|---|---|
| 1 | [El juego y su espacio de estados](01-juego-y-espacio-de-estados.md) | reglas, 255 168 partidas vs 5 478 posiciones, simetrías, perspectiva, solver exacto | `game.py`, `solver.py` |
| 2 | [Backtracking (minimax)](02-backtracking-minimax.md) | negamax, deshacer, poda alfa-beta, memoización, ¿tiene sentido? | `agents/minimax.py` |
| 3 | [Red neuronal](03-red-neuronal.md) | aprendizaje supervisado, métrica exacta de fuerza, fuga de datos, sobreajuste, la historia de un error de criterio | `train_nn.py`, `nn.py` |
| 4 | [Aprendizaje por refuerzo](04-aprendizaje-por-refuerzo.md) | Q-learning tabular, actualización negamax, auto-juego | `train_rl.py` |
| 5 | [Evaluación y glosario](05-evaluacion-y-glosario.md) | métricas, torneo, reproducir, tests, glosario | `match.py`, `evaluate.py`, `tests/` |

Las fórmulas usan LaTeX (`$...$`); se ven bien en GitHub y en el preview de VS Code.
