# 5. Evaluación, reproducibilidad y glosario

Código: `tictactoe/match.py`, `tictactoe/evaluate.py`, `tests/`.

## 5.1 Tres formas de medir (y por qué hacen falta)

**A. Calidad de la política** (`match.optimal_rates`): sobre las **4 520 posiciones** alcanzables, qué
fracción de las veces el agente elige una jugada óptima según el solver exacto. Se informa también la
tasa en las **2 406 posiciones de jugada única** (las difíciles).

**B. Probabilidad exacta de perder contra un rival perfecto** (`match.loss_vs_perfect`): se recorre el
árbol de partidas con el rival jugando una jugada óptima al azar. **Es la medida de fuerza real**, sin
ruido de muestreo (documento 3, sección 3.4).

**C. Resultado en partidas** (`match.duel`): `n` partidas alternando quién empieza.

| Agente | A. Jugada óptima | A. (jugada única) | B. P(perder vs perfecto) |
|---|---|---|---|
| Aleatorio | 58.0 % | 38.5 % | 87.3 % |
| Backtracking | **100 %** | **100 %** | **0.0 %** |
| Red neuronal | **100 %** | **100 %** | **0.0 %** |
| Refuerzo (Q-learning) | **100 %** | **100 %** | **0.0 %** |

**La A sola engaña.** La red de la primera versión tenía un 97.7 % en A y, sin embargo, perdía el
28.7 % de las partidas (B) contra un rival perfecto: A promedia todas las posiciones por igual y casi no
ve las pocas posiciones críticas que deciden la partida. Detalle en el
[documento 3](03-red-neuronal.md#38-la-historia-del-error-la-lección-principal).

Validación de las métricas por rutas independientes: el 58.0 % del aleatorio coincide con un cálculo
directo de esperanza; y B coincide con la simulación (28.7 % exacto frente a 30.2 % simulado en 400
partidas, y un test lo contrasta con el azar).

## 5.2 Torneo (400 partidas por cruce)

`python -m tictactoe.evaluate` — columnas: gana A / tablas / gana B.

| A | B | gana A | tablas | gana B |
|---|---|---|---|---|
| Aleatorio | Backtracking | 0 | 49 | 351 |
| Aleatorio | Red neuronal | 0 | 38 | 362 |
| Aleatorio | Refuerzo | 0 | 44 | 356 |
| Backtracking | Red neuronal | 0 | 400 | 0 |
| Backtracking | Refuerzo | 0 | 400 | 0 |
| Red neuronal | Refuerzo | 0 | 400 | 0 |

Lecturas:

- **Los tres agentes son perfectos**: ninguno pierde jamás y empatan entre sí las 400 partidas. Contra
  el azar ganan 351–362 de 400 y empatan el resto (nunca pierden).
- Esto **no era así antes**: con la primera versión, la red perdía 121 de 400 contra el backtracking y
  111 contra el Refuerzo, y 7 contra el azar. El torneo es el que destapó el problema.
- Los empates contra el azar (38–49 de 400) son partidas en las que el azar no cometió un error decisivo.
- Cuando todos son perfectos, el torneo **deja de discriminar**: para comparar agentes fuertes harían
  falta otras métricas (velocidad de decisión, eficiencia de entrenamiento, robustez a semillas).

## 5.3 Reproducir

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q                          # 25 tests
.venv/bin/python -m tictactoe.train_nn                  # ≈ 6 s, escribe models/nn.npz
.venv/bin/python -m tictactoe.train_nn --holdout        # experimento de generalización (no guarda)
.venv/bin/python -m tictactoe.train_rl                  # ≈ 30 s, escribe models/rl.npz
.venv/bin/python -m tictactoe.evaluate                  # métricas y torneo
.venv/bin/python -m tictactoe                           # jugar
```

Los reentrenamientos **sobrescriben** `models/*.npz`. Ambos entrenamientos aceptan `--seed`.
La variabilidad entre semillas se midió para la red neuronal (6 semillas, todas 0 % de derrotas); para el
Q-learning los números son de **un entrenamiento**.

## 5.4 Tests

| Test | Garantiza |
|---|---|
| `test_espacio_de_estados` | 5 478 / 4 520 / 765 posiciones |
| `test_simetrias_consistentes`, `test_perspectiva_...` | simetrías y vista "propias/rival" correctas |
| `test_minimax_coincide_con_el_solver...` | el backtracking es óptimo en las 4 520 posiciones |
| `test_minimax_deshace_el_tablero_y_la_poda...` | el deshacer es correcto; la poda no cambia el resultado |
| `test_minimax_nunca_pierde` | invencible |
| `test_refuerzo_juega_perfecto` | la tabla Q es óptima y empata siempre con el backtracking |
| `test_red_neuronal_juega_perfecto` | el modelo distribuido tiene pérdida exacta 0 y empata siempre con el backtracking |
| `test_perdida_exacta_*` | la métrica exacta da 0 para el backtracking, > 80 % para el azar y coincide con la simulación |
| `test_q_learning_aprende_algo_con_pocos_episodios` | el bucle de Q-learning funciona |
| `tests/test_session.py` | turnos, alternancia de roles, marcador, la IA no deja mover al humano |
| `test_gradientes_coinciden_con_numericos` | el backprop manual es correcto |

## 5.5 Glosario

| Término | Significado |
|---|---|
| **Juego resuelto** | se conoce el resultado con juego perfecto desde cualquier posición |
| **Posición vs partida** | tablero concreto vs secuencia de jugadas que lo produce |
| **Minimax / negamax** | yo maximizo y el rival minimiza (negamax: un solo caso con cambio de signo) |
| **Backtracking** | DFS que aplica una opción, recurre y la deshace |
| **Poda alfa-beta** | descartar ramas que no pueden cambiar el resultado |
| **Memoización** | guardar el resultado de posiciones ya calculadas |
| **Simetría / clase de simetría** | tableros equivalentes por giro o reflejo |
| **Etiqueta blanda** | objetivo repartido entre varias respuestas correctas |
| **Fuga de datos** | información de test que se cuela en el entrenamiento |
| **Sobreajuste** | memorizar el entrenamiento y generalizar peor |
| **Parada temprana** | quedarse con los pesos de la mejor época en validación |
| **L2** | penalización a los pesos grandes |
| **Q-learning tabular** | aprender `Q(s,a)` en una tabla con el objetivo de Bellman |
| **Actualización negamax** | objetivo = opuesto del mejor valor del rival |
| **Auto-juego** | el agente aprende jugando contra sí mismo |
| **ε-greedy** | explorar al azar con probabilidad ε |
