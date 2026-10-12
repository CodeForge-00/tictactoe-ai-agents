# 4. Aprendizaje por refuerzo: Q-learning con auto-juego

Código: `tictactoe/train_rl.py`, `tictactoe/agents/neural.py` (`RLAgent`).

La teoría general (retorno, función Q, ecuación de Bellman, ε-greedy, auto-juego, recompensas) está
en el [documento 5 de Breakout](https://github.com/CodeForge-00/breakout-ai-agents/blob/main/docs/05-aprendizaje-por-refuerzo.md).
Aquí se usa la versión **tabular** (sin red neuronal), y se explica lo propio de un juego de dos
jugadores de suma cero.

## 4.1 Por qué una tabla y no un DQN

El tres en raya solo tiene **5 478 posiciones** (y 19 683 índices en base 3). La función Q cabe en una
tabla de `19 683 × 9` valores (≈ 0.7 MB; ≈ 55 KB comprimida). Con tabla no hay aproximación de
función, ni *replay*, ni red objetivo: el algoritmo converge de forma estable y se ve con claridad qué
aprende. En Breakout (7 números continuos) la tabla era imposible y hizo falta un DQN.

## 4.2 Formulación

| Elemento | Valor |
|---|---|
| Estado | tablero visto por quien mueve (`state_index`, perspectiva propias/rival) |
| Acciones | las casillas libres |
| Recompensa | **solo al final**: `+1` ganar, `0` tablas; perder es el `+1` del rival |
| Descuento | `γ = 1` (el episodio es corto y siempre termina) |

No hay recompensa intermedia ni *reward shaping*: el agente descubre qué jugadas llevan a ganar solo
por el resultado final.

## 4.3 La actualización negamax

Un juego de dos jugadores tiene un detalle: tras mi jugada **mueve el rival**. El siguiente estado
`s'` lo ve el rival, y su valor `max_a' Q(s', a')` está medido **para él**. Como el juego es de
suma cero, **para mí vale su opuesto**:

$$Q(s,a) \leftarrow Q(s,a) + \alpha\Big(\underbrace{-\gamma \max_{a'} Q(s',a')}_{\text{objetivo}} - Q(s,a)\Big)$$

y si mi jugada termina la partida el objetivo es directamente `+1` (gano) o `0` (tablas). Es el
equivalente, en aprendizaje, del **negamax** del [documento 2](02-backtracking-minimax.md): el mismo
cambio de signo.

```python
done = bool(winner(board)) or is_full(board)
if done:  target = 1.0 if winner(board) else 0.0
else:     target = -max(q[s2][m] for m in legal_moves(board))       # negamax
q[s, a] += alpha * (target - q[s, a])
```

## 4.4 Auto-juego

Un único agente juega **ambos lados** y actualiza **la misma tabla** (por la perspectiva
propias/rival, lo aprendido jugando con X sirve para O). El rival mejora al mismo ritmo que el
aprendiz: se enfrenta siempre a alguien de su nivel.

**Exploración**: ε-greedy con ε decreciente de 0.5 a 0.05 durante el primer 80 % del entrenamiento.
Sin exploración, nunca se visitarían posiciones raras (jugadas malas del rival) y la tabla quedaría
sin valores justo donde más se necesita.

**¿Por qué converge a juego perfecto?** En un juego determinista y finito con recompensa terminal, el
valor de una jugada es el resultado de jugar bien a partir de ahí; la actualización negamax es la
ecuación de Bellman de un juego de suma cero y su punto fijo es el valor de minimax. Con tabla y
suficiente exploración, `Q` se acerca a ese punto fijo.

## 4.5 Hiperparámetros y entrenamiento

| Parámetro | Valor |
|---|---|
| Episodios | 200 000 (≈ 30 s) |
| `α` (tasa de aprendizaje) | 0.2 |
| `ε` | 0.5 → 0.05 (primer 80 %) |
| `γ` | 1 |
| Q inicial | 0 |

`python -m tictactoe.train_rl`. Cada 10 % del entrenamiento imprime partidas contra el backtracking
y contra el azar (ganadas/tablas/perdidas). En una corrida corta de prueba de 50 000 episodios el
agente ya empataba **100 de 100** partidas contra el backtracking.

## 4.6 Resultados y verificación

- **Jugada óptima en las 4 520 posiciones: 100 %** (100 % también en las 2 406 de jugada única).
- **Nunca pierde**: 400 empates de 400 contra el backtracking; contra el azar gana 356 y empata 44.
- Estados con algún valor distinto de cero aprendido: 4 279 de 4 520 (el resto son posiciones donde
  todas las jugadas valen 0 o que casi nunca se visitan).

**Los valores aprendidos son los valores exactos del juego.** Comparando la tabla con el solver:
en la posición con X en una esquina y O a mover, la tabla da `−1.0` para todas las casillas excepto
`0.0` en el centro, exactamente la verdad (solo el centro empata). En posiciones donde todo empata,
`Q = 0` en todas las jugadas libres. Es decir, el agente no solo "juega bien": ha reconstruido
la función de valor de minimax **solo a partir del resultado de las partidas**.

## 4.7 Comparación con los otros dos agentes

| | Backtracking | Red neuronal | Q-learning |
|---|---|---|---|
| Necesita | reglas del juego | etiquetas (solver) | solo el resultado |
| Aprende | no | sí (supervisado) | sí (por refuerzo) |
| Jugada óptima | 100 % | 100 % | **100 %** |
| Pierde contra perfecto | no | no | no |
| Coste de decidir | ≈ 30 ms (peor caso) | decenas de µs | µs (consulta de tabla) |
| Escala a juegos grandes | no | sí | no (tabla) → DQN |

La lección: aquí **tabla + refuerzo igualan al algoritmo exacto** sin conocer el árbol; pero solo
porque el espacio es diminuto. Para algo mayor haría falta aproximar Q con una red (como en el DQN
de Breakout), y entonces reaparecen la inestabilidad y los problemas de selección de modelo.

## 4.8 Ejercicios

1. Pon `--eps-end 0` y `--eps-start 0.05`. ¿Cuántos episodios necesita para dejar de perder?
2. Entrena contra un rival **aleatorio fijo** en vez de auto-juego. ¿Aprende a ganar al azar pero
   falla contra el backtracking?
3. Inicializa `Q` con valores optimistas (`+1`). ¿Cambia la velocidad de convergencia?
4. Sustituye la tabla por la red `18 → 64 → 64 → 9` y aplica el mismo objetivo negamax (un DQN).
5. Mide cuántos episodios hacen falta para alcanzar el 100 % de jugada óptima con 3 semillas.

Siguiente: [5. Evaluación y glosario](05-evaluacion-y-glosario.md).
