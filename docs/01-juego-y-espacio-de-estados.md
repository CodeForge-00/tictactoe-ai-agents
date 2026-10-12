# 1. El juego y su espacio de estados

Código: `tictactoe/game.py`, `tictactoe/solver.py`.

## 1.1 Reglas y representación

Dos jugadores, X (empieza) y O, colocan fichas alternadamente en un tablero 3×3. Gana quien
completa una fila, columna o diagonal; si se llena sin ganador, hay tablas.

El tablero es una **lista de 9 enteros** (`0` vacío, `1` = X, `2` = O), fila a fila:

```
 0 | 1 | 2
-----------
 3 | 4 | 5          LINES = ((0,1,2), (3,4,5), (6,7,8),     filas
-----------                  (0,3,6), (1,4,7), (2,5,8),     columnas
 6 | 7 | 8                   (0,4,8), (2,4,6))              diagonales
```

`Game.move(i)` coloca la ficha del jugador en turno, detecta la línea ganadora y rechaza
jugadas ilegales (casilla ocupada o partida terminada).

## 1.2 Cuántas posiciones hay (medido, no estimado)

| Magnitud | Valor |
|---|---|
| Secuencias de jugadas sin reglas (`9!`) | 362 880 |
| **Partidas legales** (se detienen al ganar) | **255 168** |
| de ellas: gana X / gana O / tablas | 131 184 / 77 904 / 46 080 |
| **Posiciones alcanzables** (tableros distintos) | **5 478** |
| de ellas terminales / aún por jugar | 958 / **4 520** |
| Clases de simetría (todas / por jugar) | 765 / 627 |

Hay muchísimas menos *posiciones* que *partidas*, porque llegar a la misma posición por órdenes
distintos de jugadas cuenta como una sola posición. Esta diferencia (255 168 → 5 478) es la que
explota la programación dinámica del solver (1.5) y la que hace posible una tabla de valores
completa (documento 4).

Posiciones por número de fichas: `1, 9, 72, 252, 756, 1260, 1520, 1140, 390, 78` (de 0 a 9 fichas).

## 1.3 Simetrías

El cuadrado tiene **8 simetrías** (4 giros × 2 reflejos). Un tablero girado es "el mismo" tablero:
las mismas jugadas son óptimas, transformadas. `game.SYMMETRIES` guarda las 8 permutaciones
(`perm[i]` = casilla de origen de la casilla `i`), generadas por cierre a partir de un giro de 90°
y un reflejo. Se usan para:

- **aumentar datos** de la red neuronal ×8 (documento 3);
- **repartir en train/val/test por clase de simetría** y evitar fugas de información (documento 3);
- contar posiciones esencialmente distintas (`canonical`: la menor de las 8 variantes).

`transform_move(m, perm)` dice dónde cae la jugada `m` tras la transformación; el test
`test_simetrias_consistentes` lo comprueba.

## 1.4 La perspectiva "propias / del rival"

Los agentes con modelo (red neuronal y Q-learning) no ven "X" y "O" sino **"mis fichas" y "las del
rival"**:

- `features(board, player)`: 18 números = 9 casillas propias (1/0) + 9 del rival (1/0).
- `state_index(board, player)`: número en base 3 (`0` vacío, `1` propia, `2` rival), entre 0 y 19 682.

Así **un mismo modelo juega con X y con O** y se aprovecha el doble de experiencia. Es el equivalente
del "truco del espejo" del [Breakout](https://github.com/CodeForge-00/breakout-ai-agents/blob/main/docs/02-del-juego-a-los-agentes.md).
El test `test_perspectiva_independiente_del_simbolo` verifica que intercambiar X↔O y el jugador no
cambia lo que ve el modelo.

## 1.5 El solver exacto (`solver.py`)

El tres en raya es un **juego resuelto**. `outcome(board)` calcula con juego perfecto el resultado
para quien mueve (`+1` gana, `0` tablas, `−1` pierde) con **memoización** (`lru_cache`):

$$V(s) = \max_{a}\, \big(-V(s_a)\big) \qquad V(\text{terminal}) \in \{+1, 0, -1\}$$

La memoización convierte un árbol de 255 168 partidas en un recorrido de unos miles de posiciones:
resolver el juego desde el tablero vacío tarda ≈ 11 ms. **Resultado: con juego perfecto el
tres en raya es un empate**; las 9 aperturas de X dan tablas.

Importante: este solver **no es el agente de backtracking del juego** (documento 2). Se usa para:

1. etiquetar los datos de la red neuronal con las jugadas óptimas,
2. medir la calidad de cualquier agente (`match.optimal_rates`),
3. verificar de forma independiente el backtracking (`test_minimax_coincide_con_el_solver...`).

## 1.6 ¿Por qué es un buen banco de pruebas?

Porque se conoce la respuesta correcta en **todas** las posiciones. Eso permite cosas que en
juegos grandes son imposibles: medir exactamente cuánto se equivoca un modelo, separar
"memorizar" de "generalizar", y comprobar que el aprendizaje por refuerzo redescubre valores
exactos. Siguiente: [2. Backtracking (minimax)](02-backtracking-minimax.md).
