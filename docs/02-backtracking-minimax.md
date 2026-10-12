# 2. Backtracking: minimax con deshacer y poda alfa-beta

Código: `tictactoe/agents/minimax.py`. Compáralo con
[el backtracking de Breakout](https://github.com/CodeForge-00/breakout-ai-agents/blob/main/docs/03-backtracking.md):
mismo esqueleto (aplicar → recursión → deshacer → podar), pero aquí **sí tiene sentido llevarlo
hasta el final del árbol**.

## 2.1 De backtracking a minimax

Backtracking explora un árbol de decisiones: **elige** una opción, **la aplica**, **recurre** y
**la deshace** al volver. En un solo agente basta con maximizar. En un juego de **dos jugadores
de suma cero**, el rival también elige, y asumimos que lo hará lo mejor posible *para él*:

- en mis turnos **maximizo** el valor,
- en los del rival **lo minimizo** (el rival maximiza el suyo, que es lo opuesto al mío).

Eso es **minimax**. La variante **negamax** lo escribe con un solo caso, usando que el valor
para quien mueve es el opuesto al valor del rival tras la jugada:

$$V(s) = \max_{a}\, \big(-V(s_a)\big)$$

```python
def _negamax(self, b, p, depth, alpha, beta):
    best = -INF
    for m in range(9):
        if b[m] != EMPTY: continue
        b[m] = p                                   # APLICAR
        if winner(b):        v = 10 - depth        # ganar pronto es mejor
        elif EMPTY not in b: v = 0                 # tablas
        else:                v = -self._negamax(b, other(p), depth + 1, -beta, -alpha)
        b[m] = EMPTY                               # DESHACER
        best = max(best, v)
        alpha = max(alpha, best)
        if alpha >= beta: break                    # PODA
    return best
```

Se muta **un único tablero** in situ y se restaura al volver: no se copia el tablero en cada
nodo. `test_minimax_deshace_el_tablero...` comprueba que queda idéntico.

## 2.2 Valoración con profundidad

Valorar solo `+1/0/−1` haría que el agente tratara igual "ganar ahora" y "ganar dentro de 5
jugadas" (o perder ahora y perder tarde). Se usa `10 − profundidad` para ganar y su opuesto para
perder: **gana lo antes posible y, si va a perder, aguanta lo máximo**. Con ese margen, es la
profundidad la que desempata. Entre jugadas igual de buenas se elige **al azar** (con semilla),
para que las partidas no sean siempre idénticas.

## 2.3 Poda alfa-beta

Se mantienen dos cotas: `alpha` (lo mejor que *yo* puedo garantizarme) y `beta` (lo mejor que
*el rival* me permitirá). Si en algún nodo `alpha ≥ beta`, el rival nunca dejará que se llegue
a esa rama (ya tiene una alternativa mejor para él), así que **se descartan las jugadas
restantes sin evaluarlas**. La poda **no cambia el resultado** (solo ahorra trabajo): el test
compara las puntuaciones con y sin poda y son idénticas.

Al evaluar las 9 aperturas del tablero vacío (con ventana completa por jugada):

| | llamadas a `_negamax` | tiempo |
|---|---|---|
| sin poda | 294 777 | ≈ 460 ms |
| **con poda** | **20 089** | **≈ 30 ms** |

≈ 15× menos nodos. El ahorro depende de la posición (en el test, con 2 fichas colocadas, es de
≈ 4×) y del **orden** en que se prueban las jugadas: aquí es el orden de casillas 0-8, sin
ordenación heurística, así que hay margen de mejora (probar antes el centro y las esquinas
podría podar más).

## 2.3b Memoización frente a poda (dos ideas distintas)

- La **poda** evita explorar ramas que no pueden mejorar el resultado.
- La **memoización** evita **recalcular posiciones repetidas** (se llega a la misma por
  órdenes distintos). `solver.py` la usa: 5 478 posiciones en lugar de 255 168 partidas.

Se pueden combinar (tabla de transposición). El agente del juego usa solo poda para que el código
sea la versión más limpia de "aplicar / recursión / deshacer"; el solver, solo memoización.

## 2.4 ¿Tiene sentido backtracking aquí? Análisis

**Sí, y es el caso ideal.**

- El juego es **determinista**, de **información perfecta** y con un árbol **pequeño**: 4 520
  posiciones por jugar. La búsqueda exhaustiva es viable (≈ 30 ms en el peor caso).
- Por eso el agente juega **de forma perfecta**: acierta la jugada óptima en el **100 %** de las
  4 520 posiciones (verificado contra el solver, `test_minimax_coincide_con_el_solver...`) y **no
  pierde nunca**; con él solo puedes empatar.
- A diferencia de Breakout, no hay que limitar la profundidad ni modelar al rival: el rival se
  modela como "el mejor posible", que es justo lo correcto para jugar con seguridad.

**Límites:** no aprende nada, y su coste crece exponencialmente con el tamaño del juego (en
Connect 4 ya hay que limitar la profundidad; en ajedrez o Go es inviable sin heurísticas o
redes). Aquí es una "solución", no una "técnica de IA" que generalice.

## 2.5 Relación con tu código anterior (`src/player/MinMaxPlayer.py`)

El `MinMaxPlayer` original ya implementa la idea de minimax, con un criterio de desempate por
profundidad. Esta versión cambia tres cosas: (1) **negamax** en un solo caso en vez de dos ramas;
(2) **deshacer** sobre una lista mutable en vez de crear un tablero nuevo (cadena) en cada nodo;
(3) **poda alfa-beta**. El código original se conserva intacto en el repositorio.

## 2.6 Ejercicios

1. Ordena las jugadas (centro, esquinas, bordes) y mide cuántos nodos ahorras con poda.
2. Añade memoización a `_negamax` (diccionario por posición) y compara con la poda sola.
3. Limita la profundidad a 2, 3, 4… con una heurística de hoja: ¿cuándo deja de ser perfecto?
4. Implementa un nivel "fácil": con probabilidad `p` elige una jugada al azar.

Siguiente: [3. Red neuronal](03-red-neuronal.md).
