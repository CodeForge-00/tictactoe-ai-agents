# 3. Red neuronal: aprender la jugada óptima

Código: `tictactoe/nn.py` (MLP + Adam en numpy), `tictactoe/train_nn.py`,
`tictactoe/agents/neural.py` (`NeuralAgent`), `tictactoe/match.py` (`loss_vs_perfect`).

La teoría general del MLP (neurona, ReLU, softmax, entropía cruzada, retropropagación, inicialización
He, Adam) está explicada **una sola vez** en el
[documento 4 de Breakout](https://github.com/CodeForge-00/breakout-ai-agents/blob/main/docs/04-red-neuronal.md);
aquí está la misma implementación (`nn.py` es una copia) y se cuenta lo que es específico de este
problema.

> **Resumen.** La red distribuida juega perfecto (0 % de derrotas contra un rival perfecto). Llegar ahí
> costó corregir un error mío: la primera versión elegía el modelo con una métrica que **no medía la
> fuerza de juego** y quedaba infrajustada justo en las jugadas que deciden la partida. La historia
> completa (incluida una hipótesis que resultó falsa) está en 3.8: es la parte más instructiva.

## 3.1 Planteamiento

Como el juego está resuelto ([doc 1](01-juego-y-espacio-de-estados.md#15-el-solver-exacto-solverpy)),
se puede **etiquetar cada posición con sus jugadas óptimas** y entrenar un clasificador:

- **Entrada** (18): `features(board, player)` = 9 casillas propias + 9 del rival (perspectiva
  "mis fichas / las del rival", un mismo modelo para X y O).
- **Salida** (9): una puntuación por casilla; se juega la de mayor probabilidad **entre las libres**.
- **Arquitectura distribuida**: `18 → 128 → 128 → 9` (20 105 parámetros). El modo `--holdout` usa
  `18 → 64 → 64 → 9` (5 961).

Es **aprendizaje supervisado** puro, a diferencia del Breakout, donde la red imitaba a un experto
heurístico y hacía falta DAgger. Aquí el "experto" es exacto y todas las posiciones se pueden enumerar.

## 3.2 Decisiones de diseño

**Etiquetas blandas.** En una posición puede haber varias jugadas igual de buenas (de las 4 520,
solo 2 406 tienen una única jugada óptima). El objetivo es una **distribución uniforme entre las
jugadas óptimas**, no un *one-hot*. Consecuencia: la pérdida de entrenamiento **no puede llegar a 0**;
su mínimo es la entropía media de las etiquetas (≈ 0.48). Una pérdida estancada en ~0.5 no es un fallo.

**Enmascarar casillas ocupadas.** Antes del softmax se ponen a `−10⁹` los logits de casillas ocupadas:
la red no gasta capacidad en aprender que no se puede jugar ahí, y el gradiente de esas salidas es 0.

**Aumento con simetrías.** Cada posición se transforma con las 8 simetrías (y sus jugadas óptimas con
ella): ×8 de datos gratis (4 520 posiciones → 34 308 muestras, sin contar duplicados simétricos).
Ojo: esto **no obliga** a la red a ser invariante a las simetrías, solo le da ejemplos de ello (ver 3.7).

## 3.3 Dos preguntas, dos modos

| | `python -m tictactoe.train_nn` | `python -m tictactoe.train_nn --holdout` |
|---|---|---|
| Pregunta | ¿Cuál es el **mejor jugador** que puede ser esta red? | ¿**Generaliza** a posiciones que no ha visto? |
| Datos | **todas** las 4 520 posiciones | 60 % entrenamiento / 20 % validación / 20 % test, por clase de simetría |
| Criterio para elegir el modelo | **pérdida exacta contra un rival perfecto** (3.4) | jugada óptima media en validación |
| Se guarda | `models/nn.npz` (el modelo del juego) | no (salvo `--out`) |

Entrenar con todas las posiciones no es hacer trampa: **todas pueden darse en una partida**, no hay
posiciones "no vistas" en producción. La pregunta de generalización es legítima pero distinta, y por eso
tiene su propio modo.

## 3.4 Medir la fuerza: la probabilidad exacta de perder

Una métrica por posición (¿acertó una jugada óptima?) no mide cuánto se gana o se pierde. La medida que
importa se puede calcular **exacta**, sin simular partidas (`match.loss_vs_perfect`):

- el rival juega una jugada **óptima al azar** (uniforme entre las óptimas);
- el agente juega su política (la red es determinista);
- se recorre el árbol de partidas con memoización y se promedia entre jugar de primero y de segundo.

Es la probabilidad de que el agente **pierda**. Tarda unas décimas de segundo y no tiene ruido de muestreo.

**Contraste con simulación** (la métrica exacta debe coincidir con jugar de verdad):

| Agente | Exacta | Simulada (400 partidas) |
|---|---|---|
| Red de la primera versión | 28.7 % | 30.2 % (121 / 400) |
| Backtracking | 0.0 % | 0 / 400 |
| Aleatorio | 87.3 % | – |

## 3.5 Resultado del modelo distribuido

```
python -m tictactoe.train_nn
posiciones: 4520 (34308 muestras con simetrías) | red 18→128→128→9
época   25  métrica 0.0159        ← la métrica es la probabilidad de perder
época   50  métrica 0.0000
época  100  métrica 0.0000
probabilidad de perder contra un rival perfecto: 0.0%
jugada óptima en todas las posiciones: 100.0%
```

| Medida | Valor |
|---|---|
| Probabilidad de perder contra un rival perfecto | **0.0 %** |
| Jugada óptima (4 520 posiciones) | 100 % |
| Torneo contra Backtracking / Refuerzo (400 partidas cada uno) | 400 tablas, 0 derrotas |

**Robustez entre semillas:** con 6 semillas distintas (0–5) la pérdida es 0.0 % **en las épocas 50, 100,
150 y 200 de todas**. No es suerte de una semilla.

### Exploración de configuraciones (todas las posiciones, midiendo la pérdida exacta)

| Red | `lr` | L2 | Pérdida (épocas 25 / 50 / 100 / 200 / 300) |
|---|---|---|---|
| 64 | 1e-3 | 1e-3 | 27.8 / 11.1 / 5.6 / 0.0 / 5.6 % |
| 64 | 1e-3 | 0 | 20.0 / 5.6 / 7.1 / 0.0 / 0.0 % |
| 128 | 1e-3 | 1e-3 | 27.8 / 5.6 / 7.1 / 1.6 / 1.6 % |
| **128** | **3e-3** | **1e-4** | **1.6 / 0.0 / 0.0 / 0.0 / 0.0 %** |

La métrica es **"afilada"**: cambiar una sola respuesta de apertura la mueve ≈ 5.6 puntos. Por eso en
las configuraciones débiles *oscila* entre 0 % y 5.6 %, y por eso se elige la época por esta métrica y no
se confía en el último valor. Con la configuración elegida se mantiene en 0 % de forma estable.

Elegir entre 4 configuraciones midiendo en el mismo sitio donde se evalúa podría ser sesgo de selección;
aquí **no lo es en el sentido habitual**, porque la métrica se calcula sobre el juego completo y es el
objetivo mismo (no una muestra ruidosa), y se confirmó con 6 semillas.

## 3.6 Experimento de generalización (`--holdout`)

Para saber si la red generaliza se agrupan las posiciones por **clase de simetría** y se reparten las
**clases** (no las posiciones sueltas): si se repartieran sueltas, una versión girada de una posición de
entrenamiento podría caer en validación y la red la "reconocería" sin generalizar (**fuga de datos**).

| Conjunto | Posiciones | Para qué |
|---|---|---|
| entrenamiento | 2 736 (60 %) | ajustar pesos |
| validación | 858 (20 %) | decidir cuándo parar |
| **test** | 926 (20 %) | informar el resultado; no interviene en las decisiones |

### Línea base: ¿cuánto vale un 97 %?

| Política | Jugada óptima | En posiciones de jugada única |
|---|---|---|
| Aleatoria entre libres | 58.0 % | 38.5 % |
| Red **sin entrenar** | 62.9 % | – |
| Red entrenada, **test** | **97.5 %** | – |

El azar ya acierta el 58 % porque muchas posiciones perdonan. El 97 % es una mejora real, pero la
mayor parte del margen está en el 40 % restante.

### Sobreajuste

Red de 64, `lr = 3e-3`, **sin regularizar**: el entrenamiento sube y la validación baja.

| Época | Entrenamiento | Validación |
|---|---|---|
| 1 | 97.8 % | 97.4 % |
| 5 | 99.4 % | 96.4 % |
| 10 | 99.9 % | 95.6 % |
| 40 | 100.0 % | 93.2 % |

La red **memoriza** las posiciones de entrenamiento. Se combate con **parada temprana** y con
**regularización L2** (`1e-3`, solo sobre pesos; estabiliza el entrenamiento).

### Barrido y lo que no se concluyó

Se probaron 24 combinaciones (tamaño 16–128, `lr` 1e-3/3e-3, L2 0/1e-3/1e-2) **midiendo solo en
validación**. La **mejor validación** de cada una quedó entre 95 % y 98 %. Con 858 posiciones el error
estándar es ≈ 0.6 %, así que **casi todas esas diferencias son ruido**. No se eligió la mejor de las 24
(eso sería sobreajustar a la validación). Se fijó un criterio simple de antemano: red de 64, L2 1e-3.

### Resultado del modo `--holdout`

```
posiciones: 4520 | entrenamiento 2736 (20876 con simetrías) | validación 858 | test 926
jugada óptima  validación 97.4% | TEST 97.5% | todas las posiciones 98.1%
probabilidad de perder contra un rival perfecto: 26.6%
```

**Responde a su pregunta: sí generaliza** (97.5 % en test, que no se usó para decidir). Pero **jugar bien
es otra cosa**: pierde el 26.6 % de las partidas contra un rival perfecto.

## 3.7 Dónde falla el modelo `--holdout` (y por qué se le podía ganar)

Los errores no están repartidos: se concentran en las primeras jugadas.

| Fichas en el tablero | Posiciones | Jugadas que empeoran el resultado |
|---|---|---|
| 1 (respuesta a la apertura) | 9 | **4 (44 %)** |
| 2 | 72 | 6 (8.3 %) |
| 3 | 252 | 20 (7.9 %) |
| 4 | 756 | 18 (2.4 %) |
| 5 | 1 140 | 23 (2.0 %) |
| 6 – 8 | 2 290 | 14 (0.6 %) |

Respuesta de la red (con O) a las 9 aperturas de X:

| X abre en | Responde | ¿Correcto? |
|---|---|---|
| esquina 0, 6, 8 | otra esquina | **no** (solo el centro empata; X fuerza la victoria) |
| esquina 2 | el centro | sí |
| centro | un borde | **no** (debía ser una esquina) |
| bordes 1, 3, 5, 7 | centro o esquina | sí |

Dos observaciones:

1. **No es invariante a las simetrías.** Las cuatro esquinas son equivalentes, pero responde bien a
   una y mal a las otras tres. El aumento de datos da *ejemplos* de simetría; no la impone. Una
   arquitectura equivariante, o promediar las 8 transformaciones al decidir, sí lo garantizaría.
2. **Receta histórica para ganarle** (a esta versión): abrir con una esquina (salvo la 2) o con el
   centro. Es determinista, así que se repite. Con el modelo distribuido actual ya no funciona.

## 3.8 La historia del error (la lección principal)

**Qué pasó.** La primera versión distribuida era el modelo de `--holdout`: se elegía la época por
**jugada óptima media en validación**. El usuario la venció y no sabíamos por qué.

**Hipótesis 1 (falsa): "el reparto por clases dejó las aperturas fuera del entrenamiento".** Se
comprobó y no era así: la esquina y el centro **sí** estaban en entrenamiento (solo el borde se reservó
para validación), y la red falla precisamente en esas. Conviene registrar la hipótesis falsa: era
plausible, se verificó antes de afirmarla, y se descartó.

**Hipótesis 2 (confirmada): la parada temprana eligió un modelo sin ajustar.** Con parada en las
primeras épocas, la red **falla las posiciones de apertura de su propio conjunto de entrenamiento**.
Esas pocas posiciones pesan muy poco en la pérdida (3 clases frente a 627) y la red tarda en
ajustarlas. Entrenando más, mejoran (red de 64, solo posiciones de entrenamiento):

| Entrenamiento (sin parada temprana) | Aperturas correctas | Errores en entrenamiento con 1 ficha | P(perder) |
|---|---|---|---|
| 3 épocas | 4/9 | 5/5 | 28.7 % |
| 20 épocas | 4/9 | 5/5 | 29.6 % |
| 60 épocas | 7/9 | 2/5 | 16.3 % |
| 150 épocas | 8/9 | 1/5 | 10.6 % |

**Por qué.** La jugada óptima *media* pondera igual las 4 520 posiciones, la mayoría perdonadoras
(varias jugadas buenas). Casi no se entera de las críticas, que son pocas y de una única jugada válida,
pero son las que **deciden contra un rival perfecto**: una partida son ~4 decisiones propias, y un solo
fallo basta para perder. El 97.7 % parecía "casi perfecto" y la tasa de derrota real era del 29 %.

**Arreglo.** (1) Una métrica que mide lo que importa (3.4); (2) elegir el modelo por ella; (3) entrenar
con todas las posiciones, porque en producción no hay posiciones nuevas; (4) conservar el experimento
de generalización en su propio modo, donde la pregunta es otra.

**Lección general:** el criterio con el que eliges un modelo debe medir lo que te importa. Una métrica
promedio puede esconder los casos raros pero decisivos, y la parada temprana, que parece "siempre buena",
puede elegir justo un modelo infrajustado en ellos.

*Corrección de versiones anteriores de este documento:* atribuían la debilidad de la red a "generalizar
peor"; era en buena parte el criterio de parada.

## 3.9 Ejercicios

1. **Dificultad ajustable**: ahora la red es perfecta. Entrena versiones débiles (`--epochs 4`,
   `--holdout`) y haz que el juego ofrezca niveles. ¿Qué relación hay entre épocas y P(perder)?
2. **Equivarianza**: al decidir, promedia las probabilidades sobre las 8 simetrías (transformando
   la entrada y deshaciendo la transformación en la salida). ¿Desaparece el fallo de las esquinas en
   el modo `--holdout`?
3. **Posiciones críticas**: pondera la pérdida con más peso las posiciones de jugada única en el modo
   `--holdout`. ¿Baja P(perder) sin sobreajustar?
4. Prueba `--hidden 8` y `--hidden 256`: ¿dónde aparecen el subajuste y el sobreajuste?
5. DAgger: haz jugar a la red contra el backtracking y etiqueta las posiciones que visita (como en
   Breakout). ¿Corrige los errores críticos con menos datos?

Siguiente: [4. Aprendizaje por refuerzo](04-aprendizaje-por-refuerzo.md).
