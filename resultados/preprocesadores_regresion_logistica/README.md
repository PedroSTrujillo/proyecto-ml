# Regresión logística con distintos preprocesadores

Fecha: 2026-10-07. Objetivo: ver qué preprocesador distingue mejor cada clase y cuáles conviene combinar en una votación.

## Método

- Datos: el `X_train` del notebook (split 80/20, `random_state=42`, estratificado). El test no se usó.
- Validación cruzada estratificada de 5 folds (`random_state=42`). El preprocesador se ajusta dentro de cada fold.
- `LogisticRegression(max_iter=3000)` con C ∈ {1, 10, 30}. Para cada preprocesador se reporta el mejor C.
- Tipos de reseña, según `extraer_opiniones` de la sección "Análisis de las oraciones":

| Tipo | Reseñas |
|---|---|
| Neutral | 2 861 |
| Una sola polaridad | 2 425 |
| Mixta con cierre decisivo | 1 692 |
| Mixta sin cierre decisivo | 2 387 |
| Sin opinión detectada | 235 |

- Scripts en `scripts/`. `prep.py` carga las funciones del notebook y define los preprocesadores, `evaluar.py` entrena y `analizar.py` calcula las métricas.
- Tablas completas en `tabla_modelos.csv` y `errores_compartidos.csv`.

## Resultados por modelo

"Sep. pos/neg" es la accuracy entre las reseñas que realmente son positivas o negativas, eligiendo solo entre esas dos clases. Las cuatro últimas columnas son la accuracy dentro de cada tipo de reseña.

| Preprocesador | Acc | F1 neg | F1 neu | F1 pos | Sep. pos/neg | Una polaridad | Mixta decisiva | Mixta no decisiva | Neutral |
|---|---|---|---|---|---|---|---|---|---|
| P10 actual + oración decisiva | **0.893** | **0.853** | 0.987 | **0.853** | **0.857** | 0.877 | 0.971 | **0.737** | 0.997 |
| P11 actual + caracteres | 0.889 | 0.846 | **0.991** | 0.845 | 0.848 | **0.880** | 0.966 | 0.716 | **0.999** |
| P9 actual (sin 1ª + última + contraste) | 0.888 | 0.850 | 0.981 | 0.847 | 0.853 | 0.866 | **0.975** | 0.730 | 0.993 |
| P12 caracteres en texto + última + contraste | 0.883 | 0.839 | 0.987 | 0.839 | 0.842 | 0.860 | 0.966 | 0.721 | 0.994 |
| P15 posiciones desde el final + texto | 0.865 | 0.815 | 0.986 | 0.812 | 0.818 | 0.845 | 0.895 | 0.716 | 0.992 |
| P14 posiciones desde el final | 0.865 | 0.816 | 0.984 | 0.812 | 0.819 | 0.838 | 0.907 | 0.721 | 0.988 |
| P6 solo última oración | 0.853 | 0.828 | 0.923 | 0.815 | 0.838 | 0.821 | 0.923 | 0.708 | 0.966 |
| P8 solo oración decisiva | 0.836 | 0.801 | 0.923 | 0.788 | 0.806 | 0.750 | 0.920 | 0.731 | 0.987 |
| P13 posiciones desde el inicio | 0.788 | 0.709 | 0.976 | 0.706 | 0.714 | 0.716 | 0.748 | 0.663 | 0.988 |
| P4 caracteres (2,5) texto completo | 0.739 | 0.633 | 0.986 | 0.632 | 0.637 | 0.734 | 0.581 | 0.543 | 0.996 |
| P5 caracteres (3,6) texto completo | 0.739 | 0.632 | 0.988 | 0.630 | 0.635 | 0.736 | 0.572 | 0.546 | 0.998 |
| P1 palabras (1,2) texto completo | 0.727 | 0.617 | 0.983 | 0.617 | 0.623 | 0.718 | 0.567 | 0.523 | 0.995 |
| P3 palabras (1,2) sin stemming | 0.727 | 0.614 | 0.982 | 0.618 | 0.624 | 0.707 | 0.580 | 0.528 | 0.994 |
| P2 palabras (1,3) texto completo | 0.722 | 0.610 | 0.976 | 0.614 | 0.621 | 0.687 | 0.577 | 0.533 | 0.994 |
| P7 solo después del contraste | 0.522 | 0.505 | 0.560 | 0.457 | 0.660 | 0.183 | 0.973 | 0.044 | 0.976 |

## Conclusiones

1. **La posición importa más que el tipo de n-grama.** Con todo el texto en una sola bolsa de palabras o caracteres (P1–P5), la accuracy queda en 72–74 %, y en las reseñas mixtas los modelos aciertan 52–58 %, prácticamente al azar. Separar la última oración y la cláusula después del contraste en columnas propias (P9) sube a 88.8 %.
2. **Neutral está resuelto** (F1 0.98–0.99 en casi todos). Los n-gramas de caracteres son los mejores en esa clase: P11 llega a un recall de 0.999.
3. **Positivo y negativo se separan mejor con P10.** Su F1 es 0.853 en ambas clases, porque la columna de oración decisiva aporta +0.5 puntos sobre P9. Ningún modelo favorece una de las dos clases: el F1 de negativo y el de positivo siempre quedan a menos de 0.01 de distancia.
4. **Cada modelo es mejor en un tipo de reseña distinto:**
   - Mixta con cierre decisivo: P9 (0.975) y P7, que solo mira la cláusula después del contraste (0.973).
   - Una sola polaridad y sin opinión detectada: P11, gracias a los caracteres.
   - Mixta sin cierre decisivo: P10 (0.737). Este grupo tiene un techo cercano a 0.74 con cualquier modelo.
5. **Errores compartidos:** los modelos que separan posiciones (P9, P10, P11) comparten 52–67 % de sus errores, y P4/P5 y P1/P2/P3 son casi redundantes entre sí (0.72–0.87). Las mayores diferencias aparecen entre modelos de familias distintas, como P10 con P8 (0.38) o P12 con P9 (0.46).

## Votación suave (promedio de probabilidades)

Selección voraz hacia adelante sobre las predicciones fuera de fold:

| Votantes | Accuracy |
|---|---|
| P10 | 0.8931 |
| P10 + P9 | 0.8958 |
| P10 + P9 + P12 | 0.8964 |
| P10 + P9 + P12 + P8 | 0.8989 |
| **P10 + P9 + P12 + P8 + P7** | **0.8998** |

Métricas de la mejor votación:

| Métrica | Valor |
|---|---|
| F1 macro | 0.904 |
| F1 negativo | 0.862 |
| F1 neutral | 0.989 |
| F1 positivo | 0.862 |
| Sep. pos/neg | 0.865 |
| Mixta decisiva | 0.985 |
| Mixta no decisiva | 0.736 |

**Advertencia:** los votantes se eligieron con las mismas predicciones con las que se mide la accuracy, así que la ganancia de +0.7 puntos es optimista. Hay que confirmarla en el test o con validación cruzada anidada.
