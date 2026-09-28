# Validación cruzada de las 10 mejores por arquitectura

Se eligieron las diez mejores configuraciones de LogPowerLaw, RBF4 y RBF5 según el RMSE de validación de la suite anterior. Se hicieron cinco pliegues por series completas sobre las 355 filas de desarrollo; las 254 filas de prueba originales, incluidas las 209 cíclicas, quedaron fuera.

Los 30 ajustes compartieron exactamente las mismas asignaciones. Cada una de las 17 series de desarrollo estuvo en un solo pliegue de validación; cada fila tuvo una predicción fuera de muestra. La normalización y el escalado del objetivo se ajustaron solo con el entrenamiento de cada pliegue.

| Arquitectura | Mejor combinación | RMSE conjunto fuera de muestra | Media RMSE por pliegue | Desv. típica entre pliegues |
| --- | --- | ---: | ---: | ---: |
| log-power-law | density,MCN,a | 5.77049707e-05 | 5.30122725e-05 | 1.95905624e-05 |
| rbf_4 | trace_sigma,density,MCN,a | 7.43795333e-05 | 6.75912698e-05 | 2.48466398e-05 |
| rbf_5 | trace_sigma,density,MCN,a | 0.000177692107 | 0.000162294736 | 6.63781301e-05 |

Ganador por RMSE conjunto fuera de muestra: **log-power-law** con **density,MCN,a** (5.77049707e-05).

Las diez candidatas por arquitectura se preseleccionaron usando una validación anterior incluida en estas 355 filas. Por tanto, la validación cruzada evita fuga entre pliegues y reduce la dependencia de un único reparto, pero aún puede reflejar sesgo de selección. Las métricas de prueba no se usaron para esta clasificación.

Archivos: cv_summary.csv contiene las 30 configuraciones ordenadas, cv_winner.json identifica la mejor, selected_configs.csv documenta la preselección y cada subdirectorio guarda métricas, predicciones fuera de muestra y asignación de pliegues.
