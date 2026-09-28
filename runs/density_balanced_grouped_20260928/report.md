# Suite equilibrada por densidad

Datos procesados: 200 horizontales, 200 verticales y 209 cíclicos. Las filas verticales se eligieron buscando densidades igualmente espaciadas dentro de cada una de las nueve series, con un cupo casi uniforme por serie.

Partición por series completas (semilla 42): entrenamiento 310 filas (160 horizontales, 150 verticales); validación 45 (20 horizontales, 25 verticales); prueba 254 (20 horizontales, 25 verticales, 209 cíclicas). Ninguna serie cíclica participó en entrenamiento ni validación.

Se evaluaron 63 combinaciones de entradas para cada uno de LogPowerLaw, RBF4 y RBF5. LogPowerLaw usó un máximo de 3000 épocas, paciencia 300 y escalado de ln(K). El criterio de elección fue RMSE de K físico en validación; prueba no intervino en la elección.

| Modelo | Mejor combinación por validación | RMSE validación | RMSE prueba |
| --- | --- | ---: | ---: |
| log-power-law | density,MCN,q_over_p,a | 1.81345471e-05 | 0.000209388015 |
| rbf_4 | trace_sigma,density,MCN,a | 3.92194089e-05 | 0.000166137558 |
| rbf_5 | trace_sigma,density,MCN,a | 6.9673689e-05 | 0.000209063611 |

Ganador: **log-power-law**, entradas **density,MCN,q_over_p,a**; mejor época **2176** de **2476** ejecutadas.

El mejor RBF4 por validación tiene menor RMSE en prueba que el ganador. La prueba está dominada por las 209 filas cíclicas no vistas; por eso conviene leer ambas métricas y el desglose por orientación. No se cambió el ganador usando prueba.

Archivos: summary.csv ordena las 189 configuraciones; winner.json identifica la elegida; best_by_model_test_orientation.csv desglosa prueba; cada subdirectorio incluye configuración, particiones, predicciones, métricas y modelo guardado.
