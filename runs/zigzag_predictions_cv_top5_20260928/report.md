# Predicciones de zigzag

Se procesaron las 82 filas de stable_packings_zigzag.csv con src.preprocessing.preprocess. K_true corresponde a tr(K)/3 calculado a partir del tensor K del archivo. Para inferencia se usaron los checkpoints guardados en la suite original de las cinco configuraciones mejor clasificadas por validación cruzada dentro de cada arquitectura; los modelos no se reajustaron con zigzag.

Se ejecutó src.predict.py por separado para LogPowerLaw, RBF4 y RBF5. Cada subdirectorio contiene test_predictions.csv (formato largo), predictions_wide.csv (una fila por checkpoint y cinco columnas K_pred_1 a K_pred_5) y los gráficos de predicción y error. No se calculó ningún promedio de modelos.

zigzag_metrics.csv relaciona cada número de columna con la configuración, su posición en CV y el error observado en zigzag. zigzag_processed.csv conserva los identificadores y valores físicos procesados. Los 15 conjuntos de predicciones tienen 82 valores finitos y positivos.
