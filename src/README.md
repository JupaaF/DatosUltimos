# Datos para cada modelo

`splitting.py` divide los valores físicos antes de cualquier ajuste.
`normalization.py` selecciona y transforma las entradas de la MLP usando
estadísticas de entrenamiento. `datasets.py` define el contrato de cada
modelo y `data.py` construye los loaders compartidos.

| Modelo | Dataset | Muestra |
| --- | --- | --- |
| `MLP` | `MLPDataset` | `(features, K)` |
| `LogPowerLawMLP` | `LogPowerLawDataset` | `(log_p, features, log_K)` |

En ambos datasets,
`features` se estandariza. Por defecto, si incluye `trace_sigma`, se aplica
primero su logaritmo natural. Se puede configurar con `log_columns`.
`orientation` también puede usarse como entrada y se codifica como
`horizontal=0`, `vertical=1` antes del ajuste del normalizador.
La rama física `log_p` y el objetivo `log_K` nunca se estandarizan.
En estos CSV, `K` corresponde a `trace_K = tr(K)/3`.

```python
from src.data import build_dataloaders, load_data
from src.datasets import LogPowerLawDataset
from src.models import LogPowerLawMLP
from src.splitting import mixed_series_split

# horizontal_path y vertical_path apuntan a los CSV procesados.
data = load_data(horizontal_path, vertical_path)
split = mixed_series_split(data, seed=42)
# Selección ilustrativa: reutilizar split al comparar otras combinaciones.
columns = ["density", "MCN", "q_over_p", "a"]
loaders = build_dataloaders(
    split, LogPowerLawDataset, feature_columns=columns, batch_size=16,
)
model = LogPowerLawMLP(input_dim=len(loaders.normalizer.columns))
for log_p, features, log_K in loaders.train:
    prediction = model(log_p, features)
    loss = (prediction - log_K).square().mean()
    # optimizer.zero_grad(), loss.backward(), optimizer.step()
```

El muestreo por orientación se equilibra solo en entrenamiento. Para usar
todas sus filas en cada época, pasar `balance_orientations=False`.
Validación y test siempre recorren todas sus filas sin barajarlas.

Para añadir un modelo con un contrato diferente, añadir una subclase de
`ModelDataset` que reciba `(data, normalizer)` y defina sus objetivos y
`__getitem__`. Pasarla a `build_dataloaders`; no hace falta duplicar la
partición ni el muestreo. Las muestras deben contener tensores que el
`DataLoader` estándar pueda agrupar. Las columnas de entrada se eligen
explícitamente; no incluir el objetivo entre los predictores.

El normalizador devuelto conserva columnas, transformaciones, media y
escala; debe guardarse junto al modelo para reutilizarlo en inferencia.
`train.py` guarda estos datos en `best_model.pt`.

Verificación: `.venv/bin/python -m unittest discover -s tests`.

Las columnas son obligatorias al crear un `Normalizer` o los loaders.
Los módulos usan imports de paquete; ejecutar el ejemplo con
`.venv/bin/python -m src.data` desde la raíz del proyecto.

## Entrenamiento

```bash
.venv/bin/python -m src.train --model log-power-law \
  --features density MCN q_over_p a --epochs 1000 --patience 100

.venv/bin/python -m src.train --model log-power-law \
  --features density MCN q_over_p a --scale-target

.venv/bin/python -m src.train --model mlp \
  --features density MCN trace_sigma q_over_p a

.venv/bin/python -m src.train --model rbf_4 \
  --features trace_sigma density MCN
```

La selección de entradas es obligatoria. Los modelos neuronales usan Adam; la pérdida
es MSE sobre K físico para `mlp` y sobre log natural de K para `log-power-law`.
Estas pérdidas no son directamente comparables entre modelos. Las métricas
finales incluyen RMSE, MAE y R² en unidades físicas de K.

La mejor época se selecciona por el mínimo MSE de validación, se restauran
sus pesos y después se evalúa test. `--patience` controla la parada temprana;
`--min-delta` fija la mejora mínima que reinicia la paciencia. El mínimo real
se guarda aunque la mejora sea inferior a ese umbral.

`rbf_4` y `rbf_5` son interpoladores RBF de SciPy: se ajustan una sola vez
sobre todas las filas de train normalizadas y sobre `ln(K)`, por lo que no
utilizan épocas, batches, optimizador ni parada temprana. Se guardan en
`best_model.pkl`; validación y test conservan el mismo split por series que
los modelos neuronales.

`--split-seed` (42 por defecto) fija las particiones por series independientemente
de `--seed`, que controla inicialización y muestreo. Mantener la partición
en las futuras barridas y seleccionar configuraciones con validación.
Validación y test comparten la serie vertical reservada, pero no filas.
La reproducibilidad numérica depende también del dispositivo y entorno.

Cada ejecución crea un directorio nuevo en `runs/`, o el indicado mediante
`--output-dir`, y guarda:

- `best_model.pt` (red neuronal) o `best_model.pkl` (RBF): modelo, normalización
  y transformación del objetivo.
- `config.json`: argumentos de la ejecución.
- `partitions.csv`: datos físicos con partición e identificador de fila.
- `history.csv`: pérdidas por época (train en modo entrenamiento, con dropout).
- `metrics.json`: mejor época y métricas finales de validación y test.
- `validation_predictions.csv` y `test_predictions.csv`: predicciones y metadatos.

No se sobrescriben directorios existentes. El checkpoint es para inferencia;
no incluye el estado del optimizador para reanudar entrenamiento.
Consultar todas las opciones con `.venv/bin/python -m src.train --help`.

## Comparación de predicciones

`predict.py` acepta una lista de directorios de entrenamiento (o rutas a sus
`best_model.pt`), reconstruye cada modelo y predice las series horizontal y
vertical de test guardadas en `partitions.csv`:

```bash
.venv/bin/python -m src.predict \
  --models runs/estudio/modelo_1 runs/estudio/modelo_2 \
  --labels modelo_1 modelo_2
```

Los modelos deben compartir exactamente la misma partición de test. El comando
genera `test_K_comparison.png`, respetando el orden de las muestras y usando
escala logarítmica para K, y `test_predictions.csv`, que reúne los valores
reales y predichos (incluida `trace_sigma`). También crea `test_K_error.png`
con el error firmado y `test_K_relative_error.png` con el error relativo
absoluto porcentual.
