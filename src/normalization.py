"""Transformaciones de entradas, ajustadas únicamente con entrenamiento."""

import numpy as np
import pandas as pd


LOG_COLUMNS = ["trace_sigma"]
TARGET_COLUMN = "trace_K"  # El CSV contiene tr(K)/3.
ORIENTATION_VALUES = {"horizontal": 0.0, "vertical": 1.0}


def normalization_features(
    data: pd.DataFrame,
    feature_columns,
    log_columns=None,
) -> pd.DataFrame:
    """Selecciona columnas en orden y aplica logaritmos antes de estandarizar."""
    columns = list(feature_columns)
    logs = [c for c in LOG_COLUMNS if c in columns] if log_columns is None else list(log_columns)
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("Las entradas deben ser una lista no vacía de columnas únicas")
    if not set(logs).issubset(columns):
        raise ValueError("Las columnas logarítmicas deben formar parte de las entradas")
    result = data[columns].copy()
    if "orientation" in columns:
        encoded = result["orientation"].map(ORIENTATION_VALUES)
        if encoded.isna().any():
            unknown = sorted(result.loc[encoded.isna(), "orientation"].astype(str).unique())
            raise ValueError(f"Orientaciones desconocidas: {unknown}")
        result["orientation"] = encoded
    result = result.astype(np.float64)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("La normalización requiere valores finitos y sin ausentes")
    if (result[logs] <= 0).any().any():
        raise ValueError(f"El logaritmo requiere valores positivos en {logs}")
    result[logs] = np.log(result[logs])
    return result


class Normalizer:
    """Parámetros compartidos por entrenamiento, validación e inferencia."""

    def __init__(self, training_data: pd.DataFrame, feature_columns, log_columns=None):
        if training_data.empty:
            raise ValueError("El conjunto de entrenamiento no puede estar vacío")
        self.columns = tuple(feature_columns)
        self.log_columns = tuple(
            c for c in LOG_COLUMNS if c in self.columns
        ) if log_columns is None else tuple(log_columns)
        features = normalization_features(training_data, self.columns, self.log_columns)
        self.mean = features.mean()
        self.scale = features.std(ddof=0).replace(0, 1)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        features = normalization_features(data, self.columns, self.log_columns)
        return (features - self.mean) / self.scale
