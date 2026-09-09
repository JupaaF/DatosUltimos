"""Particiones sobre valores físicos, antes de ajustar el Normalizer."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DataSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def random_split(data: pd.DataFrame, seed: int = 42) -> DataSplit:
    """70/15/15 por filas; puede repartir una misma serie entre particiones."""
    if len(data) < 7:
        raise ValueError("Se requieren al menos 7 filas para el reparto 70/15/15")
    indices = np.random.default_rng(seed).permutation(len(data))
    train_end = int(0.70 * len(data))
    val_end = train_end + int(0.15 * len(data))
    return DataSplit(*(data.iloc[idx].copy() for idx in (
        indices[:train_end], indices[train_end:val_end], indices[val_end:],
    )))


def horizontal_series_split(
    data: pd.DataFrame,
    test_series: str,
    validation_series: str | None = None,
    validation_fraction: float = 0.15,
    seed: int = 42,
) -> DataSplit:
    """Reserva series horizontales completas por su series_id.

    Sin validation_series, reserva aleatoriamente validation_fraction de las
    filas restantes para validación (no garantiza separación de sus series).
    Con validation_series, mantiene también esa serie íntegra y usa el resto
    para entrenamiento. Los índices y metadatos originales se conservan.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction debe estar entre 0 y 1")
    horizontal = data["orientation"].eq("horizontal")
    available = set(data.loc[horizontal, "series_id"])
    for series in (test_series, validation_series):
        if series is not None and series not in available:
            raise ValueError(f"Serie horizontal desconocida: {series}. Disponibles: {sorted(available)}")
    if test_series == validation_series:
        raise ValueError("Test y validación deben usar series diferentes")
    test_mask = horizontal & data["series_id"].eq(test_series)
    test = data.loc[test_mask].copy()
    remaining = data.loc[~test_mask]
    if validation_series is not None:
        val_mask = remaining["orientation"].eq("horizontal") & remaining["series_id"].eq(validation_series)
        train = remaining.loc[~val_mask].copy()
        validation = remaining.loc[val_mask].copy()
    else:
        indices = np.random.default_rng(seed).permutation(len(remaining))
        n_val = max(1, int(len(remaining) * validation_fraction))
        validation = remaining.iloc[indices[:n_val]].copy()
        train = remaining.iloc[indices[n_val:]].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("Las tres particiones deben contener datos")
    return DataSplit(train, validation, test)
