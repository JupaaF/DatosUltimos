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


def mixed_series_split(data: pd.DataFrame, seed: int = 42) -> DataSplit:
    """Reserva al azar dos series horizontales y una vertical.

    Una horizontal completa va a validación y la otra a test. La vertical
    se reparte aleatoriamente entre ambos, sin compartir filas: la mitad
    (redondeada hacia abajo) va a validación y el resto a test. Todas las
    demás series van a entrenamiento. La semilla reproduce la selección
    y el reparto para los mismos datos y orden de filas.

    Los índices y metadatos originales se conservan. Validación y test
    comparten una serie vertical, aunque sus muestras son distintas.
    """
    if data["series_id"].isna().any():
        raise ValueError("Todas las filas requieren series_id")
    if not data["orientation"].isin(["horizontal", "vertical"]).all():
        raise ValueError("Orientaciones desconocidas")
    horizontal = data["orientation"].eq("horizontal").to_numpy()
    vertical = data["orientation"].eq("vertical").to_numpy()
    horizontal_series = sorted(data.loc[horizontal, "series_id"].unique())
    vertical_series = sorted(data.loc[vertical, "series_id"].unique())
    if len(horizontal_series) < 2 or not vertical_series:
        raise ValueError("Se requieren al menos dos series horizontales y una vertical")
    vertical_counts = data.loc[vertical].groupby("series_id").size()
    if (vertical_counts < 2).any():
        raise ValueError("Cada serie vertical debe tener al menos dos filas para poder dividirla")

    rng = np.random.default_rng(seed)
    validation_series, test_series = rng.choice(horizontal_series, size=2, replace=False)
    shared_series = rng.choice(vertical_series)
    validation_mask = horizontal & data["series_id"].eq(validation_series).to_numpy()
    test_mask = horizontal & data["series_id"].eq(test_series).to_numpy()
    shared_indices = rng.permutation(np.flatnonzero(
        vertical & data["series_id"].eq(shared_series).to_numpy()
    ))
    midpoint = len(shared_indices) // 2
    validation_mask[shared_indices[:midpoint]] = True
    test_mask[shared_indices[midpoint:]] = True
    train = data.iloc[np.flatnonzero(~(validation_mask | test_mask))].copy()
    if train.empty:
        raise ValueError("Deben quedar series para entrenamiento")
    return DataSplit(
        train,
        data.iloc[np.flatnonzero(validation_mask)].copy(),
        data.iloc[np.flatnonzero(test_mask)].copy(),
    )
