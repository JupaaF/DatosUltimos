"""Un Dataset por contrato de modelo; selección de entradas en el Normalizer.

Para añadir otro modelo, heredar de ModelDataset e implementar __getitem__
con sus entradas y objetivo. Los loaders compartidos aceptan esa nueva clase.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .normalization import Normalizer, TARGET_COLUMN


def physical_values(data: pd.DataFrame, column: str, *, positive: bool = False):
    values = data[column].to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(values).all():
        raise ValueError(f"{column} requiere valores finitos y sin ausentes")
    if positive and (values <= 0).any():
        raise ValueError(f"El logaritmo requiere valores positivos en {column}")
    return values


class ModelDataset(Dataset):
    """Entradas normalizadas y metadatos comunes a los datasets de modelos."""

    def __init__(self, data: pd.DataFrame, normalizer: Normalizer):
        self.columns = list(normalizer.columns)
        self.normalizer = normalizer
        self.orientations = data["orientation"].to_numpy(copy=True)
        self.data = torch.from_numpy(
            normalizer.transform(data).to_numpy(dtype=np.float32, copy=True)
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raise NotImplementedError("Cada dataset define el contrato de su modelo")


class MLPDataset(ModelDataset):
    """MLP clásica: devuelve (features normalizadas, K físico)."""

    def __init__(self, data: pd.DataFrame, normalizer: Normalizer):
        super().__init__(data, normalizer)
        self.targets = torch.from_numpy(physical_values(data, TARGET_COLUMN).astype(np.float32))

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class LogPowerLawDataset(ModelDataset):
    """LogPowerLawMLP: devuelve (log_p, features normalizadas, log_K).

    Los logaritmos son naturales y no se estandarizan. La presión se lee
    siempre del dato físico, independientemente de las columnas de la MLP.
    Si trace_sigma también se selecciona como entrada de la MLP, esa copia
    sigue las transformaciones del Normalizer. K es tr(K)/3 en el CSV.
    """

    def __init__(self, data: pd.DataFrame, normalizer: Normalizer):
        super().__init__(data, normalizer)
        self.log_pressure = torch.from_numpy(
            np.log(physical_values(data, "trace_sigma", positive=True)).astype(np.float32)
        )
        self.targets = torch.from_numpy(
            np.log(physical_values(data, TARGET_COLUMN, positive=True)).astype(np.float32)
        )

    def __getitem__(self, idx):
        return self.log_pressure[idx], self.data[idx], self.targets[idx]

