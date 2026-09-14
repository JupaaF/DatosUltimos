from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass

from torch.utils.data import DataLoader, Sampler

from .splitting import DataSplit, mixed_series_split
from . import normalization
from .datasets import ModelDataset, MLPDataset, LogPowerLawDataset


def load_data(filename_horizontal: Path, filename_vertical: Path) -> pd.DataFrame:
    """Carga los valores físicos para dividirlos antes de normalizar."""
    return pd.concat(
        [pd.read_csv(filename_horizontal), pd.read_csv(filename_vertical)],
        ignore_index=True,
    )


class BalancedOrientationSampler(Sampler[int]):
    """Igual número por orientación en cada época, sin reemplazo.

    Usar solo con entrenamiento. La mayoría se submuestrea de nuevo en cada
    iteración; el equilibrio es por época, no necesariamente por minibatch.
    """

    def __init__(self, dataset: ModelDataset, seed: int = 42):
        orientations = dataset.orientations
        if not np.isin(orientations, ["vertical", "horizontal"]).all():
            raise ValueError("Orientaciones desconocidas")
        self.groups = [np.flatnonzero(orientations == name)
                       for name in ("vertical", "horizontal")]
        self.count = min(map(len, self.groups))
        if self.count == 0:
            raise ValueError("El muestreo equilibrado requiere ambas orientaciones en entrenamiento")
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Permite reproducir explícitamente una época."""
        self.epoch = epoch

    def __len__(self):
        return 2 * self.count

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        indices = np.concatenate([
            rng.choice(group, self.count, replace=False) for group in self.groups
        ])
        rng.shuffle(indices)
        self.epoch += 1
        return iter(indices.tolist())


@dataclass
class ModelDataLoaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader
    normalizer: normalization.Normalizer


def build_dataloaders(
    split: DataSplit,
    dataset_class: type[ModelDataset],
    *,
    feature_columns,
    log_columns=None,
    batch_size: int = 16,
    balance_orientations: bool = True,
    seed: int = 42,
) -> ModelDataLoaders:
    """Construye loaders para un contrato de dataset y una selección de entradas.

    Reutilizar el mismo split en las barridas permite comparar las variantes.
    Cada llamada ajusta un normalizador solo con train y lo comparte con
    validación y test. Estos últimos recorren todas sus filas, en orden.
    """
    normalizer = normalization.Normalizer(split.train, feature_columns, log_columns)
    train = dataset_class(split.train, normalizer)
    validation = dataset_class(split.validation, normalizer)
    test = dataset_class(split.test, normalizer)
    sampler = BalancedOrientationSampler(train, seed) if balance_orientations else None
    train_loader = DataLoader(
        train, batch_size=batch_size, sampler=sampler,
        shuffle=sampler is None, generator=torch.Generator().manual_seed(seed),
    )
    return ModelDataLoaders(
        train_loader,
        DataLoader(validation, batch_size=batch_size),
        DataLoader(test, batch_size=batch_size),
        normalizer,
    )


def main() -> None:
    dataset_dir = Path(__file__).resolve().parents[1] / "dataset" / "processed"
    data = load_data(
        dataset_dir / "stable_packings_horizontal.csv",
        dataset_dir / "stable_packings_vertical.csv",
    )
    split = mixed_series_split(data)
    # Columnas ilustrativas para la red logarítmica; la barrida las decidirá.
    for dataset_class, columns in (
        (MLPDataset, ["density", "MCN", "trace_sigma", "q_over_p", "a"]),
        (LogPowerLawDataset, ["density", "MCN", "q_over_p", "a"]),
    ):
        loaders = build_dataloaders(split, dataset_class, feature_columns=columns)
        print(dataset_class.__name__)
        for name in ("train", "validation", "test"):
            loader = getattr(loaders, name)
            print(f"{name}: {len(loader.dataset)} filas, {len(columns)} entradas MLP")
        print(f"Entrenamiento equilibrado: {len(loaders.train.sampler)} muestras por época")


if __name__ == "__main__":
    main()
