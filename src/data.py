from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

if __package__:
    from .splitting import mixed_series_split
else:
    from splitting import mixed_series_split


FEATURE_COLUMNS = [
    "density", "MCN", "trace_sigma", "q_over_p", "a",
]
LOG_COLUMNS = ["trace_sigma"]
TARGET_COLUMN = "trace_K"  # El CSV contiene tr(K)/3.


def load_data(filename_horizontal: Path, filename_vertical: Path) -> pd.DataFrame:
    """Carga los valores físicos para dividirlos antes de normalizar."""
    return pd.concat(
        [pd.read_csv(filename_horizontal), pd.read_csv(filename_vertical)],
        ignore_index=True,
    )


def normalization_features(data: pd.DataFrame) -> pd.DataFrame:
    """Excluye trace_Fabric y aplica logaritmos antes de estandarizar."""
    result = data[FEATURE_COLUMNS].astype(np.float64).copy()
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("La normalización requiere valores finitos y sin ausentes")
    if (result[LOG_COLUMNS] <= 0).any().any():
        raise ValueError(f"El logaritmo requiere valores positivos en {LOG_COLUMNS}")
    result[LOG_COLUMNS] = np.log(result[LOG_COLUMNS])
    return result


class Normalizer:
    """Parámetros compartidos por entrenamiento, validación e inferencia."""

    def __init__(self, training_data: pd.DataFrame):
        if training_data.empty:
            raise ValueError("El conjunto de entrenamiento no puede estar vacío")
        features = normalization_features(training_data)
        self.mean = features.mean()
        self.scale = features.std(ddof=0).replace(0, 1)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return (normalization_features(data) - self.mean) / self.scale


class CustomDataset(Dataset):
    """Recibe una partición y un normalizador ajustado solo con entrenamiento."""

    def __init__(self, data: pd.DataFrame, normalizer: Normalizer):
        self.columns = FEATURE_COLUMNS.copy()
        self.normalizer = normalizer
        self.orientations = data["orientation"].to_numpy(copy=True)
        self.data = torch.from_numpy(
            normalizer.transform(data).to_numpy(dtype=np.float32, copy=True)
        )
        self.targets = torch.from_numpy(data[TARGET_COLUMN].to_numpy(dtype=np.float32, copy=True))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class BalancedOrientationSampler(Sampler[int]):
    """Igual número por orientación en cada época, sin reemplazo.

    Usar solo con entrenamiento. La mayoría se submuestrea de nuevo en cada
    iteración; el equilibrio es por época, no necesariamente por minibatch.
    """

    def __init__(self, dataset: CustomDataset, seed: int = 42):
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


def main() -> None:
    dataset_dir = Path(__file__).resolve().parents[1] / "dataset" / "processed"
    data = load_data(
        dataset_dir / "stable_packings_horizontal.csv",
        dataset_dir / "stable_packings_vertical.csv",
    )
    split = mixed_series_split(data)
    normalizer = Normalizer(split.train)
    for name, partition in (("train", split.train), ("validation", split.validation), ("test", split.test)):
        dataset = CustomDataset(partition, normalizer)
        print(f"{name}: {len(dataset)} filas, {len(dataset.columns)} columnas")
        if name == "train":
            loader = DataLoader(dataset, batch_size=16,
                                sampler=BalancedOrientationSampler(dataset))
            print(f"Entrenamiento equilibrado: {len(loader.sampler)} muestras por época")


if __name__ == "__main__":
    main()
