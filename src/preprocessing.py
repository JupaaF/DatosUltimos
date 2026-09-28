from pathlib import Path

import numpy as np
import pandas as pd

DATASET_DIR = Path(__file__).resolve().parents[1] / "dataset"
VERTICAL_SAMPLES_PER_SERIES = 15


def series_numbers(data: pd.DataFrame) -> pd.Series:
    """Numera las series según los reinicios de checkpoint."""
    return data["checkpoint"].diff().le(0).cumsum() + 1


def representative_vertical_rows(
    data: pd.DataFrame,
    series: pd.Series,
    samples_per_series: int = VERTICAL_SAMPLES_PER_SERIES,
) -> tuple[pd.DataFrame, pd.Series]:
    """Selecciona posiciones equiespaciadas o todas si la serie es corta."""
    selected = []
    for _, positions in series.groupby(series, sort=False).groups.items():
        positions = np.asarray(list(positions), dtype=int)
        if len(positions) <= samples_per_series:
            selected.extend(positions)
            continue
        offsets = np.rint(
            np.linspace(0, len(positions) - 1, samples_per_series)
        ).astype(int)
        selected.extend(positions[offsets])
    selected = np.asarray(selected, dtype=int)
    return data.loc[selected].copy(), series.loc[selected].copy()


def tensor_invariants(data: pd.DataFrame, prefix: str) -> tuple[pd.Series, pd.Series]:
    """Devuelve tr(T)/3 y sqrt(3 J2), con J2 = tr(dev(T) @ dev(T)) / 2."""
    axes = "xyz"
    trace = sum(data[f"{prefix}_{axis}{axis}"] for axis in axes)
    deviator = {
        (i, j): data[f"{prefix}_{i}{j}"] - (trace / 3 if i == j else 0)
        for i in axes
        for j in axes
    }
    j2 = sum(deviator[i, j] * deviator[j, i] for i in axes for j in axes) / 2
    if (j2 < 0).any():
        raise ValueError(f"J2 negativo en el tensor {prefix}")
    return trace / 3, (3 * j2) ** 0.5


def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    # packing es la fracción de volumen sólido (densidad relativa).
    result = data[["packing", "MCN"]].rename(columns={"packing": "density"}).copy()
    # Las columnas trace_* contienen tr(T)/3; trace_sigma es la presión.
    result["trace_sigma"], result["q"] = tensor_invariants(data, "sigma")
    result["trace_K"], result["kd"] = tensor_invariants(data, "K")
    result["trace_Fabric"], result["a"] = tensor_invariants(data, "Fabric")
    result["q_over_p"] = result["q"] / result["trace_sigma"].where(result["trace_sigma"] != 0)
    return result[[
        "density", "MCN", "trace_sigma", "trace_K", "trace_Fabric",
        "q", "q_over_p", "kd", "a",
    ]]


def main() -> None:
    raw_dir = DATASET_DIR / "raw"
    processed_dir = DATASET_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    for orientation in ("vertical", "horizontal"):
        filename = f"stable_packings_{orientation}.csv"
        raw = pd.read_csv(raw_dir / filename)
        series = series_numbers(raw)
        if orientation == "vertical":
            raw, series = representative_vertical_rows(raw, series)
        data = preprocess(raw)
        # Los CSV originales concatenan series; checkpoint se reinicia en cada una.
        data["orientation"] = orientation
        data["series_id"] = orientation + "_" + series.astype(str)
        data.to_csv(processed_dir / filename, index=False)
        print(f"{filename}: {len(data)} filas, {len(data.columns)} columnas → {processed_dir}")



if __name__ == "__main__":
    main()
