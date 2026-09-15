"""Compara en test las predicciones de uno o varios modelos entrenados.

Ejemplo:
    python -m src.predict --models runs/estudio/modelo_1 runs/estudio/modelo_2
"""

import argparse
from datetime import datetime
from pathlib import Path
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .models import (
    ConditionalLogPowerLawMLP, LogPowerLawMLP, MLP,
)
from .normalization import TARGET_COLUMN, normalization_features


ROOT = Path(__file__).resolve().parents[1]
MODEL_CLASSES = {
    "mlp": MLP,
    "log-power-law": LogPowerLawMLP,
    "conditional-log-power-law": ConditionalLogPowerLawMLP,
}


def resolve_run(path: Path) -> tuple[Path, Path]:
    """Devuelve (directorio de ejecución, checkpoint)."""
    path = path.expanduser().resolve()
    if path.is_dir():
        run_dir = path
        checkpoints = [path / "best_model.pt", path / "best_model.pkl"]
        checkpoint = next((candidate for candidate in checkpoints if candidate.is_file()), checkpoints[0])
    else:
        checkpoint = path
        run_dir = path.parent
    if not checkpoint.is_file():
        raise FileNotFoundError(f"No existe el checkpoint: {checkpoint}")
    if not (run_dir / "partitions.csv").is_file():
        raise FileNotFoundError(f"No existe partitions.csv en {run_dir}")
    return run_dir, checkpoint


def load_test_data(path: Path) -> pd.DataFrame:
    """Carga un CSV físico y, si existe, selecciona su partición de test."""
    data = pd.read_csv(path)
    if "partition" in data.columns:
        data = data.loc[data.partition == "test"].copy()
    required = {"row_id", "orientation", "series_id", "trace_sigma", TARGET_COLUMN}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Faltan columnas en el test común: {sorted(missing)}")
    return data


def predict_test(path: Path, device: str = "cpu", test_data: pd.DataFrame | None = None) -> pd.DataFrame:
    """Carga un artefacto de entrenamiento y predice su partición de test."""
    run_dir, checkpoint_path = resolve_run(path)
    if checkpoint_path.suffix == ".pkl":
        if device != "cpu":
            raise ValueError("Los modelos RBF de SciPy solo admiten --device cpu")
        with checkpoint_path.open("rb") as handle:
            checkpoint = pickle.load(handle)
        model = checkpoint["model"]
        model_name = checkpoint["model_name"]
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model_name = checkpoint.get("model_name")
        if model_name not in MODEL_CLASSES:
            raise ValueError(f"Modelo desconocido en {checkpoint_path}: {model_name!r}")
        model = MODEL_CLASSES[model_name](**checkpoint["model_kwargs"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device).eval()

    if test_data is None:
        partitions = pd.read_csv(run_dir / "partitions.csv")
        test = partitions.loc[partitions["partition"] == "test"].copy()
    else:
        test = test_data.copy()
    if test.empty:
        raise ValueError(f"La partición de test está vacía en {run_dir}")

    normalizer = checkpoint["normalizer"]
    columns = list(normalizer["columns"])
    features = normalization_features(test, columns, normalizer["log_columns"])
    values = (features.to_numpy(dtype=float) - np.asarray(normalizer["mean"])) / np.asarray(normalizer["scale"])
    if checkpoint_path.suffix == ".pkl":
        prediction = model.predict(values)
    else:
        inputs = [torch.tensor(values, dtype=torch.float32, device=device)]
        if model_name in {
            "log-power-law", "conditional-log-power-law",
        }:
            if (test["trace_sigma"] <= 0).any():
                raise ValueError("trace_sigma debe ser positiva para log-power-law")
            log_pressure = torch.tensor(
                np.log(test["trace_sigma"].to_numpy()), dtype=torch.float32, device=device
            )
            inputs.insert(0, log_pressure)
        with torch.no_grad():
            prediction = model(*inputs).cpu().double().numpy()
        target_normalizer = checkpoint.get("target_normalizer")
        if target_normalizer is not None:
            prediction = (
                prediction * float(target_normalizer["scale"])
                + float(target_normalizer["mean"])
            )
    target_transform = checkpoint["target_transform"]
    if target_transform == "log":
        prediction = np.exp(prediction)
    elif target_transform != "identity":
        raise ValueError(f"Transformación del objetivo desconocida: {target_transform!r}")

    result = test[["row_id", "orientation", "series_id", "trace_sigma", TARGET_COLUMN]].copy()
    result = result.rename(columns={TARGET_COLUMN: "K_true"})
    result["K_predicted"] = prediction
    return result


def combine_predictions(
    paths: list[Path],
    labels: list[str] | None = None,
    device: str = "cpu",
    test_data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Predice varios modelos y comprueba que usan el mismo conjunto de test."""
    if not paths:
        raise ValueError("Debe seleccionarse al menos un modelo")
    resolved = [resolve_run(path) for path in paths]
    if labels is None:
        labels = [run_dir.name for run_dir, _ in resolved]
    if len(labels) != len(paths):
        raise ValueError("Debe haber una etiqueta por modelo")
    if len(set(labels)) != len(labels):
        raise ValueError("Las etiquetas de los modelos deben ser únicas")

    frames = []
    reference = None
    reference_columns = ["row_id", "orientation", "series_id", "trace_sigma", "K_true"]
    for path, label in zip(paths, labels):
        prediction = predict_test(path, device, test_data)
        current = prediction[reference_columns].reset_index(drop=True)
        if reference is None:
            reference = current
        elif not current.equals(reference):
            raise ValueError("Los modelos seleccionados no comparten la misma partición de test")
        prediction.insert(0, "model", label)
        frames.append(prediction)
    return pd.concat(frames, ignore_index=True)


def load_saved_predictions(
    paths: list[Path], labels: list[str], test_data: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Carga predicciones ya agregadas, por ejemplo ensembles de semillas."""
    if len(paths) != len(labels):
        raise ValueError("Debe haber una etiqueta por archivo de predicciones")
    frames = []
    required = {"row_id", "orientation", "series_id", "K_true", "K_predicted"}
    allowed_ids = None if test_data is None else set(test_data.row_id)
    for path, label in zip(paths, labels):
        frame = pd.read_csv(path)
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Faltan columnas en {path}: {sorted(missing)}")
        if allowed_ids is not None:
            frame = frame.loc[frame.row_id.isin(allowed_ids)].copy()
        frame.insert(0, "model", label)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def validate_common_test(predictions: pd.DataFrame) -> None:
    reference = None
    key_columns = ["row_id", "orientation", "series_id"]
    for _, frame in predictions.groupby("model", sort=False):
        current = frame.sort_values("row_id").reset_index(drop=True)
        if reference is None:
            reference = current
        else:
            same_keys = current[key_columns].equals(reference[key_columns])
            same_target = np.allclose(
                current["K_true"], reference["K_true"], rtol=1e-6, atol=1e-12
            )
            if not same_keys or not same_target:
                raise ValueError("Los modelos seleccionados no comparten el mismo test")


def plot_predictions(predictions: pd.DataFrame, output: Path) -> Path:
    """Genera un panel por serie de test respetando el orden de sus muestras."""
    series = list(predictions.groupby(["orientation", "series_id"], sort=False))
    if not series:
        raise ValueError("No hay predicciones que representar")
    figure, axes = plt.subplots(1, len(series), figsize=(7 * len(series), 5), squeeze=False)
    for axis, ((orientation, series_id), group) in zip(axes.flat, series):
        truth = group.drop_duplicates("row_id").sort_values("row_id")
        sample = np.arange(1, len(truth) + 1)
        axis.plot(sample, truth["K_true"], "o-", color="black", label="K real")
        for model, model_data in group.groupby("model", sort=False):
            model_data = model_data.sort_values("row_id")
            axis.plot(sample, model_data["K_predicted"], "o--", markersize=3, label=model)
        axis.set_yscale("log")
        axis.set_title(f"{orientation.capitalize()} — {series_id}")
        axis.set_xlabel("Muestra de la serie de test")
        axis.set_ylabel("K")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
    figure.suptitle("K real y predicha en las series de test")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_errors(predictions: pd.DataFrame, output: Path, *, relative: bool = False) -> Path:
    """Representa únicamente el error de cada modelo en cada serie de test."""
    series = list(predictions.groupby(["orientation", "series_id"], sort=False))
    if not series:
        raise ValueError("No hay predicciones que representar")
    figure, axes = plt.subplots(1, len(series), figsize=(7 * len(series), 5), squeeze=False)
    for axis, ((orientation, series_id), group) in zip(axes.flat, series):
        for model, model_data in group.groupby("model", sort=False):
            model_data = model_data.sort_values("row_id")
            error = model_data["K_predicted"] - model_data["K_true"]
            if relative:
                error = 100.0 * error.abs() / model_data["K_true"].abs()
            sample = np.arange(1, len(model_data) + 1)
            axis.plot(sample, error, "o-", markersize=3, label=model)
        if not relative:
            axis.axhline(0.0, color="black", linewidth=1, alpha=0.7)
        axis.set_title(f"{orientation.capitalize()} — {series_id}")
        axis.set_xlabel("Muestra de la serie de test")
        axis.set_ylabel("Error relativo absoluto (%)" if relative else "K predicha − K real")
        axis.grid(True, alpha=0.25)
        axis.legend()
    title = "Error relativo absoluto de las predicciones" if relative else "Error firmado de las predicciones"
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="*", type=Path, default=[],
        help="Directorios de ejecución o rutas a best_model.pt",
    )
    parser.add_argument("--labels", nargs="+", help="Etiquetas opcionales, una por modelo")
    parser.add_argument(
        "--prediction-files", nargs="*", type=Path, default=[],
        help="CSV de predicciones ya calculadas, incluidos ensembles",
    )
    parser.add_argument("--prediction-labels", nargs="+", help="Una etiqueta por prediction-file")
    parser.add_argument("--test-data", type=Path, help="CSV físico que define un test común")
    parser.add_argument("--output-dir", type=Path, help="Directorio nuevo para la comparación")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    output = args.output_dir or ROOT / "runs" / f"prediction_comparison_{datetime.now():%Y%m%d-%H%M%S-%f}"
    if output.exists():
        parser.error(f"El directorio de salida ya existe: {output}")
    try:
        if not args.models and not args.prediction_files:
            raise ValueError("Debe seleccionarse al menos un modelo o archivo de predicciones")
        common_test = load_test_data(args.test_data) if args.test_data else None
        frames = []
        if args.models:
            frames.append(combine_predictions(args.models, args.labels, args.device, common_test))
        if args.prediction_files:
            if args.prediction_labels is None:
                raise ValueError("--prediction-labels es obligatorio con --prediction-files")
            frames.append(load_saved_predictions(
                args.prediction_files, args.prediction_labels, common_test
            ))
        predictions = pd.concat(frames, ignore_index=True)
        validate_common_test(predictions)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    output.mkdir(parents=True)
    predictions.to_csv(output / "test_predictions.csv", index=False)
    plot_path = plot_predictions(predictions, output / "test_K_comparison.png")
    error_path = plot_errors(predictions, output / "test_K_error.png")
    relative_error_path = plot_errors(
        predictions, output / "test_K_relative_error.png", relative=True
    )
    print(f"Predicciones: {output / 'test_predictions.csv'}")
    print(f"Gráfico: {plot_path}")
    print(f"Error: {error_path}")
    print(f"Error relativo: {relative_error_path}")
    return output


if __name__ == "__main__":
    main()
