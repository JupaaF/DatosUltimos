"""Entrenamiento supervisado: python -m src.train --help."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
import pickle
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch

from .data import ModelDataLoaders, build_dataloaders, load_data
from .datasets import LogPowerLawDataset, MLPDataset
from .models import (
    ConditionalLogPowerLawMLP, LogPowerLawMLP, MLP,
    RBF4, RBF5,
)
from .normalization import Normalizer, TARGET_COLUMN
from .splitting import mixed_series_split


# Cada registro acopla modelo, contrato de datos y transformación del objetivo.
MODEL_SPECS = {
    "mlp": (MLP, MLPDataset, "identity"),
    "log-power-law": (LogPowerLawMLP, LogPowerLawDataset, "log"),
    "conditional-log-power-law": (ConditionalLogPowerLawMLP, LogPowerLawDataset, "log"),
}
RBF_SPECS = {"rbf_4": RBF4, "rbf_5": RBF5}
ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TrainConfig:
    epochs: int = 10000
    learning_rate: float = 1e-3
    patience: int = 1000
    min_delta: float = 0.0
    seed: int = 42
    device: str = "cpu"
    log_every: int = 25

    def validate(self):
        if self.epochs < 1 or self.patience < 1 or self.log_every < 1:
            raise ValueError("epochs, patience y log_every deben ser positivos")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate debe ser finito y positivo")
        if not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("min_delta debe ser finito y no negativo")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model,
    loader,
    device,
    optimizer=None,
    *,
    target_mean=0.0,
    target_scale=1.0,
    physical_target_scale=1.0,
    physical_loss_weight=0.0,
):
    """MSE ponderado por muestras, en el espacio del objetivo del dataset."""
    training = optimizer is not None
    model.train(training)
    total_loss, count = 0.0, 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            *inputs, target = [tensor.to(device) for tensor in batch]
            prediction = model(*inputs)
            scaled_target = (target - target_mean) / target_scale
            if prediction.shape != scaled_target.shape:
                raise ValueError("La predicción y el objetivo deben tener la misma forma")
            log_loss = (prediction - scaled_target).square().mean()
            if physical_loss_weight:
                predicted_log_target = prediction * target_scale + target_mean
                predicted_physical = predicted_log_target.exp()
                physical_target = target.exp()
                physical_loss = (
                    (predicted_physical - physical_target) / physical_target_scale
                ).square().mean()
                loss = log_loss + physical_loss_weight * physical_loss
            else:
                loss = log_loss
            if not torch.isfinite(loss):
                raise ValueError("Pérdida no finita durante entrenamiento o validación")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * target.numel()
            count += target.numel()
    if count == 0:
        raise ValueError("El loader no puede estar vacío")
    return total_loss / count


def fit(
    model,
    loaders: ModelDataLoaders,
    config: TrainConfig,
    *,
    target_mean: float = 0.0,
    target_scale: float = 1.0,
    physical_target_scale: float = 1.0,
    physical_loss_weight: float = 0.0,
):
    """Entrena y restaura el mínimo MSE de validación; nunca consulta test.

    min_delta controla la paciencia, pero se conserva cualquier nuevo mínimo.
    El llamador debe fijar la semilla antes de construir modelo y loaders.
    """
    config.validate()
    if not math.isfinite(target_mean) or not math.isfinite(target_scale) or target_scale <= 0:
        raise ValueError("El escalado del objetivo debe ser finito y tener escala positiva")
    if (
        not math.isfinite(physical_target_scale) or physical_target_scale <= 0
        or not math.isfinite(physical_loss_weight) or physical_loss_weight < 0
    ):
        raise ValueError("La configuración de la pérdida física no es válida")
    device = torch.device(config.device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history = []
    best_loss, patience_reference = math.inf, math.inf
    stale_epochs, best_epoch = 0, 0
    best_state = None
    for epoch in range(1, config.epochs + 1):
        train_loss = run_epoch(
            model, loaders.train, device, optimizer,
            target_mean=target_mean, target_scale=target_scale,
            physical_target_scale=physical_target_scale,
            physical_loss_weight=physical_loss_weight,
        )
        validation_loss = run_epoch(
            model, loaders.validation, device,
            target_mean=target_mean, target_scale=target_scale,
            physical_target_scale=physical_target_scale,
            physical_loss_weight=physical_loss_weight,
        )
        history.append({"epoch": epoch, "train_mse": train_loss, "validation_mse": validation_loss})
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if validation_loss < patience_reference - config.min_delta:
            patience_reference, stale_epochs = validation_loss, 0
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % config.log_every == 0:
            print(f"Época {epoch}: train MSE={train_loss:.6g}, validation MSE={validation_loss:.6g}")
        if stale_epochs >= config.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return pd.DataFrame(history), best_epoch


def evaluate(
    model,
    loader,
    device,
    target_transform: str,
    *,
    target_mean: float = 0.0,
    target_scale: float = 1.0,
):
    """Evalúa todas las muestras del loader y devuelve métricas físicas de K."""
    if target_transform not in ("identity", "log"):
        raise ValueError("Transformación del objetivo desconocida")
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for batch in loader:
            *inputs, target = [tensor.to(device) for tensor in batch]
            predictions.append(model(*inputs).cpu().double())
            targets.append(target.cpu().double())
    prediction, target = torch.cat(predictions), torch.cat(targets)
    prediction = prediction * target_scale + target_mean
    mse = (prediction - target).square().mean().item()
    physical_prediction = prediction.exp() if target_transform == "log" else prediction
    physical_target = target.exp() if target_transform == "log" else target
    if not torch.isfinite(physical_prediction).all() or not math.isfinite(mse):
        raise ValueError("Predicciones no finitas al evaluar")
    error = physical_prediction - physical_target
    denominator = (physical_target - physical_target.mean()).square().sum().item()
    metrics = {
        "n": len(target), "target_mse": mse,
        "K_rmse": error.square().mean().sqrt().item(),
        "K_mae": error.abs().mean().item(),
        "K_r2": 1 - error.square().sum().item() / denominator if denominator > 0 else None,
    }
    return metrics, pd.DataFrame({
        "target": target.numpy(), "prediction": prediction.numpy(),
        "K_true": physical_target.numpy(), "K_predicted": physical_prediction.numpy(),
    })


def rbf_arrays(data: pd.DataFrame, normalizer: Normalizer):
    """Entradas normalizadas y objetivo ln(K) para los modelos RBF."""
    target = data[TARGET_COLUMN].to_numpy(dtype=np.float64)
    if not np.isfinite(target).all() or (target <= 0).any():
        raise ValueError(f"{TARGET_COLUMN} debe contener valores finitos y positivos")
    features = normalizer.transform(data).to_numpy(dtype=np.float64)
    return features, np.log(target)


def evaluate_rbf(model, data: pd.DataFrame, normalizer: Normalizer):
    """Evalúa una RBF cuyo objetivo es el logaritmo natural de K."""
    features, target = rbf_arrays(data, normalizer)
    prediction = model.predict(features)
    if not np.isfinite(prediction).all():
        raise ValueError("Predicciones no finitas al evaluar la RBF")
    physical_target = np.exp(target)
    with np.errstate(over="ignore", invalid="ignore"):
        physical_prediction = np.exp(prediction)
        error = physical_prediction - physical_target
        squared_error = np.square(error)
    physical_metrics_finite = bool(
        np.isfinite(physical_prediction).all() and np.isfinite(squared_error).all()
    )
    denominator = np.square(physical_target - physical_target.mean()).sum()
    metrics = {
        "n": len(target),
        "target_mse": float(np.square(prediction - target).mean()),
        "K_rmse": float(np.sqrt(squared_error.mean())) if physical_metrics_finite else None,
        "K_mae": float(np.abs(error).mean()) if physical_metrics_finite else None,
        "K_r2": (
            float(1 - squared_error.sum() / denominator)
            if physical_metrics_finite and denominator > 0 else None
        ),
        "physical_metrics_finite": physical_metrics_finite,
    }
    predictions = pd.DataFrame({
        "target": target,
        "prediction": prediction,
        "K_true": physical_target,
        "K_predicted": physical_prediction,
    })
    return metrics, predictions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[*MODEL_SPECS, *RBF_SPECS], default="log-power-law")
    parser.add_argument(
        "--features", nargs="+", required=True,
        help="Columnas de entrada, en orden; orientation se codifica como horizontal=0, vertical=1",
    )
    parser.add_argument("--log-columns", nargs="*", default=None, help="Por defecto: trace_sigma si está seleccionada; sin valores: ningún log en features")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset" / "processed")
    parser.add_argument("--output-dir", type=Path, help="Directorio nuevo para esta ejecución")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42, help="Semilla independiente para mantener la partición en las barridas")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--no-balance", action="store_true", help="Usar todas las filas de entrenamiento en cada época")
    parser.add_argument(
        "--scale-target", action="store_true",
        help="Estandarizar ln(K) con estadísticas de train (modelos logarítmicos)",
    )
    parser.add_argument(
        "--physical-loss-weight", type=float, default=0.0,
        help="Peso del MSE adicional sobre K normalizado (modelos logarítmicos)",
    )
    args = parser.parse_args(argv)
    config = TrainConfig(args.epochs, args.learning_rate, args.patience, args.min_delta, args.seed, args.device, args.log_every)
    config.validate()
    if args.batch_size < 1 or not 0 <= args.dropout < 1:
        parser.error("batch-size debe ser positivo y dropout debe estar en [0, 1)")
    if set(args.features) & {TARGET_COLUMN, "series_id"}:
        parser.error("Las entradas no pueden contener el objetivo ni series_id")
    logarithmic_models = {"log-power-law", "conditional-log-power-law"}
    if (args.scale_target or args.physical_loss_weight) and args.model not in logarithmic_models:
        parser.error("El escalado y la pérdida física solo están disponibles para modelos logarítmicos")
    if not math.isfinite(args.physical_loss_weight) or args.physical_loss_weight < 0:
        parser.error("physical-loss-weight debe ser finito y no negativo")
    seed_everything(args.seed)
    data = load_data(
        args.data_dir / "stable_packings_horizontal.csv",
        args.data_dir / "stable_packings_vertical.csv",
    )
    split = mixed_series_split(data, seed=args.split_seed)
    output = args.output_dir or ROOT / "runs" / f"{args.model}_{datetime.now():%Y%m%d-%H%M%S-%f}"
    output.mkdir(parents=True, exist_ok=False)
    partitions = pd.concat([
        partition.assign(partition=name, row_id=partition.index)
        for name, partition in (("train", split.train), ("validation", split.validation), ("test", split.test))
    ])
    partitions.to_csv(output / "partitions.csv", index=False)
    run_config = {**vars(args), "data_dir": str(args.data_dir.resolve()), "output_dir": str(output.resolve())}
    (output / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    if args.model in RBF_SPECS:
        normalizer = Normalizer(split.train, args.features, args.log_columns)
        model = RBF_SPECS[args.model](scale_features=False)
        train_features, train_target = rbf_arrays(split.train, normalizer)
        model.fit(train_features, train_target)
        normalizer_state = {
            "columns": list(normalizer.columns),
            "log_columns": list(normalizer.log_columns),
            "mean": normalizer.mean.tolist(),
            "scale": normalizer.scale.tolist(),
        }
        with (output / "best_model.pkl").open("wb") as handle:
            pickle.dump({
                "model_name": args.model,
                "model": model,
                "normalizer": normalizer_state,
                "target_column": TARGET_COLUMN,
                "target_transform": "log",
                "model_description": model.describe(),
            }, handle)
        pd.DataFrame(columns=["epoch", "train_mse", "validation_mse"]).to_csv(
            output / "history.csv", index=False
        )
        metrics = {
            "best_epoch": None,
            "epochs_run": 0,
            "target_transform": "log",
            "fit": "scipy_rbf_single_fit",
            "model_description": model.describe(),
        }
        for name, partition in (("validation", split.validation), ("test", split.test)):
            metrics[name], predictions = evaluate_rbf(model, partition, normalizer)
            predictions.insert(0, "row_id", partition.index.to_numpy())
            predictions["orientation"] = partition.orientation.to_numpy()
            predictions["series_id"] = partition.series_id.to_numpy()
            predictions.to_csv(output / f"{name}_predictions.csv", index=False)
        (output / "metrics.json").write_text(
            json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8"
        )
        test_rmse = metrics["test"]["K_rmse"]
        rmse_text = f"{test_rmse:.6g}" if test_rmse is not None else "no finito"
        print(f"Ajuste RBF completo. Test RMSE(K): {rmse_text}")
        print(f"Resultados: {output}")
        return output

    model_class, dataset_class, target_transform = MODEL_SPECS[args.model]
    loaders = build_dataloaders(
        split, dataset_class, feature_columns=args.features, log_columns=args.log_columns,
        batch_size=args.batch_size, balance_orientations=not args.no_balance, seed=args.seed,
    )
    model_kwargs = {"input_dim": len(args.features), "dropout": args.dropout}
    if args.model == "conditional-log-power-law":
        log_pressure = np.log(split.train["trace_sigma"].to_numpy(dtype=np.float64))
        model_kwargs.update({
            "pressure_mean": float(log_pressure.mean()),
            "pressure_scale": float(log_pressure.std(ddof=0)),
        })
    model = model_class(**model_kwargs)
    target_normalizer = None
    if args.scale_target:
        log_target = np.log(split.train[TARGET_COLUMN].to_numpy(dtype=np.float64))
        target_mean = float(log_target.mean())
        target_scale = float(log_target.std(ddof=0))
        target_normalizer = {"mean": target_mean, "scale": target_scale}
    else:
        target_mean, target_scale = 0.0, 1.0
    physical_target_scale = float(
        split.train[TARGET_COLUMN].to_numpy(dtype=np.float64).std(ddof=0)
    )
    history, best_epoch = fit(
        model, loaders, config,
        target_mean=target_mean,
        target_scale=target_scale,
        physical_target_scale=physical_target_scale,
        physical_loss_weight=args.physical_loss_weight,
    )
    history.to_csv(output / "history.csv", index=False)
    normalizer = loaders.normalizer
    torch.save({
        "model_name": args.model, "model_kwargs": model_kwargs,
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "normalizer": {"columns": list(normalizer.columns), "log_columns": list(normalizer.log_columns),
                       "mean": normalizer.mean.tolist(), "scale": normalizer.scale.tolist()},
        "target_column": TARGET_COLUMN, "target_transform": target_transform,
        "target_normalizer": target_normalizer,
        "physical_target_scale": physical_target_scale,
        "physical_loss_weight": args.physical_loss_weight,
        "best_epoch": best_epoch, "train_config": asdict(config),
    }, output / "best_model.pt")
    metrics = {"best_epoch": best_epoch, "epochs_run": len(history), "target_transform": target_transform}
    for name, loader, partition in (
        ("validation", loaders.validation, split.validation), ("test", loaders.test, split.test),
    ):
        metrics[name], predictions = evaluate(
            model, loader, torch.device(args.device), target_transform,
            target_mean=target_mean, target_scale=target_scale,
        )
        predictions.insert(0, "row_id", partition.index.to_numpy())
        predictions["orientation"] = partition.orientation.to_numpy()
        predictions["series_id"] = partition.series_id.to_numpy()
        predictions.to_csv(output / f"{name}_predictions.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Mejor época: {best_epoch}. Test RMSE(K): {metrics['test']['K_rmse']:.6g}")
    print(f"Resultados: {output}")
    return output


if __name__ == "__main__":
    main()
