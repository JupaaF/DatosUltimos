"""Entrenamiento supervisado: python -m src.train --help."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch

from .data import ModelDataLoaders, build_dataloaders, load_data
from .datasets import LogPowerLawDataset, MLPDataset
from .models import LogPowerLawMLP, MLP
from .normalization import TARGET_COLUMN
from .splitting import mixed_series_split


# Cada registro acopla modelo, contrato de datos y transformación del objetivo.
MODEL_SPECS = {
    "mlp": (MLP, MLPDataset, "identity"),
    "log-power-law": (LogPowerLawMLP, LogPowerLawDataset, "log"),
}
ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TrainConfig:
    epochs: int = 1000
    learning_rate: float = 1e-3
    patience: int = 100
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


def run_epoch(model, loader, device, optimizer=None):
    """MSE ponderado por muestras, en el espacio del objetivo del dataset."""
    training = optimizer is not None
    model.train(training)
    total_loss, count = 0.0, 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            *inputs, target = [tensor.to(device) for tensor in batch]
            prediction = model(*inputs)
            if prediction.shape != target.shape:
                raise ValueError("La predicción y el objetivo deben tener la misma forma")
            loss = (prediction - target).square().mean()
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


def fit(model, loaders: ModelDataLoaders, config: TrainConfig):
    """Entrena y restaura el mínimo MSE de validación; nunca consulta test.

    min_delta controla la paciencia, pero se conserva cualquier nuevo mínimo.
    El llamador debe fijar la semilla antes de construir modelo y loaders.
    """
    config.validate()
    device = torch.device(config.device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history = []
    best_loss, patience_reference = math.inf, math.inf
    stale_epochs, best_epoch = 0, 0
    best_state = None
    for epoch in range(1, config.epochs + 1):
        train_loss = run_epoch(model, loaders.train, device, optimizer)
        validation_loss = run_epoch(model, loaders.validation, device)
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


def evaluate(model, loader, device, target_transform: str):
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_SPECS, default="log-power-law")
    parser.add_argument("--features", nargs="+", required=True, help="Columnas MLP, en orden")
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
    args = parser.parse_args(argv)
    config = TrainConfig(args.epochs, args.learning_rate, args.patience, args.min_delta, args.seed, args.device, args.log_every)
    config.validate()
    if args.batch_size < 1 or not 0 <= args.dropout < 1:
        parser.error("batch-size debe ser positivo y dropout debe estar en [0, 1)")
    if set(args.features) & {TARGET_COLUMN, "orientation", "series_id"}:
        parser.error("Las entradas no pueden contener el objetivo ni metadatos")
    seed_everything(args.seed)
    data = load_data(
        args.data_dir / "stable_packings_horizontal.csv",
        args.data_dir / "stable_packings_vertical.csv",
    )
    split = mixed_series_split(data, seed=args.split_seed)
    model_class, dataset_class, target_transform = MODEL_SPECS[args.model]
    loaders = build_dataloaders(
        split, dataset_class, feature_columns=args.features, log_columns=args.log_columns,
        batch_size=args.batch_size, balance_orientations=not args.no_balance, seed=args.seed,
    )
    model_kwargs = {"input_dim": len(args.features), "dropout": args.dropout}
    model = model_class(**model_kwargs)
    output = args.output_dir or ROOT / "runs" / f"{args.model}_{datetime.now():%Y%m%d-%H%M%S-%f}"
    output.mkdir(parents=True, exist_ok=False)
    partitions = pd.concat([
        partition.assign(partition=name, row_id=partition.index)
        for name, partition in (("train", split.train), ("validation", split.validation), ("test", split.test))
    ])
    partitions.to_csv(output / "partitions.csv", index=False)
    run_config = {**vars(args), "data_dir": str(args.data_dir.resolve()), "output_dir": str(output.resolve())}
    (output / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    history, best_epoch = fit(model, loaders, config)
    history.to_csv(output / "history.csv", index=False)
    normalizer = loaders.normalizer
    torch.save({
        "model_name": args.model, "model_kwargs": model_kwargs,
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "normalizer": {"columns": list(normalizer.columns), "log_columns": list(normalizer.log_columns),
                       "mean": normalizer.mean.tolist(), "scale": normalizer.scale.tolist()},
        "target_column": TARGET_COLUMN, "target_transform": target_transform,
        "best_epoch": best_epoch, "train_config": asdict(config),
    }, output / "best_model.pt")
    metrics = {"best_epoch": best_epoch, "epochs_run": len(history), "target_transform": target_transform}
    for name, loader, partition in (
        ("validation", loaders.validation, split.validation), ("test", loaders.test, split.test),
    ):
        metrics[name], predictions = evaluate(model, loader, torch.device(args.device), target_transform)
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
