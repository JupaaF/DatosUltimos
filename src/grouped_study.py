"""Validación cruzada por series y ajuste final del ConditionalLogPowerLaw."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import build_dataloaders, load_data
from .datasets import LogPowerLawDataset
from .models import ConditionalLogPowerLawMLP
from .normalization import TARGET_COLUMN
from .splitting import DataSplit, grouped_series_folds, grouped_series_test_split
from .train import ROOT, TrainConfig, evaluate, fit, run_epoch, seed_everything


def target_statistics(data: pd.DataFrame):
    log_target = np.log(data[TARGET_COLUMN].to_numpy(dtype=np.float64))
    physical = data[TARGET_COLUMN].to_numpy(dtype=np.float64)
    return float(log_target.mean()), float(log_target.std(ddof=0)), float(physical.std(ddof=0))


def model_kwargs(
    data: pd.DataFrame,
    input_dim: int,
    dropout: float,
    hidden_sizes: list[int],
):
    log_pressure = np.log(data["trace_sigma"].to_numpy(dtype=np.float64))
    return {
        "input_dim": input_dim,
        "dropout": dropout,
        "hidden_sizes": hidden_sizes,
        "pressure_mean": float(log_pressure.mean()),
        "pressure_scale": float(log_pressure.std(ddof=0)),
    }


def physical_metrics(predictions: pd.DataFrame):
    error = predictions.K_predicted.to_numpy() - predictions.K_true.to_numpy()
    true = predictions.K_true.to_numpy()
    denominator = np.square(true - true.mean()).sum()
    log_error = predictions.prediction.to_numpy() - predictions.target.to_numpy()
    return {
        "n": len(predictions),
        "target_mse": float(np.square(log_error).mean()),
        "K_rmse": float(np.sqrt(np.square(error).mean())),
        "K_mae": float(np.abs(error).mean()),
        "K_r2": float(1 - np.square(error).sum() / denominator) if denominator > 0 else None,
    }


def common_data(data_dir: Path, split_seed: int):
    data = load_data(
        data_dir / "stable_packings_horizontal.csv",
        data_dir / "stable_packings_vertical.csv",
    )
    return grouped_series_test_split(data, seed=split_seed)


def run_cv(args, config: TrainConfig, output: Path):
    development, test = common_data(args.data_dir, args.split_seed)
    folds = grouped_series_folds(development, n_splits=args.folds, seed=args.split_seed)
    fold_results, out_of_fold = [], []
    assignments = development[["orientation", "series_id"]].copy()
    assignments.insert(0, "row_id", development.index)
    assignments["fold"] = -1

    for fold_index, split in enumerate(folds):
        fold_seed = args.seed + fold_index
        seed_everything(fold_seed)
        loaders = build_dataloaders(
            split, LogPowerLawDataset, feature_columns=args.features,
            batch_size=args.batch_size, balance_orientations=True, seed=fold_seed,
        )
        kwargs = model_kwargs(
            split.train, len(args.features), args.dropout, args.hidden_sizes
        )
        model = ConditionalLogPowerLawMLP(**kwargs)
        target_mean, target_scale, physical_scale = target_statistics(split.train)
        history, best_epoch = fit(
            model, loaders, config,
            target_mean=target_mean,
            target_scale=target_scale,
            physical_target_scale=physical_scale,
            physical_loss_weight=args.physical_loss_weight,
        )
        metrics, predictions = evaluate(
            model, loaders.validation, torch.device(args.device), "log",
            target_mean=target_mean, target_scale=target_scale,
        )
        predictions.insert(0, "row_id", split.validation.index.to_numpy())
        predictions["orientation"] = split.validation.orientation.to_numpy()
        predictions["series_id"] = split.validation.series_id.to_numpy()
        predictions["fold"] = fold_index
        out_of_fold.append(predictions)
        history.to_csv(output / f"fold_{fold_index}_history.csv", index=False)
        validation_series = sorted(split.validation.series_id.unique().tolist())
        fold_results.append({
            "fold": fold_index,
            "seed": fold_seed,
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "validation_series": validation_series,
            "metrics": metrics,
        })
        assignments.loc[assignments.series_id.isin(validation_series), "fold"] = fold_index

    predictions = pd.concat(out_of_fold).sort_values("row_id")
    predictions.to_csv(output / "oof_predictions.csv", index=False)
    assignments.to_csv(output / "fold_assignments.csv", index=False)
    test.assign(partition="test", row_id=test.index).to_csv(output / "held_out_test.csv", index=False)
    metrics = {
        "selection_metric": "out_of_fold.K_rmse",
        "out_of_fold": physical_metrics(predictions),
        "folds": fold_results,
        "suggested_final_epochs": int(round(np.median([fold["best_epoch"] for fold in fold_results]))),
        "test_evaluated": False,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    return metrics


def fit_fixed_epochs(
    model, loader, config, epochs, target_mean, target_scale,
    physical_scale, physical_weight,
):
    device = torch.device(config.device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history = []
    for epoch in range(1, epochs + 1):
        loss = run_epoch(
            model, loader, device, optimizer,
            target_mean=target_mean,
            target_scale=target_scale,
            physical_target_scale=physical_scale,
            physical_loss_weight=physical_weight,
        )
        history.append({"epoch": epoch, "train_loss": loss})
        if epoch == 1 or epoch % config.log_every == 0:
            print(f"Época final {epoch}: train loss={loss:.6g}")
    model.eval()
    return pd.DataFrame(history)


def run_final(args, config: TrainConfig, output: Path):
    if args.final_epochs is None or args.final_epochs < 1:
        raise ValueError("--final-epochs debe ser positivo en modo final")
    development, test = common_data(args.data_dir, args.split_seed)
    split = DataSplit(development, test, test)
    seed_everything(args.seed)
    loaders = build_dataloaders(
        split, LogPowerLawDataset, feature_columns=args.features,
        batch_size=args.batch_size, balance_orientations=True, seed=args.seed,
    )
    kwargs = model_kwargs(
        development, len(args.features), args.dropout, args.hidden_sizes
    )
    model = ConditionalLogPowerLawMLP(**kwargs)
    target_mean, target_scale, physical_scale = target_statistics(development)
    history = fit_fixed_epochs(
        model, loaders.train, config, args.final_epochs,
        target_mean, target_scale, physical_scale,
        args.physical_loss_weight,
    )
    history.to_csv(output / "history.csv", index=False)
    normalizer = loaders.normalizer
    target_normalizer = {"mean": target_mean, "scale": target_scale}
    torch.save({
        "model_name": "conditional-log-power-law",
        "model_kwargs": kwargs,
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "normalizer": {
            "columns": list(normalizer.columns),
            "log_columns": list(normalizer.log_columns),
            "mean": normalizer.mean.tolist(),
            "scale": normalizer.scale.tolist(),
        },
        "target_column": TARGET_COLUMN,
        "target_transform": "log",
        "target_normalizer": target_normalizer,
        "physical_target_scale": physical_scale,
        "physical_loss_weight": args.physical_loss_weight,
        "best_epoch": None,
        "trained_epochs": args.final_epochs,
        "train_config": asdict(config),
    }, output / "best_model.pt")
    metrics, predictions = evaluate(
        model, loaders.test, torch.device(args.device), "log",
        target_mean=target_mean, target_scale=target_scale,
    )
    predictions.insert(0, "row_id", test.index.to_numpy())
    predictions["orientation"] = test.orientation.to_numpy()
    predictions["series_id"] = test.series_id.to_numpy()
    predictions.to_csv(output / "test_predictions.csv", index=False)
    partitions = pd.concat([
        development.assign(partition="train", row_id=development.index),
        test.assign(partition="test", row_id=test.index),
    ])
    partitions.to_csv(output / "partitions.csv", index=False)
    result = {"trained_epochs": args.final_epochs, "test": metrics}
    (output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["cv", "final"], required=True)
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset" / "processed")
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--final-epochs", type=int)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--patience", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--physical-loss-weight", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=[32, 16])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-every", type=int, default=1000)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error(f"El directorio de salida ya existe: {args.output_dir}")
    if args.physical_loss_weight < 0:
        parser.error("El peso de pérdida debe ser no negativo")
    if not args.hidden_sizes or any(size < 1 for size in args.hidden_sizes):
        parser.error("hidden-sizes debe contener tamaños positivos")
    args.output_dir.mkdir(parents=True)
    config = TrainConfig(
        args.epochs, args.learning_rate, args.patience, 0.0,
        args.seed, args.device, args.log_every,
    )
    config.validate()
    serializable = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (args.output_dir / "config.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    try:
        result = run_cv(args, config, args.output_dir) if args.mode == "cv" else run_final(args, config, args.output_dir)
    except Exception:
        # Conserva config y logs para diagnosticar una ejecución fallida.
        raise
    print(json.dumps(result, indent=2))
    print(f"Resultados: {args.output_dir}")
    return args.output_dir


if __name__ == "__main__":
    main()
