"""Valida una configuración con pliegues por series sobre el desarrollo original."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import build_dataloaders
from .datasets import LogPowerLawDataset
from .models import LogPowerLawMLP, RBF4, RBF5
from .normalization import Normalizer, TARGET_COLUMN
from .splitting import grouped_series_folds
from .train import TrainConfig, evaluate, evaluate_rbf, fit, rbf_arrays, seed_everything


RBF_MODELS = {"rbf_4": RBF4, "rbf_5": RBF5}


def physical_metrics(predictions: pd.DataFrame) -> dict:
    true = predictions["K_true"].to_numpy(dtype=float)
    predicted = predictions["K_predicted"].to_numpy(dtype=float)
    log_error = predictions["prediction"].to_numpy(dtype=float) - predictions["target"].to_numpy(dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        error = predicted - true
        squared = np.square(error)
    finite = bool(np.isfinite(squared).all())
    denominator = np.square(true - true.mean()).sum()
    return {
        "n": len(predictions),
        "target_mse": float(np.square(log_error).mean()),
        "K_rmse": float(np.sqrt(squared.mean())) if finite else None,
        "K_mae": float(np.abs(error).mean()) if finite else None,
        "K_r2": float(1 - squared.sum() / denominator) if finite and denominator > 0 else None,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--model", choices=["log-power-law", *RBF_MODELS], required=True)
    parser.add_argument("--combination", required=True)
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--patience", type=int, default=300)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("El directorio de salida ya existe")
    source_partitions = args.source_run / args.model / args.combination / "partitions.csv"
    data = pd.read_csv(source_partitions).set_index("row_id", drop=False)
    development = data.loc[data.partition.ne("test")].copy()
    held_out_test = data.loc[data.partition.eq("test")]
    if development.empty or held_out_test.empty or development.orientation.eq("cyclic").any():
        raise ValueError("La partición de origen no preserva la prueba cíclica")
    folds = grouped_series_folds(development, n_splits=args.folds, seed=args.seed)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "config.json").write_text(json.dumps({
        "source_run": str(args.source_run.resolve()), "model": args.model,
        "combination": args.combination, "features": args.features,
        "folds": args.folds, "seed": args.seed,
        "epochs": args.epochs, "patience": args.patience,
        "held_out_test_rows": len(held_out_test),
    }, indent=2), encoding="utf-8")
    predictions = []
    fold_results = []
    assignments = development[["row_id", "orientation", "series_id"]].copy()
    assignments["fold"] = -1
    for index, split in enumerate(folds):
        seed_everything(args.seed)
        if args.model in RBF_MODELS:
            normalizer = Normalizer(split.train, args.features)
            model = RBF_MODELS[args.model](scale_features=False)
            features, target = rbf_arrays(split.train, normalizer)
            model.fit(features, target)
            metrics, fold_predictions = evaluate_rbf(model, split.validation, normalizer)
            best_epoch, epochs_run = None, 0
        else:
            loaders = build_dataloaders(
                split, LogPowerLawDataset, feature_columns=args.features,
                batch_size=16, balance_orientations=False, seed=args.seed,
            )
            model = LogPowerLawMLP(input_dim=len(args.features), dropout=0.1)
            log_target = np.log(split.train[TARGET_COLUMN].to_numpy(dtype=float))
            target_mean = float(log_target.mean())
            target_scale = float(log_target.std(ddof=0))
            config = TrainConfig(
                epochs=args.epochs, patience=args.patience,
                seed=args.seed, device="cpu", log_every=1000,
            )
            history, best_epoch = fit(
                model, loaders, config,
                target_mean=target_mean, target_scale=target_scale,
            )
            epochs_run = len(history)
            history.to_csv(args.output_dir / f"fold_{index}_history.csv", index=False)
            metrics, fold_predictions = evaluate(
                model, loaders.validation, torch.device("cpu"), "log",
                target_mean=target_mean, target_scale=target_scale,
            )
        fold_predictions.insert(0, "row_id", split.validation.index.to_numpy())
        fold_predictions["orientation"] = split.validation.orientation.to_numpy()
        fold_predictions["series_id"] = split.validation.series_id.to_numpy()
        fold_predictions["fold"] = index
        predictions.append(fold_predictions)
        held_series = sorted(split.validation.series_id.unique().tolist())
        assignments.loc[assignments.series_id.isin(held_series), "fold"] = index
        fold_results.append({
            "fold": index, "validation_series": held_series,
            "train_n": len(split.train), "validation_n": len(split.validation),
            "best_epoch": best_epoch, "epochs_run": epochs_run,
            "metrics": metrics,
        })
        print(f"Pliegue {index + 1}/{args.folds}: RMSE(K)={metrics['K_rmse']}", flush=True)
    oof = pd.concat(predictions).sort_values("row_id")
    if len(oof) != len(development) or not oof.row_id.is_unique or assignments.fold.lt(0).any():
        raise ValueError("Las predicciones fuera de muestra no cubren una vez cada fila")
    oof.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    assignments.to_csv(args.output_dir / "fold_assignments.csv", index=False)
    fold_rmse = [fold["metrics"]["K_rmse"] for fold in fold_results]
    result = {
        "selection_metric": "out_of_fold.K_rmse",
        "out_of_fold": physical_metrics(oof),
        "fold_rmse_mean": float(np.mean(fold_rmse)) if all(x is not None for x in fold_rmse) else None,
        "fold_rmse_std": float(np.std(fold_rmse, ddof=1)) if all(x is not None for x in fold_rmse) else None,
        "folds": fold_results,
        "test_evaluated": False,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"OOF RMSE(K)={result['out_of_fold']['K_rmse']}")


if __name__ == "__main__":
    main()
