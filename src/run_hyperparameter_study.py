"""Barrido de hiperparámetros sobre las mejores combinaciones ConditionalLogPowerLaw.

La selección se realiza exclusivamente mediante CV agrupada por series. Tras el
barrido, entrena con cinco semillas el mejor hiperparámetro de cada combinación.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from .grouped_study import physical_metrics


SEEDS = (42, 43, 44, 45, 46)


def architecture_name(hidden_sizes):
    return "x".join(str(size) for size in hidden_sizes)


def parse_architecture(value):
    try:
        sizes = tuple(int(size) for size in value.split("x"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use formatos como 32x16 o 64x32x16") from error
    if not sizes or any(size < 1 for size in sizes):
        raise argparse.ArgumentTypeError("Las capas deben tener tamaños positivos")
    return sizes


def hyperparameters(args):
    grid = itertools.product(
        args.architectures,
        args.learning_rates,
        args.batch_sizes,
        args.dropouts,
        args.physical_loss_weights,
    )
    return [
        {
            "hidden_sizes": hidden_sizes,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "dropout": dropout,
            "physical_loss_weight": physical_loss_weight,
        }
        for hidden_sizes, learning_rate, batch_size, dropout, physical_loss_weight in grid
    ]


def run_command(command, log_path):
    environment = os.environ.copy()
    environment.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            env=environment,
        )
    return log_path


def execute(commands, workers):
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_command, command, log): label
            for label, command, log in commands
        }
        for future in as_completed(futures):
            label = futures[future]
            future.result()
            print(f"OK {label}", flush=True)


def read_best_features(ranking_path, top):
    ranking = pd.read_csv(ranking_path).sort_values("cv_K_rmse").head(top)
    required = {"combination", "features", "cv_K_rmse"}
    if not required <= set(ranking.columns):
        raise ValueError(f"El ranking debe contener {sorted(required)}")
    return ranking.to_dict("records")


def trial_name(index, parameters):
    return (
        f"trial_{index:03d}_h{architecture_name(parameters['hidden_sizes'])}"
        f"_lr{parameters['learning_rate']:g}_b{parameters['batch_size']}"
        f"_d{parameters['dropout']:g}_w{parameters['physical_loss_weight']:g}"
    )


def ensemble_predictions(seed_dirs, output):
    frames = [pd.read_csv(path / "test_predictions.csv") for path in seed_dirs]
    keys = ["row_id", "orientation", "series_id", "target", "K_true"]
    reference = frames[0][keys].reset_index(drop=True)
    if not all(frame[keys].reset_index(drop=True).equals(reference) for frame in frames[1:]):
        raise ValueError("Las semillas no comparten exactamente el mismo test")
    predictions = reference.copy()
    predictions["K_predicted"] = np.mean(
        [frame.K_predicted.to_numpy(dtype=float) for frame in frames], axis=0
    )
    predictions["prediction"] = np.log(predictions.K_predicted)
    predictions.to_csv(output / "ensemble_predictions.csv", index=False)
    metrics = {
        "seeds": list(SEEDS),
        "aggregation": "arithmetic_mean_K",
        "test": physical_metrics(predictions),
        "by_orientation": {
            orientation: physical_metrics(group)
            for orientation, group in predictions.groupby("orientation")
        },
    }
    (output / "ensemble_metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8"
    )
    return metrics


def cv_command(args, features, parameters, output):
    return [
        sys.executable,
        "-m",
        "src.grouped_study",
        "--mode",
        "cv",
        "--features",
        *features,
        "--hidden-sizes",
        *(str(size) for size in parameters["hidden_sizes"]),
        "--learning-rate",
        str(parameters["learning_rate"]),
        "--batch-size",
        str(parameters["batch_size"]),
        "--dropout",
        str(parameters["dropout"]),
        "--physical-loss-weight",
        str(parameters["physical_loss_weight"]),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--seed",
        "42",
        "--split-seed",
        str(args.split_seed),
        "--output-dir",
        str(output),
    ]


def final_command(args, features, parameters, epochs, seed, output):
    return [
        sys.executable,
        "-m",
        "src.grouped_study",
        "--mode",
        "final",
        "--features",
        *features,
        "--hidden-sizes",
        *(str(size) for size in parameters["hidden_sizes"]),
        "--learning-rate",
        str(parameters["learning_rate"]),
        "--batch-size",
        str(parameters["batch_size"]),
        "--dropout",
        str(parameters["dropout"]),
        "--physical-loss-weight",
        str(parameters["physical_loss_weight"]),
        "--final-epochs",
        str(epochs),
        "--seed",
        str(seed),
        "--split-seed",
        str(args.split_seed),
        "--output-dir",
        str(output),
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--patience", type=int, default=1000)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--architectures",
        nargs="+",
        type=parse_architecture,
        default=[(16, 8), (32, 16), (64, 32)],
    )
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[3e-4, 1e-3])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[16, 32])
    parser.add_argument("--dropouts", nargs="+", type=float, default=[0.0, 0.1])
    parser.add_argument("--physical-loss-weights", nargs="+", type=float, default=[0.25, 1.0])
    args = parser.parse_args(argv)

    if args.output_dir.exists():
        parser.error(f"El directorio de salida ya existe: {args.output_dir}")
    if args.workers < 1 or args.top < 1 or args.epochs < 1 or args.patience < 1:
        parser.error("workers, top, epochs y patience deben ser positivos")
    if any(size < 1 for size in args.batch_sizes):
        parser.error("Los batch sizes deben ser positivos")
    if any(rate <= 0 for rate in args.learning_rates):
        parser.error("Los learning rates deben ser positivos")
    if any(not 0 <= dropout < 1 for dropout in args.dropouts):
        parser.error("Los dropout deben pertenecer a [0, 1)")
    if any(weight < 0 for weight in args.physical_loss_weights):
        parser.error("Los pesos de pérdida deben ser no negativos")

    selected = read_best_features(args.ranking, args.top)
    parameters = hyperparameters(args)
    args.output_dir.mkdir(parents=True)
    manifest = {
        "ranking": str(args.ranking.resolve()),
        "selected_models": selected,
        "hyperparameter_count": len(parameters),
        "total_cv_candidates": len(selected) * len(parameters),
        "grid": [
            {**item, "hidden_sizes": list(item["hidden_sizes"])} for item in parameters
        ],
        "epochs": args.epochs,
        "patience": args.patience,
        "split_seed": args.split_seed,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    cv_root = args.output_dir / "cv"
    cv_root.mkdir()
    commands = []
    trials = []
    for selected_model in selected:
        features = selected_model["features"].split(",")
        model_root = cv_root / selected_model["combination"]
        model_root.mkdir()
        for index, item in enumerate(parameters, start=1):
            name = trial_name(index, item)
            output = model_root / name
            trials.append((selected_model, features, name, item, output))
            commands.append((
                f"CV {selected_model['combination']} {name}",
                cv_command(args, features, item, output),
                model_root / f"{name}.log",
            ))
    execute(commands, args.workers)

    ranking_rows = []
    for selected_model, features, name, item, output in trials:
        metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        ranking_rows.append({
            "combination": selected_model["combination"],
            "features": ",".join(features),
            "trial": name,
            "hidden_sizes": architecture_name(item["hidden_sizes"]),
            "learning_rate": item["learning_rate"],
            "batch_size": item["batch_size"],
            "dropout": item["dropout"],
            "physical_loss_weight": item["physical_loss_weight"],
            "cv_K_rmse": metrics["out_of_fold"]["K_rmse"],
            "cv_K_mae": metrics["out_of_fold"]["K_mae"],
            "cv_K_r2": metrics["out_of_fold"]["K_r2"],
            "final_epochs": metrics["suggested_final_epochs"],
        })
    ranking_rows.sort(key=lambda row: row["cv_K_rmse"])
    with (args.output_dir / "hyperparameter_ranking.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=ranking_rows[0])
        writer.writeheader()
        writer.writerows(ranking_rows)

    best_per_model = []
    for selected_model in selected:
        candidates = [
            row for row in ranking_rows
            if row["combination"] == selected_model["combination"]
        ]
        best_per_model.append(min(candidates, key=lambda row: row["cv_K_rmse"]))
    pd.DataFrame(best_per_model).to_csv(
        args.output_dir / "best_hyperparameters.csv", index=False
    )

    final_root = args.output_dir / "final_best_five_seeds"
    final_root.mkdir()
    commands = []
    for row in best_per_model:
        item = {
            "hidden_sizes": parse_architecture(row["hidden_sizes"]),
            "learning_rate": row["learning_rate"],
            "batch_size": row["batch_size"],
            "dropout": row["dropout"],
            "physical_loss_weight": row["physical_loss_weight"],
        }
        features = row["features"].split(",")
        model_root = final_root / row["combination"]
        model_root.mkdir()
        for seed in SEEDS:
            output = model_root / f"seed_{seed}"
            commands.append((
                f"FINAL {row['combination']} seed={seed}",
                final_command(args, features, item, int(row["final_epochs"]), seed, output),
                model_root / f"seed_{seed}.log",
            ))
    execute(commands, args.workers)

    summaries = []
    for row in best_per_model:
        model_root = final_root / row["combination"]
        metrics = ensemble_predictions(
            [model_root / f"seed_{seed}" for seed in SEEDS], model_root
        )
        summaries.append({
            **row,
            "ensemble_test_K_rmse": metrics["test"]["K_rmse"],
            "ensemble_test_K_mae": metrics["test"]["K_mae"],
            "ensemble_test_K_r2": metrics["test"]["K_r2"],
        })
    pd.DataFrame(summaries).sort_values("ensemble_test_K_rmse").to_csv(
        args.output_dir / "ensemble_summary.csv", index=False
    )
    print(f"Estudio completo: {args.output_dir}")


if __name__ == "__main__":
    main()
