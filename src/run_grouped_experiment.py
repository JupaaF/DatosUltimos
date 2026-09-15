"""Ejecuta el estudio completo ConditionalLogPowerLaw con CV por series."""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from .grouped_study import physical_metrics
from .train import ROOT


FEATURES = ["trace_sigma", "density", "MCN", "q", "q_over_p", "a"]
SEEDS = [42, 43, 44, 45, 46]


def combinations():
    result = []
    index = 0
    for size in range(1, len(FEATURES) + 1):
        for features in itertools.combinations(FEATURES, size):
            index += 1
            result.append((index, list(features)))
    return result


def run_command(command, log_path: Path):
    environment = os.environ.copy()
    environment.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, env=environment)
    return log_path


def execute(commands, workers):
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_command, command, log): label for label, command, log in commands}
        for future in as_completed(futures):
            label = futures[future]
            future.result()
            print(f"OK {label}", flush=True)


def slug(index, features):
    return f"{index:02d}_{'+'.join(features)}"


def ensemble_seed_predictions(seed_dirs, output):
    frames = [pd.read_csv(path / "test_predictions.csv") for path in seed_dirs]
    keys = ["row_id", "orientation", "series_id", "target", "K_true"]
    reference = frames[0][keys].reset_index(drop=True)
    if not all(frame[keys].reset_index(drop=True).equals(reference) for frame in frames[1:]):
        raise ValueError("Las semillas del ensemble no comparten el mismo test")
    predictions = reference.copy()
    predictions["K_predicted"] = np.mean(
        [frame.K_predicted.to_numpy(dtype=float) for frame in frames], axis=0
    )
    predictions["prediction"] = np.log(predictions.K_predicted)
    predictions.to_csv(output / "ensemble_predictions.csv", index=False)
    metrics = {
        "seeds": SEEDS,
        "aggregation": "arithmetic_mean_K",
        "test": physical_metrics(predictions),
        "by_orientation": {
            name: physical_metrics(group) for name, group in predictions.groupby("orientation")
        },
    }
    (output / "ensemble_metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8"
    )
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--patience", type=int, default=1000)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error(f"El directorio de salida ya existe: {args.output_dir}")
    if args.workers < 1 or not 1 <= args.top <= 63:
        parser.error("workers y top deben ser positivos; top no puede superar 63")
    args.output_dir.mkdir(parents=True)
    cv_root = args.output_dir / "cv"
    cv_root.mkdir()
    specs = combinations()

    commands = []
    for index, features in specs:
        name = slug(index, features)
        output = cv_root / name
        command = [
            sys.executable, "-m", "src.grouped_study", "--mode", "cv",
            "--features", *features, "--epochs", str(args.epochs),
            "--patience", str(args.patience), "--seed", "42", "--split-seed", "42",
            "--physical-loss-weight", "1", "--output-dir", str(output),
        ]
        commands.append((f"CV {name}", command, cv_root / f"{name}.log"))
    execute(commands, args.workers)

    ranking = []
    for index, features in specs:
        name = slug(index, features)
        metrics = json.loads((cv_root / name / "metrics.json").read_text())
        ranking.append({
            "combination": name,
            "features": ",".join(features),
            "cv_K_rmse": metrics["out_of_fold"]["K_rmse"],
            "cv_K_mae": metrics["out_of_fold"]["K_mae"],
            "cv_K_r2": metrics["out_of_fold"]["K_r2"],
            "final_epochs": metrics["suggested_final_epochs"],
        })
    ranking.sort(key=lambda row: row["cv_K_rmse"])
    with (args.output_dir / "cv_ranking.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ranking[0])
        writer.writeheader()
        writer.writerows(ranking)

    top = ranking[:args.top]
    final_root = args.output_dir / "final_top10_five_seeds"
    final_root.mkdir()
    commands = []
    for row in top:
        features = row["features"].split(",")
        for seed in SEEDS:
            output = final_root / row["combination"] / f"seed_{seed}"
            output.parent.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable, "-m", "src.grouped_study", "--mode", "final",
                "--features", *features, "--final-epochs", str(row["final_epochs"]),
                "--seed", str(seed), "--split-seed", "42",
                "--physical-loss-weight", "1", "--output-dir", str(output),
            ]
            commands.append((
                f"FINAL {row['combination']} seed={seed}", command,
                output.parent / f"seed_{seed}.log",
            ))
    execute(commands, args.workers)

    summary = []
    for row in top:
        output = final_root / row["combination"]
        seed_dirs = [output / f"seed_{seed}" for seed in SEEDS]
        ensemble = ensemble_seed_predictions(seed_dirs, output)
        summary.append({
            **row,
            "ensemble_test_K_rmse": ensemble["test"]["K_rmse"],
            "ensemble_test_K_mae": ensemble["test"]["K_mae"],
            "ensemble_test_K_r2": ensemble["test"]["K_r2"],
        })
    with (args.output_dir / "ensemble_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0])
        writer.writeheader()
        writer.writerows(summary)
    print(f"Estudio completo: {args.output_dir}")


if __name__ == "__main__":
    main()
