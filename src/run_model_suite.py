"""Ejecuta todas las combinaciones de entradas para uno o varios modelos."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd


FEATURES = ["trace_sigma", "density", "MCN", "q", "q_over_p", "a"]


def feature_combinations():
    index = 0
    for size in range(1, len(FEATURES) + 1):
        for features in itertools.combinations(FEATURES, size):
            index += 1
            yield index, list(features)


def run_one(command, log_path: Path):
    environment = os.environ.copy()
    environment.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT,
            check=True, env=environment,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", required=True,
        choices=["rbf_4", "rbf_5", "log-power-law"],
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--patience", type=int, default=1000)
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error(f"El directorio de salida ya existe: {args.output_dir}")
    if args.workers < 1:
        parser.error("workers debe ser positivo")

    args.output_dir.mkdir(parents=True)
    specs = list(feature_combinations())
    commands = []
    for model in args.models:
        model_root = args.output_dir / model
        model_root.mkdir()
        for index, features in specs:
            name = f"{index:02d}_{'+'.join(features)}"
            output = model_root / name
            command = [
                sys.executable, "-m", "src.train",
                "--model", model,
                "--features", *features,
                "--split-seed", str(args.split_seed),
                "--epochs", str(args.epochs),
                "--patience", str(args.patience),
                "--output-dir", str(output),
            ]
            if model == "log-power-law":
                command.append("--scale-target")
            commands.append((model, name, command, model_root / f"{name}.log"))

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_one, command, log): (model, name)
            for model, name, command, log in commands
        }
        for future in as_completed(futures):
            model, name = futures[future]
            future.result()
            print(f"OK {model} {name}", flush=True)

    rows = []
    for model in args.models:
        for index, features in specs:
            name = f"{index:02d}_{'+'.join(features)}"
            metrics = json.loads(
                (args.output_dir / model / name / "metrics.json").read_text()
            )
            rows.append({
                "model": model,
                "combination": name,
                "features": ",".join(features),
                "validation_K_rmse": metrics["validation"]["K_rmse"],
                "validation_K_mae": metrics["validation"]["K_mae"],
                "validation_K_r2": metrics["validation"]["K_r2"],
                "test_K_rmse": metrics["test"]["K_rmse"],
                "test_K_mae": metrics["test"]["K_mae"],
                "test_K_r2": metrics["test"]["K_r2"],
                "best_epoch": metrics.get("best_epoch"),
                "epochs_run": metrics.get("epochs_run"),
            })
    summary = pd.DataFrame(rows).sort_values(
        ["model", "validation_K_rmse"], na_position="last"
    )
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    print(f"Suite completa: {args.output_dir}")


if __name__ == "__main__":
    main()
