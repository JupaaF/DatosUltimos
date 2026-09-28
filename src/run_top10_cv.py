"""Valida por series las diez mejores combinaciones de cada arquitectura."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd


MODELS = ("log-power-law", "rbf_4", "rbf_5")


def run_one(command: list[str], log_path: Path) -> tuple[int, str]:
    environment = os.environ.copy()
    environment.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
    return result.returncode, str(log_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--patience", type=int, default=300)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("El directorio de salida ya existe")
    if args.folds < 2 or args.workers < 1 or not 1 <= args.top <= 63:
        parser.error("folds, workers y top deben ser positivos")
    summary = pd.read_csv(args.source_run / "summary.csv")
    selected = pd.concat([
        summary.loc[summary.model.eq(model)].sort_values("validation_K_rmse").head(args.top)
        for model in MODELS
    ], ignore_index=True)
    if len(selected) != len(MODELS) * args.top or selected.validation_K_rmse.isna().any():
        raise ValueError("La suite de origen no contiene suficientes candidatos válidos")
    args.output_dir.mkdir(parents=True)
    selected[["model", "combination", "features", "validation_K_rmse"]].to_csv(
        args.output_dir / "selected_configs.csv", index=False
    )
    (args.output_dir / "config.json").write_text(json.dumps({
        "source_run": str(args.source_run.resolve()),
        "folds": args.folds, "top_per_model": args.top, "seed": args.seed,
        "epochs": args.epochs, "patience": args.patience,
        "selection": "top by previous validation K RMSE within each architecture",
        "test_evaluated": False,
    }, indent=2), encoding="utf-8")
    commands = []
    for row in selected.itertuples():
        model_root = args.output_dir / row.model
        model_root.mkdir(exist_ok=True)
        output = model_root / row.combination
        command = [
            sys.executable, "-m", "src.top10_cv_worker",
            "--source-run", str(args.source_run),
            "--model", row.model,
            "--combination", row.combination,
            "--features", *row.features.split(","),
            "--output-dir", str(output),
            "--folds", str(args.folds),
            "--seed", str(args.seed),
            "--epochs", str(args.epochs),
            "--patience", str(args.patience),
        ]
        commands.append((row.model, row.combination, command, model_root / f"{row.combination}.log"))
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_one, command, log): (model, name)
            for model, name, command, log in commands
        }
        for future in as_completed(futures):
            model, name = futures[future]
            code, log_path = future.result()
            if code:
                failures.append({"model": model, "combination": name, "return_code": code, "log": log_path})
                print(f"ERROR {model} {name}: {log_path}", flush=True)
            else:
                print(f"OK {model} {name}", flush=True)
    if failures:
        pd.DataFrame(failures).to_csv(args.output_dir / "failures.csv", index=False)
        raise RuntimeError(f"Fallaron {len(failures)} configuraciones; revisar failures.csv")
    rows = []
    for row in selected.itertuples():
        metrics = json.loads((args.output_dir / row.model / row.combination / "metrics.json").read_text())
        rows.append({
            "model": row.model, "combination": row.combination, "features": row.features,
            "previous_validation_K_rmse": row.validation_K_rmse,
            "cv_oof_K_rmse": metrics["out_of_fold"]["K_rmse"],
            "cv_oof_K_mae": metrics["out_of_fold"]["K_mae"],
            "cv_oof_K_r2": metrics["out_of_fold"]["K_r2"],
            "cv_fold_rmse_mean": metrics["fold_rmse_mean"],
            "cv_fold_rmse_std": metrics["fold_rmse_std"],
        })
    ranking = pd.DataFrame(rows).sort_values("cv_oof_K_rmse", na_position="last")
    ranking.to_csv(args.output_dir / "cv_summary.csv", index=False)
    winner = ranking.iloc[0].to_dict()
    (args.output_dir / "cv_winner.json").write_text(
        json.dumps(winner, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"Ganador CV: {winner['model']} {winner['combination']}")
    print(f"Resultados: {args.output_dir}")


if __name__ == "__main__":
    main()
