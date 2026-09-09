"""Comparación reproducible: python -m src.train --seed 42."""

import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

if __package__:
    from .data import FEATURE_COLUMNS, TARGET_COLUMN, load_data, normalization_features
    from .models import model_candidates
    from .splitting import random_split
else:
    from data import FEATURE_COLUMNS, TARGET_COLUMN, load_data, normalization_features
    from models import model_candidates
    from splitting import random_split


def regression_metrics(y_true, y_pred):
    """MAE, RMSE y 1 - R²; cuanto menor, mejor."""
    return {"MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "1-R2": 1 - r2_score(y_true, y_pred)}


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=root / "results" / "random_split")
    args = parser.parse_args()
    folder = root / "dataset" / "processed"
    data = load_data(folder / "stable_packings_horizontal.csv",
                     folder / "stable_packings_vertical.csv")
    split = random_split(data, seed=args.seed)
    x_train = normalization_features(split.train)
    x_val = normalization_features(split.validation)
    x_test = normalization_features(split.test)
    if not np.isfinite(data[TARGET_COLUMN]).all() or (data[TARGET_COLUMN] <= 0).any():
        raise ValueError("trace_K debe ser positivo y finito para comparar salida logarítmica")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    partitions = []
    for name, frame in (("train", split.train), ("validation", split.validation), ("test", split.test)):
        metadata = frame[["orientation", "series_id"]].copy()
        metadata["partition"] = name
        metadata["seed"] = args.seed
        partitions.append(metadata)
    pd.concat(partitions).sort_index().to_csv(args.output_dir / "partitions.csv", index_label="row_id")

    records, fitted = [], []
    for name, target, model in model_candidates(args.seed):
        with warnings.catch_warnings(record=True) as fit_warnings:
            warnings.simplefilter("always")
            model.fit(x_train, split.train[TARGET_COLUMN])
        record = {"model": name, "target_transform": target,
                  "fit_warnings": " | ".join(dict.fromkeys(str(w.message) for w in fit_warnings)),
                  **regression_metrics(split.validation[TARGET_COLUMN], model.predict(x_val))}
        records.append(record)
        fitted.append(model)
    validation = pd.DataFrame(records)
    best = int(validation["RMSE"].argmin())
    validation["selected"] = validation.index == best
    validation.to_csv(args.output_dir / "validation_metrics.csv", index=False, float_format="%.6e")

    # Test se evalúa después de seleccionar por validación; no decide el ganador.
    test_records, predictions = [], []
    for i, (record, model) in enumerate(zip(records, fitted)):
        predicted = model.predict(x_test)
        test_records.append({"model": record["model"], "target_transform": record["target_transform"],
                             "fit_warnings": record["fit_warnings"],
                             "selected": i == best,
                             **regression_metrics(split.test[TARGET_COLUMN], predicted)})
        frame = split.test[["orientation", "series_id", TARGET_COLUMN]].copy()
        frame["prediction"] = predicted
        frame["model"] = record["model"]
        frame["target_transform"] = record["target_transform"]
        predictions.append(frame)
    pd.DataFrame(test_records).to_csv(args.output_dir / "test_metrics.csv", index=False, float_format="%.6e")
    pd.concat(predictions).to_csv(args.output_dir / "test_predictions.csv", index_label="row_id")
    joblib.dump({"model": fitted[best], "features": FEATURE_COLUMNS,
                 "log_columns": ["trace_sigma"], "target": TARGET_COLUMN,
                 "seed": args.seed, "configuration": records[best]}, args.output_dir / "best_model.joblib")
    print(f"Random split: {len(split.train)}/{len(split.validation)}/{len(split.test)}; seed={args.seed}")
    print(f"Seleccionado: {records[best]['model']} ({records[best]['target_transform']})")
    for label, record in (("Validación", records[best]), ("Test", test_records[best])):
        print(f"{label}: " + ", ".join(
            f"{metric}={record[metric]:.6e}" for metric in ("MAE", "RMSE", "1-R2")
        ))
    if records[best]["fit_warnings"]:
        print("Avisos del ajuste:", records[best]["fit_warnings"])
    print(f"Resultados: {args.output_dir}")


if __name__ == "__main__":
    main()
