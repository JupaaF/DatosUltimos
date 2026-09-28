"""Construye el visor HTML autocontenido de las predicciones zigzag."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runs" / "zigzag_predictions_cv_top5_20260928"
TEMPLATE = Path(__file__).with_name("predict_viewer_template.html")
OUTPUT = ROOT / "results_zigzag_predict_viewer.html"
ARCHITECTURES = ("log-power-law", "rbf_4", "rbf_5")
COLORS = {
    "log-power-law": ("#235eb4", "#3b7aca", "#5b95d3", "#1e4890", "#88afd9"),
    "rbf_4": ("#cb6237", "#df8755", "#ac4a31", "#edaa74", "#9a6852"),
    "rbf_5": ("#8751aa", "#a774bf", "#704393", "#bd96ce", "#946885"),
}


def main() -> None:
    manifest = pd.read_csv(SOURCE / "zigzag_metrics.csv")
    frames = {name: pd.read_csv(SOURCE / name / "predictions_wide.csv") for name in ARCHITECTURES}
    reference = frames[ARCHITECTURES[0]]
    required = ["row_id", "checkpoint", "time", "trace_sigma", "density", "K_true"]
    for name, frame in frames.items():
        if len(frame) != len(reference) or not frame["row_id"].equals(reference["row_id"]):
            raise ValueError(f"Las filas de {name} no coinciden con las demás")
        if not np.allclose(frame["K_true"], reference["K_true"], rtol=0, atol=1e-12):
            raise ValueError(f"K real difiere para {name}")
        if not set(required).issubset(frame.columns):
            raise ValueError(f"Faltan columnas en {name}")
    models = []
    for architecture in ARCHITECTURES:
        selected = manifest.loc[manifest.architecture.eq(architecture)].sort_values("rank_cv")
        if selected.rank_cv.tolist() != [1, 2, 3, 4, 5]:
            raise ValueError(f"Se esperan cinco redes ordenadas para {architecture}")
        for item in selected.itertuples():
            models.append({
                "id": f"{architecture}_{item.rank_cv}",
                "architecture": architecture,
                "rank": int(item.rank_cv),
                "combination": item.combination,
                "features": item.features.split(","),
                "cv_rmse": float(item.cv_oof_K_rmse),
                "zigzag_rmse": float(item.zigzag_K_rmse),
                "color": COLORS[architecture][item.rank_cv - 1],
            })
    rows = []
    for position, item in reference.iterrows():
        values = [float(frames[model["architecture"]].iloc[position][f"K_pred_{model['rank']}"])
                  for model in models]
        if not np.isfinite(values).all():
            raise ValueError(f"Predicción no finita en fila {position}")
        rows.append({
            "row_id": int(item.row_id),
            "checkpoint": int(item.checkpoint),
            "time": float(item.time),
            "pressure": float(item.trace_sigma),
            "density": float(item.density),
            "K_true": float(item.K_true),
            "values": values,
        })
    payload = json.dumps({"models": models, "rows": rows}, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("<", "\\u003c")
    template = TEMPLATE.read_text(encoding="utf-8")
    if template.count("__EMBEDDED_DATA__") != 1:
        raise ValueError("El template requiere exactamente un marcador de datos")
    OUTPUT.write_text(template.replace("__EMBEDDED_DATA__", payload), encoding="utf-8")
    print(f"{OUTPUT}: {len(rows)} puntos, {len(models)} redes")


if __name__ == "__main__":
    main()
