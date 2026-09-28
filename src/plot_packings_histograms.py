"""Genera histogramas de las mediciones raw y de K = tr(K)/3."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import AutoMinorLocator, MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "dataset" / "raw"
OUTPUT_DIR = ROOT / "figures" / "packings_histograms"
PLOTS_PER_PAGE = 6
GROUPS = (
    ("horizontal", "Horizontal", "#1f77b4"),
    ("vertical", "Vertical", "#ff7f0e"),
    ("cyclic_1", "Cíclico 1", "#2ca02c"),
    ("cyclic_2", "Cíclico 2", "#d62728"),
)


def load_groups() -> tuple[dict[str, pd.DataFrame], list[str]]:
    horizontal = pd.read_csv(RAW_DIR / "stable_packings_horizontal.csv")
    vertical = pd.read_csv(RAW_DIR / "stable_packings_vertical.csv")
    cyclic = pd.read_csv(RAW_DIR / "stable_packings_cyclic.csv")
    if horizontal.columns.tolist() != vertical.columns.tolist() or horizontal.columns.tolist() != cyclic.columns.tolist():
        raise ValueError("Los tres CSV deben tener las mismas columnas y en el mismo orden")
    for data in (horizontal, vertical, cyclic):
        data["K"] = (data["K_xx"] + data["K_yy"] + data["K_zz"]) / 3
    groups = {
        "horizontal": horizontal,
        "vertical": vertical,
        "cyclic_1": cyclic.loc[cyclic["series"].eq("cyclic_1")].copy(),
        "cyclic_2": cyclic.loc[cyclic["series"].eq("cyclic_2")].copy(),
    }
    variables = [column for column in horizontal.columns if column not in {"series", "time", "checkpoint", "K"}]
    variables.insert(variables.index("K_zz") + 1, "K")
    for name, data in groups.items():
        if data.empty:
            raise ValueError(f"El grupo {name} no contiene filas")
        for variable in variables:
            values = pd.to_numeric(data[variable], errors="raise").to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"{name}: {variable} contiene valores no finitos")
    return groups, variables


def bin_edges(values: np.ndarray) -> np.ndarray:
    """Usa entre 60 y 120 intervalos para dar más detalle al eje horizontal."""
    low, high = float(values.min()), float(values.max())
    if low == high:
        padding = abs(low) * 0.05 or 0.5
        return np.linspace(low - padding, high + padding, 61)
    suggested = len(np.histogram_bin_edges(values, bins="auto")) - 1
    return np.linspace(low, high, min(120, max(60, 2 * suggested)) + 1)


def x_label(variable: str) -> str:
    if variable in {"p", "q"} or variable.startswith("sigma_"):
        return f"{variable} [Pa]"
    return variable


def format_axis(axis: plt.Axes, variable: str) -> None:
    axis.set_title(variable, fontsize=12, loc="left", fontweight="bold")
    axis.set_xlabel(x_label(variable), fontsize=10)
    axis.set_ylabel("Frecuencia", fontsize=10)
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(labelsize=8)
    axis.ticklabel_format(axis="x", style="sci", scilimits=(-3, 4), useMathText=True)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=7))
    axis.xaxis.set_minor_locator(AutoMinorLocator(2))


def render_pdf(
    path: Path,
    title: str,
    variables: list[str],
    groups: dict[str, pd.DataFrame],
    *,
    pooled: bool,
) -> None:
    pages = (len(variables) + PLOTS_PER_PAGE - 1) // PLOTS_PER_PAGE
    with PdfPages(path, metadata={"Title": title, "Subject": "Histogramas de packings"}) as pdf:
        for page in range(pages):
            fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=False)
            fig.suptitle(f"{title} · página {page + 1}/{pages}", y=0.985, fontsize=17, fontweight="bold")
            for axis, variable in zip(axes.flat, variables[page * PLOTS_PER_PAGE:(page + 1) * PLOTS_PER_PAGE]):
                if pooled:
                    values = np.concatenate([groups[key][variable].to_numpy(dtype=float) for key, _, _ in GROUPS])
                    color = "#6f42a1"
                else:
                    key, _, color = next(group for group in GROUPS if group[1] == title)
                    values = groups[key][variable].to_numpy(dtype=float)
                axis.hist(values, bins=bin_edges(values), color=color, edgecolor="white", linewidth=0.4)
                format_axis(axis, variable)
            for axis in list(axes.flat)[len(variables[page * PLOTS_PER_PAGE:(page + 1) * PLOTS_PER_PAGE]):]:
                axis.set_visible(False)
            fig.subplots_adjust(left=0.07, right=0.97, bottom=0.075, top=0.91, wspace=0.26, hspace=0.4)
            pdf.savefig(fig)
            if pooled and page == 0:
                fig.savefig(OUTPUT_DIR / "comparacion_vista_previa.png", dpi=160)
            plt.close(fig)


def main() -> None:
    groups, variables = load_groups()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for key, label, _ in GROUPS:
        path = OUTPUT_DIR / f"histogramas_{key}.pdf"
        render_pdf(path, label, variables, groups, pooled=False)
        print(f"{path}: {len(groups[key])} filas, {len(variables)} variables")
    path = OUTPUT_DIR / "histogramas_comparacion.pdf"
    total_rows = sum(len(data) for data in groups.values())
    render_pdf(path, f"Todos los datos (n={total_rows})", variables, groups, pooled=True)
    print(f"{path}: {total_rows} filas reunidas, {len(variables)} variables")


if __name__ == "__main__":
    main()
