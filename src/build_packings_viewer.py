"""Incrusta los CSV raw en el visualizador HTML autocontenido."""

import csv
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "dataset" / "raw"
VIEWER = ROOT / "results_N64_3d_viewer.html"
DATA_BLOCK = re.compile(
    r'(<script id="embedded-data" type="application/json">).*?(</script>)',
    re.DOTALL,
)


def main() -> None:
    columns = None
    rows = []
    for kind in ("horizontal", "vertical", "cyclic"):
        with (RAW_DIR / f"stable_packings_{kind}.csv").open(newline="") as stream:
            reader = csv.DictReader(stream)
            if columns is None:
                columns = reader.fieldnames
            elif reader.fieldnames != columns:
                raise ValueError(f"Las columnas de {kind} difieren del resto")
            for record in reader:
                values = [float(record[column]) for column in columns if column != "series"]
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(f"Valor no finito en {kind}, checkpoint {record['checkpoint']}")
                by_name = dict(zip((column for column in columns if column != "series"), values))
                k = (by_name["K_xx"] + by_name["K_yy"] + by_name["K_zz"]) / 3
                group = record["series"] if kind == "cyclic" else kind
                rows.append([group, record["series"], *values, k])

    numeric_columns = [column for column in columns if column != "series"] + ["K"]
    payload = json.dumps({"columns": numeric_columns, "rows": rows}, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("<", "\\u003c")
    html = VIEWER.read_text()
    html, replacements = DATA_BLOCK.subn(lambda match: match.group(1) + payload + match.group(2), html, count=1)
    if replacements != 1:
        raise ValueError("No se encontró el bloque embedded-data del visor")
    VIEWER.write_text(html)
    print(f"{VIEWER}: {len(rows)} filas, {len(numeric_columns)} variables numéricas")


if __name__ == "__main__":
    main()
