"""Exporta artifacts/ -> panel/public/data/*.json para el panel Next.js estatico.

El panel es puramente estatico (output: 'export'): no hace fetch en runtime,
todo se genera en build a partir de estos JSON. El panel importa SOLO de
panel/public/data/, nunca de artifacts/ directamente (R9: la interfaz no
calcula, solo lee).

Nota sobre `regime_history.parquet`: el artefacto real que produce el
pipeline se llama `artifacts/latest/history.parquet` (ver
src/irfn/validation/walkforward.py / scripts/run_v3.py); no existe ningun
`regime_history.parquet`. Se lee ese archivo real. Sus columnas ya cubren lo
que pide history.json: fecha, xi_filtered_k (probabilidad por regimen),
entropy, argmax, block_id y `equity` (indice de retorno acumulado del activo
ancla sobre la ventana OOS, ya calculado en el artefacto -- se usa como
`price` de RegimeHistory; no es el precio de mercado bruto, es la curva de
equity de comprar-y-mantener sobre esas mismas fechas, que es lo unico que
viaja en este artefacto sin recalcular nada).

Uso: python scripts/export_panel_data.py
       python scripts/export_panel_data.py --asset BTC  # lee artifacts/btc/latest/,
       escribe en panel/public/data/btc/ (aditivo, no toca panel/public/data/*.json
       de SPY).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "latest"
REPORTS = ROOT / "reports"
OUT_DIR = ROOT / "panel" / "public" / "data"

VERDICT_ICONS = {"✓", "✗", "⚠", "—"}


def export_irfn_json(artifacts: Path, out_dir: Path) -> None:
    src = artifacts / "irfn.json"
    payload = json.loads(src.read_text(encoding="utf-8"))
    (out_dir / "irfn.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"irfn.json: run_id={payload.get('run_id')} version={payload.get('version')}")


def export_history_json(artifacts: Path, out_dir: Path) -> None:
    src = artifacts / "history.parquet"
    if not src.exists():
        raise FileNotFoundError(
            f"{src} no existe. El script esperaba regime_history.parquet (encargo) o, "
            "en su defecto, el artefacto real history.parquet (walk-forward OOS). "
            "Corre el walk-forward (scripts/run_v3.py o run_v1.py ablation) primero."
        )
    df = pd.read_parquet(src).sort_values("fecha").reset_index(drop=True)

    xi_cols = sorted(
        [c for c in df.columns if c.startswith("xi_filtered_")],
        key=lambda c: int(c.rsplit("_", 1)[1]),
    )
    if not xi_cols:
        raise ValueError(f"{src}: no se encontraron columnas xi_filtered_k.")

    # Frontera de bloque: primer dia de cada block_id nuevo (nunca el primer
    # dia de la serie completa, que no es una frontera real).
    block_change = df["block_id"] != df["block_id"].shift(1)
    block_change.iloc[0] = False

    records = []
    for i, row in df.iterrows():
        records.append(
            {
                "date": row["fecha"].strftime("%Y-%m-%d") if hasattr(row["fecha"], "strftime") else str(row["fecha"]),
                "xi": [float(row[c]) for c in xi_cols],
                "entropy": float(row["entropy"]),
                "price": float(row["equity"]),
                "block_boundary": bool(block_change.iloc[i]),
                "argmax": str(row["argmax"]),
            }
        )
    (out_dir / "history.json").write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    print(f"history.json: {len(records)} puntos, {sum(r['block_boundary'] for r in records)} fronteras de bloque")


def _parse_markdown_table(lines: list[str]) -> list[dict]:
    """Filas de una tabla markdown '| a | b | c |', saltando encabezado y separador."""
    rows = []
    for line in lines:
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(re.fullmatch(r":?-+:?", c) for c in cells):
            continue  # fila separadora (---|---|---)
        rows.append(cells)
    return rows


def export_validation_json(out_dir: Path, *, required: bool = True) -> None:
    """required=False (usado con --asset): reports/validation_v4.md es SIEMPRE
    el reporte de SPY (sin sufijo por activo; V4 nunca se ha corrido sobre BTC
    ni sobre ningun otro activo). Copiarlo tal cual dentro de la carpeta de
    otro activo seria enganoso -- se documenta el hueco en consola y se sigue
    sin escribir validation.json, en vez de fingir una validacion V4 que ese
    activo no tiene (R8)."""
    if not required:
        print("validation.json: omitido -- V4 solo existe para SPY (reports/validation_v4.md); "
              "este activo aun no tiene su propia validacion estadistica formal.")
        return
    src = REPORTS / "validation_v4.md"
    text = src.read_text(encoding="utf-8")
    lines = text.splitlines()

    generated_at = None
    m = re.search(r"^Fecha:\s*(.+)$", text, re.MULTILINE)
    if m:
        generated_at = m.group(1).strip()

    # Bloque "## Resumen ejecutivo" hasta el siguiente "## "
    try:
        start = next(i for i, l in enumerate(lines) if l.strip() == "## Resumen ejecutivo")
    except StopIteration as exc:
        raise ValueError(f"{src}: no se encontro la seccion '## Resumen ejecutivo'.") from exc
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    block = lines[start:end]

    table_rows = _parse_markdown_table(block)
    if not table_rows or table_rows[0] != ["Test", "Resultado en una frase", "Veredicto"]:
        raise ValueError(f"{src}: la tabla del resumen ejecutivo no tiene el encabezado esperado.")

    rows = []
    for cells in table_rows[1:]:
        if len(cells) != 3:
            continue
        test, result, verdict = cells
        rows.append({"test": test, "result": result, "verdict": verdict})

    if len(rows) != 7:
        raise ValueError(f"{src}: se esperaban 7 filas en el resumen ejecutivo, se encontraron {len(rows)}.")

    payload = {"generated_at": generated_at, "rows": rows, "source": "reports/validation_v4.md"}
    (out_dir / "validation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"validation.json: {len(rows)} filas extraidas de {src.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Exporta artifacts/ -> panel/public/data/*.json.")
    ap.add_argument("--asset", default=None,
                    help="Nombre del activo (p.ej. BTC). Sin este flag, comportamiento "
                         "identico al historico: lee artifacts/latest/, escribe en "
                         "panel/public/data/ (SPY).")
    args = ap.parse_args()

    if args.asset is None:
        artifacts, out_dir, required_validation = ARTIFACTS, OUT_DIR, True
    else:
        slug = args.asset.lower()
        artifacts = ROOT / "artifacts" / slug / "latest"
        out_dir = OUT_DIR / slug
        required_validation = False

    out_dir.mkdir(parents=True, exist_ok=True)
    export_irfn_json(artifacts, out_dir)
    export_history_json(artifacts, out_dir)
    export_validation_json(out_dir, required=required_validation)


if __name__ == "__main__":
    main()
