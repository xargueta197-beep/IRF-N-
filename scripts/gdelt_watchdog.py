"""Vigia de puntos de control OBLIGATORIOS cada 10% para el backfill de GDELT
(proceso no supervisado, corre mientras el usuario duerme).

Que hace, de forma autonoma (no necesita al agente despierto):
  - Cada ~90s calcula el avance = dias calendario cubiertos con snapshot dentro
    del rango del lote / total de dias del rango.
  - Al cruzar cada decil (10%, 20%, ... 100%) crea UN punto de control:
      * VALIDA cada snapshot del rango (JSON parseable, claves esperadas, `day`
        == nombre de archivo, n_total entero >= 0, segments es lista).
      * Lo sella APROBADO si 0 invalidos; REVISAR si hay alguno (lista cuales).
      * Lo apendea a un rastro de auditoria JSONL (durable) y reescribe un
        resumen legible en Markdown para leer en la manana.
  - Detecta fin (100% del rango) o estancamiento (log de GDELT sin cambios
    > 20 min) y hace el checkpoint final + resumen, y sale.

NO borra ni modifica datos: solo lee, valida y registra. Es idempotente:
guarda el ultimo decil sellado en un archivo de estado y reanuda si se reinicia.
"""
from __future__ import annotations

import glob
import json
import re
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADLINES = ROOT / "data" / "raw" / "headlines"
AUDIT = HEADLINES / "_gdelt_checkpoints.jsonl"
SUMMARY = HEADLINES / "_gdelt_watchdog_summary.md"
STATE = HEADLINES / "_gdelt_watchdog_state.json"
GDELT_LOG = Path(sys.argv[1]) if len(sys.argv) > 1 else None  # log del backfill (para stall)

POLL_SECS = 90
STALL_SECS = 20 * 60
STEP_PCT = 5   # granularidad de los puntos de control (%). Cambiado de 10 a 5.
EXPECTED_KEYS = {"captured_at", "source", "query", "day", "n_total", "cap_hit_any", "segments"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _range_from_log() -> tuple[date, date] | None:
    """Busca '...capturando titulares GDELT AAAA-MM-DD..AAAA-MM-DD...' en CUALQUIER
    linea del log. Antes solo miraba la 1a linea, pero el log puede empezar con
    otras lineas (p.ej. un 'start ...' de arranque): eso hacia caer al rango de
    respaldo. Ahora escanea todo el texto y toma la 1a coincidencia."""
    if GDELT_LOG is None or not GDELT_LOG.exists():
        return None
    try:
        text = GDELT_LOG.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    m = re.search(r"(\d{4}-\d{2}-\d{2})\.\.(\d{4}-\d{2}-\d{2})", text)
    if not m:
        return None
    return date.fromisoformat(m.group(1)), date.fromisoformat(m.group(2))


def _days_in_range(d0: date, d1: date) -> int:
    return (d1 - d0).days + 1


def _covered_files() -> list[Path]:
    return [Path(p) for p in glob.glob(str(HEADLINES / "*.json"))
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", Path(p).stem)]


def _validate(files: list[Path]) -> tuple[int, list[str]]:
    """Devuelve (n_validos, [archivos_invalidos con motivo])."""
    invalid: list[str] = []
    n_ok = 0
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            invalid.append(f"{f.name}: JSON ilegible ({e})")
            continue
        if not isinstance(d, dict) or not EXPECTED_KEYS.issubset(d.keys()):
            invalid.append(f"{f.name}: faltan claves esperadas")
            continue
        if d.get("day") != f.stem:
            invalid.append(f"{f.name}: 'day'={d.get('day')} no coincide con el nombre")
            continue
        if not isinstance(d.get("n_total"), int) or d["n_total"] < 0:
            invalid.append(f"{f.name}: n_total invalido ({d.get('n_total')})")
            continue
        if not isinstance(d.get("segments"), list):
            invalid.append(f"{f.name}: segments no es lista")
            continue
        n_ok += 1
    return n_ok, invalid


def _load_state() -> int:
    if STATE.exists():
        try:
            return int(json.loads(STATE.read_text(encoding="utf-8")).get("last_decile", 0))
        except Exception:
            return 0
    return 0


def _save_state(dec: int) -> None:
    STATE.write_text(json.dumps({"last_decile": dec, "updated": _now()}), encoding="utf-8")


def _checkpoint(decile: int, pct: float, covered: int, total: int, d0: date, d1: date) -> dict:
    files = [f for f in _covered_files() if d0 <= date.fromisoformat(f.stem) <= d1]
    n_ok, invalid = _validate(files)
    status = "APROBADO" if not invalid else "REVISAR"
    rec = {
        "ts": _now(), "decile_pct": decile, "avance_pct": round(pct, 1),
        "dias_cubiertos": covered, "dias_totales": total,
        "snapshots_validos": n_ok, "snapshots_invalidos": len(invalid),
        "status": status, "invalidos": invalid[:50],
    }
    with AUDIT.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    _write_summary(d0, d1)
    _save_state(decile)
    return rec


def _write_summary(d0: date, d1: date) -> None:
    rows = []
    if AUDIT.exists():
        for line in AUDIT.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    L = ["# Vigia GDELT — puntos de control (10%)\n",
         f"rango del lote: {d0}..{d1}  |  actualizado: {_now()}\n",
         "| hito % | avance % | dias | validos | invalidos | sello | hora (UTC) |",
         "| --: | --: | --: | --: | --: | :-- | :-- |"]
    for r in rows:
        L.append(f"| {r['decile_pct']} | {r['avance_pct']} | "
                 f"{r['dias_cubiertos']}/{r['dias_totales']} | {r['snapshots_validos']} | "
                 f"{r['snapshots_invalidos']} | {r['status']} | {r['ts'][11:19]} |")
    any_review = [r for r in rows if r["status"] == "REVISAR"]
    if any_review:
        L.append("\n## ATENCION: hitos marcados REVISAR (snapshots invalidos)\n")
        for r in any_review:
            for inv in r["invalidos"]:
                L.append(f"- [{r['decile_pct']}%] {inv}")
    SUMMARY.write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    rng = _range_from_log()
    if rng is None:
        # respaldo: rango conocido del lote actual
        d0, d1 = date(2025, 7, 7), date(2026, 7, 29)
    else:
        d0, d1 = rng
    total = _days_in_range(d0, d1)
    last_decile = _load_state()
    print(f"{_now()} vigia GDELT iniciado. rango {d0}..{d1} ({total} dias). "
          f"ultimo decil sellado={last_decile}%.", flush=True)

    while True:
        files = [f for f in _covered_files() if d0 <= date.fromisoformat(f.stem) <= d1]
        covered = len(files)
        pct = 100.0 * covered / total if total else 0.0
        decile = int(pct // STEP_PCT) * STEP_PCT

        # sellar cada hito nuevo cruzado (maneja saltos)
        while last_decile < min(decile, 100):
            last_decile += STEP_PCT
            rec = _checkpoint(last_decile, pct, covered, total, d0, d1)
            print(f"{_now()} CHECKPOINT {last_decile}% — {rec['status']} "
                  f"({rec['dias_cubiertos']}/{rec['dias_totales']} dias, "
                  f"{rec['snapshots_invalidos']} invalidos)", flush=True)

        # fin por cobertura completa
        if covered >= total:
            if last_decile < 100:
                rec = _checkpoint(100, 100.0, covered, total, d0, d1)
                print(f"{_now()} CHECKPOINT 100% (fin) — {rec['status']}", flush=True)
            print(f"{_now()} vigia: lote COMPLETO. saliendo.", flush=True)
            return

        # fin por estancamiento del backfill (log sin cambios)
        if GDELT_LOG is not None and GDELT_LOG.exists():
            stale = time.time() - GDELT_LOG.stat().st_mtime
            if stale > STALL_SECS:
                rec = _checkpoint(min(last_decile + STEP_PCT, 100), pct, covered, total, d0, d1)
                print(f"{_now()} vigia: backfill ESTANCADO ({int(stale)}s sin cambios) "
                      f"en {pct:.1f}%. checkpoint de parada y salida.", flush=True)
                return

        time.sleep(POLL_SECS)


if __name__ == "__main__":
    main()
