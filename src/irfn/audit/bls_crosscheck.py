"""Cliente de la API v2 de BLS, SOLO para control de calidad (@diagnostic_only).

NUNCA lo importa el pipeline del modelo. BLS sirve la serie REVISADA (el valor
vigente hoy), sin dimension de vintage: usarla como insumo del modelo violaria
R4 (datos macro desde vintages de ALFRED, jamas la serie revisada). Su unico
proposito es AUDITAR la ingesta macro cruzando, para cada mes observado:

  - BLS vigente            (valor revisado que publica BLS hoy)
  - ALFRED ultima vintage  (debe COINCIDIR con BLS vigente si la ingesta es sana)
  - ALFRED primer print    (lo que se publico en su momento; la diferencia con la
                            ultima vintage mide la magnitud de las revisiones)

API v2: POST https://api.bls.gov/publicAPI/v2/timeseries/data/
  - Con BLS_API_KEY (registrationkey, gratis): 500 consultas/dia, 20 anios/consulta.
  - Sin key: tier v1 (25 consultas/dia, 10 anios/consulta, sin calculos).
Registro gratis: https://data.bls.gov/registrationEngine/

El bloqueo efectivo contra publicar esto vive en outputs/publish.py (R1); el
decorador @diagnostic_only es la senal de intencion, igual que el smoother de Kim.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import requests

from irfn.models.hamilton import diagnostic_only

ROOT = Path(__file__).resolve().parents[3]
BLS_API_V2 = "https://api.bls.gov/publicAPI/v2/timeseries/data/"


@dataclass(frozen=True)
class CrosscheckPair:
    name: str
    bls_id: str
    alfred_id: str
    descripcion: str


# Correspondencias BLS <-> FRED/ALFRED para el cruce de QA. NO son parametros del
# modelo (por eso no van en config/): son identificadores de referencia de la
# herramienta de auditoria. Solo las series de ORIGEN BLS entre las que usa el
# proyecto; FEDFUNDS (Fed), NAPM (ISM) y RSAFS (Census) no son de BLS.
CROSSCHECK_PAIRS: dict[str, CrosscheckPair] = {
    "cpi": CrosscheckPair("cpi", "CUSR0000SA0", "CPIAUCSL", "IPC-U todos los items, SA (indice)"),
    "payrolls": CrosscheckPair("payrolls", "CES0000000001", "PAYEMS", "Empleo no agricola total, nominas (miles)"),
    "unemployment": CrosscheckPair("unemployment", "LNS14000000", "UNRATE", "Tasa de desempleo, SA (%)"),
}


def _api_key() -> str | None:
    """BLS_API_KEY del entorno/.env, o None (el cliente cae a tier v1 sin key)."""
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    return (os.environ.get("BLS_API_KEY") or "").strip() or None


def _period_to_month(year: str, period: str) -> pd.Timestamp | None:
    """Mnn -> primer dia del mes nn. M13 (promedio anual) y periodos no mensuales
    (Qnn/Snn/Ann) devuelven None: el cruce con ALFRED es a nivel mensual."""
    if not period.startswith("M") or period == "M13":
        return None
    try:
        return pd.Timestamp(int(year), int(period[1:]), 1)
    except ValueError:
        return None


@diagnostic_only
def fetch_bls_series(
    series_ids: list[str],
    *,
    start_year: int,
    end_year: int,
    api_key: str | None = None,
    timeout: float = 30.0,
) -> dict[str, pd.DataFrame]:
    """Descarga series MENSUALES de BLS v2. Devuelve {series_id: DataFrame[fecha, value]}
    ordenado ascendente por fecha. QA-only: `value` es el revisado vigente, sin vintage.

    Lanza RuntimeError si BLS no devuelve REQUEST_SUCCEEDED (p.ej. limite diario
    del tier alcanzado): un fallo de auditoria se reporta, no se enmascara.
    """
    payload: dict = {
        "seriesid": series_ids,
        "startyear": str(start_year),
        "endyear": str(end_year),
    }
    key = api_key if api_key is not None else _api_key()
    if key:
        payload["registrationkey"] = key

    resp = requests.post(BLS_API_V2, json=payload, timeout=timeout)
    resp.raise_for_status()
    body = resp.json()
    if body.get("status") != "REQUEST_SUCCEEDED":
        msg = "; ".join(body.get("message", [])) or "sin mensaje"
        raise RuntimeError(f"BLS API no exitosa ({body.get('status')}): {msg}")

    out: dict[str, pd.DataFrame] = {}
    for s in body.get("Results", {}).get("series", []):
        rows: list[dict] = []
        for d in s.get("data", []):
            fecha = _period_to_month(d.get("year", ""), d.get("period", ""))
            if fecha is None:
                continue
            try:
                val = float(d["value"])
            except (KeyError, ValueError):
                continue
            rows.append({"fecha": fecha, "value": val})
        df = (
            pd.DataFrame(rows).sort_values("fecha").reset_index(drop=True)
            if rows
            else pd.DataFrame(columns=["fecha", "value"])
        )
        out[s.get("seriesID", "?")] = df
    return out


@diagnostic_only
def crosscheck_against_alfred(
    name: str, *, n_months: int = 12, refresh: bool = False
) -> pd.DataFrame:
    """Cruza la serie BLS vigente contra ALFRED (primer print y ultima vintage) para
    los ultimos n_months meses observados en BLS. QA de la ingesta macro.

    Columnas: mes, bls_vigente, alfred_ultima_vintage, alfred_primer_print,
    dif_bls_vs_alfred_ultima (deberia ser ~0 si la ingesta es sana),
    revision_alfred (ultima_vintage - primer_print, informativa).

    Import de ALFRED perezoso a proposito: mantiene este modulo de QA desacoplado
    y deja claro que el flujo es audit -> data, nunca al reves.
    """
    from irfn.data.alfred import fetch_vintage_observations

    pair = CROSSCHECK_PAIRS[name]
    this_year = pd.Timestamp.today().year
    bls = fetch_bls_series(
        [pair.bls_id], start_year=this_year - 2, end_year=this_year
    )[pair.bls_id]

    v = fetch_vintage_observations(pair.alfred_id, refresh=refresh).dropna(subset=["value"])
    v = v.sort_values("realtime_start")
    first_print = v.groupby("obs_date")["value"].first()
    last_vintage = v.groupby("obs_date")["value"].last()

    rows: list[dict] = []
    for _, r in bls.tail(n_months).iterrows():
        od = pd.Timestamp(r["fecha"])
        fp = float(first_print[od]) if od in first_print.index else None
        lv = float(last_vintage[od]) if od in last_vintage.index else None
        rows.append(
            {
                "mes": od.date().isoformat(),
                "bls_vigente": float(r["value"]),
                "alfred_ultima_vintage": lv,
                "alfred_primer_print": fp,
                "dif_bls_vs_alfred_ultima": (float(r["value"]) - lv) if lv is not None else None,
                "revision_alfred": (lv - fp) if (lv is not None and fp is not None) else None,
            }
        )
    df = pd.DataFrame(rows)
    df.attrs["pair"] = asdict(pair)
    return df


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="QA (@diagnostic_only): cruza BLS revisado vs ALFRED vintages. NO toca el modelo."
    )
    ap.add_argument("--series", default="cpi", choices=list(CROSSCHECK_PAIRS),
                    help="serie de origen BLS a cruzar")
    ap.add_argument("--months", type=int, default=12, help="ultimos N meses a comparar")
    ap.add_argument("--refresh", action="store_true", help="refrescar el cache de vintages de ALFRED")
    args = ap.parse_args()

    pair = CROSSCHECK_PAIRS[args.series]
    df = crosscheck_against_alfred(args.series, n_months=args.months, refresh=args.refresh)
    tier = "con key (tier v2)" if _api_key() else "SIN key (tier v1, 25 consultas/dia)"
    print(f"BLS QA [{args.series}] {pair.descripcion} | BLS {pair.bls_id} vs ALFRED {pair.alfred_id} | {tier}")
    print(df.to_string(index=False))
    if df["dif_bls_vs_alfred_ultima"].abs().max() and df["dif_bls_vs_alfred_ultima"].abs().max() > 0.5:
        print("\nAVISO: hay meses donde BLS vigente y la ultima vintage de ALFRED difieren > 0.5; revisar la ingesta.")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    _main()
