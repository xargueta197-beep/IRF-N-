"""Captura de titulares GDELT: backfill reanudable + captura diaria hacia adelante.

    python scripts/capture_headlines.py                 # dias faltantes desde config
    python scripts/capture_headlines.py --since 2024-01-01
    python scripts/capture_headlines.py --max-days 200  # tanda acotada (backfill por tandas)

Mismo espiritu que capture_consensus.py: cada dia va a un snapshot JSON
inmutable en data/raw/headlines/ y los dias ya guardados se saltan, asi que el
script se puede correr N veces (cron o a mano) y solo rellena huecos. El API de
GDELT limita a 1 peticion cada 5 s: un backfill largo tarda horas y ESTA BIEN
que tarde -- corre en segundo plano y el pipeline usa lo que haya en disco,
documentando la cobertura (nunca fingiendo la que no hay).

Los dias se bajan del MAS RECIENTE al mas antiguo (newest_first): el artefacto
de hoy necesita la ventana reciente completa; el historico crece hacia atras.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from irfn.config import NewsConfig  # noqa: E402
from irfn.data.headlines import SNAPSHOT_DIR, capture_headlines_range  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("capture_headlines")


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill/captura diaria de titulares GDELT.")
    ap.add_argument("--since", default=None, help="fecha inicial (default: headlines.start_date de config)")
    ap.add_argument("--until", default=None, help="fecha final (default: ayer UTC; el dia en curso esta incompleto)")
    ap.add_argument("--max-days", type=int, default=None,
                    help="capturar a lo sumo N dias faltantes en esta corrida (tandas)")
    args = ap.parse_args()

    cfg = NewsConfig.load().headlines
    since = date.fromisoformat(args.since or cfg.start_date)
    # el dia UTC en curso se excluye por defecto: su snapshot seria parcial y los
    # snapshots son inmutables (no se reescriben cuando el dia termina).
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).date()
    until = date.fromisoformat(args.until) if args.until else yesterday

    if args.max_days is not None:
        # recorta el rango a los N dias faltantes mas recientes (newest_first).
        existing = {p.stem for p in SNAPSHOT_DIR.glob("*.json")} if SNAPSHOT_DIR.exists() else set()
        missing = []
        d = until
        while d >= since:
            if d.isoformat() not in existing:
                missing.append(d)
            if len(missing) >= args.max_days:
                break
            d -= timedelta(days=1)
        if not missing:
            log.info("sin dias faltantes en el rango; nada que capturar.")
            return
        since = min(missing)
        until = max(missing)

    log.info("capturando titulares GDELT %s..%s (query de config/news.yaml).", since, until)
    summary = capture_headlines_range(
        query=cfg.query, since=since, until=until,
        max_records=cfg.max_records,
        request_spacing_seconds=cfg.request_spacing_seconds,
        rate_limit_backoff_seconds=cfg.rate_limit_backoff_seconds,
        max_backoff_retries=cfg.max_backoff_retries,
        min_window_hours=cfg.min_window_hours,
    )
    log.info("resumen: %s", summary)


if __name__ == "__main__":
    main()
