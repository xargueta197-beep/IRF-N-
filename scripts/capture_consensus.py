"""Guarda el calendario macro CON CONSENSO todos los dias. Arranca desde
Sesion 0 aunque no se use hasta V2: el consenso NO se puede reconstruir
ex-post (Trampa 2 de la Guia de Implementacion, Parte 7.2). Cada dia que pasa
sin guardarlo es un dia de historico perdido para siempre.

Escribe un snapshot inmutable por dia en data/raw/consensus_calendar/, con
timestamp de captura. Nunca sobreescribe un snapshot anterior (mismo
principio que data/raw/: append-only). Si la fuente falla o no hay API key
configurada, escribe igual un snapshot que registra el fallo — el hueco en
el historico tambien es informacion, y es mejor que un hueco silencioso.

Uso manual:
    python scripts/capture_consensus.py

Uso con Task Scheduler (Windows) — instrucciones al final de este archivo.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "raw" / "consensus_calendar"

# Endpoint de calendario de Trading Economics. Requiere API key de pago para
# uso en produccion. "guest:guest" es la clave demo publica documentada en
# docs.tradingeconomics.com, pero VERIFICADO en vivo (2026-07-11): devuelve
# HTTP 410 Gone de forma consistente, tanto via este script como via fetch
# directo. El demo publico esta muerto en la practica aunque la
# documentacion lo siga listando; hace falta una key de pago real (ver
# reports/data_audit.md para el veredicto completo sobre costo y cobertura
# historica de consenso de esta fuente, incluido el endpoint point-in-time
# que si preserva consenso historico pre-revision).
TRADING_ECONOMICS_CALENDAR_URL = "https://api.tradingeconomics.com/calendar"


def _api_key() -> str:
    load_dotenv(REPO_ROOT / ".env")
    # "or": una variable PRESENTE pero vacia (TRADING_ECONOMICS_API_KEY= en
    # .env) debe caer al demo igual que una ausente; con .get() a secas la
    # cadena vacia se colaba como key y el endpoint devolvia 401 con c=
    # (verificado en los snapshots de 2026-07-12..16).
    return os.environ.get("TRADING_ECONOMICS_API_KEY", "").strip() or "guest:guest"


def fetch_calendar() -> dict:
    """Trae el calendario del dia (releases + consenso) desde Trading Economics.

    Devuelve un dict con la respuesta cruda (o el error) — nunca lanza, para
    que el snapshot diario se escriba siempre, incluso en fallo (ver docstring
    del modulo).
    """
    try:
        response = requests.get(
            TRADING_ECONOMICS_CALENDAR_URL,
            params={"c": _api_key(), "f": "json"},
            timeout=30,
        )
        response.raise_for_status()
        return {"ok": True, "status_code": response.status_code, "data": response.json()}
    except requests.RequestException as exc:
        return {"ok": False, "error": str(exc)}


def capture_snapshot() -> Path:
    now = datetime.now(timezone.utc)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    out_path = OUT_DIR / f"{now:%Y-%m-%d}.json"
    if out_path.exists():
        print(f"Snapshot de hoy ya existe, no se sobreescribe: {out_path}")
        return out_path

    payload = {
        "captured_at": now.isoformat(),
        "source": "tradingeconomics.com/calendar",
        "result": fetch_calendar(),
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Snapshot guardado: {out_path}")
    return out_path


def main() -> None:
    path = capture_snapshot()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload["result"]["ok"]:
        print(f"AVISO: la captura de hoy fallo: {payload['result']['error']}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# INSTRUCCIONES — Task Scheduler en Windows
# ---------------------------------------------------------------------------
#
# Objetivo: correr este script una vez al dia, todos los dias, sin depender
# de que Xavier se acuerde de hacerlo manualmente.
#
# Opcion A — linea de comandos (schtasks), mas rapido de repetir/versionar:
#
#   schtasks /create /tn "IRFN_capture_consensus" /sc daily /st 07:00 ^
#       /tr "\"C:\Users\snoub\Downloads\NO SE EN QUE GASTAR MI TIRMPO\irfn\.venv\Scripts\python.exe\" \"C:\Users\snoub\Downloads\NO SE EN QUE GASTAR MI TIRMPO\irfn\scripts\capture_consensus.py\""
#
#   (ajustar la hora /st y las rutas si el repo se mueve). Verificar con:
#       schtasks /query /tn "IRFN_capture_consensus" /v /fo list
#
#   Para borrarla mas adelante:
#       schtasks /delete /tn "IRFN_capture_consensus" /f
#
# Opcion B — interfaz grafica (Task Scheduler / "Programador de tareas"):
#   1. Abrir "Programador de tareas" (taskschd.msc).
#   2. Accion > Crear tarea basica... > nombre "IRFN_capture_consensus".
#   3. Desencadenador: Diariamente, hora fija (ej. 07:00, cuando ya cerraron
#      la mayoria de releases del dia anterior en distintas zonas horarias).
#   4. Accion: Iniciar un programa.
#      Programa/script:
#          C:\Users\snoub\Downloads\NO SE EN QUE GASTAR MI TIRMPO\irfn\.venv\Scripts\python.exe
#      Argumentos:
#          scripts\capture_consensus.py
#      Iniciar en:
#          C:\Users\snoub\Downloads\NO SE EN QUE GASTAR MI TIRMPO\irfn
#   5. Marcar "Ejecutar tanto si el usuario inicio sesion como si no" si se
#      quiere que corra aunque el equipo este bloqueado (pedira la
#      contrasena de Windows una vez al guardar).
#   6. Guardar. Probar con boton derecho > Ejecutar, y confirmar que aparece
#      un archivo nuevo en data/raw/consensus_calendar/.
#
# Verificacion de que esta corriendo de verdad (no asumirlo):
#   - Revisar semanalmente que data/raw/consensus_calendar/ tenga un archivo
#     por cada dia (sin huecos). Un hueco silencioso es exactamente el
#     historico irrecuperable que este script existe para evitar.
# ---------------------------------------------------------------------------
