# Reanudar el backfill de GDELT (M5) — target ~238-244 dias

**Decision del usuario (2026-07-30): acotar a ~238-244 dias totales** (descarto
el corpus completo 2017->hoy). Alcanza para AJUSTAR + inspeccionar Hawkes
(>=200 eventos), NO para la ablacion M5-vs-M4 en el walk-forward.

**Aviso para el futuro:** la ablacion M5 en walk-forward necesita ~7 anios de
cobertura continua (train_years=4 + 6x6m test, n_blocks_min=6 -> ~2555 dias).
Con ~238 dias solo se puede ajustar/inspeccionar el proceso de Hawkes y
documentar que la ablacion completa sigue pendiente por corpus (R8).

**Estado (2026-07-30 ~12:19):** ~64 dias capturados del corpus total (~3494 dias
2017-01-02..2026-07-29). Batch actual cubre 2017-01-02..2026-05-28 (3434 dias),
va desde 2026-05-28 hacia atras. **Es un maraton de DIAS de captura** por el
rate limit de la IP compartida (CGNAT) -> una IP dedicada/hotspot lo acortaria
mucho. Ver [[project_gdelt_network]].

## Que esta guardado (durable)
- `data/raw/headlines/*.json` — snapshots diarios inmutables (~64+).
- `_gdelt_checkpoints.jsonl` — rastro de checkpoints del rango actual (5%).
- `_gdelt_checkpoints_prev.jsonl` — checkpoints de rangos anteriores (archivado).
- `_gdelt_watchdog_state.json` / `_gdelt_watchdog_summary.md`.

## Como continuar (sesion nueva, desde irfn/, con el venv)
```
# 1) reanudar captura (target ~238-244; resumible, salta lo hecho, newest-first)
./.venv/Scripts/python.exe scripts/capture_headlines.py --max-days 180 > <LOG> 2>&1 &

# 2) relanzar el vigia de checkpoints al 5%
./.venv/Scripts/python.exe scripts/gdelt_watchdog.py <LOG> > <WD_LOG> 2>&1 &
```
Claude: re-armar el Monitor sobre `_gdelt_checkpoints.jsonl` para avisos al movil
en cada 5%. El vigia valida integridad de cada snapshot en cada hito (APROBADO/
REVISAR).

**GOTCHA de lanzamiento (Windows):** lanzar SIEMPRE via `cmd /c "...python... > LOG
2>&1"` (ASCII crudo). NO usar el redireccionamiento de PowerShell (`*>` / `>`): en
PS 5.1 escribe el log en UTF-16, y `gdelt_watchdog.py::_range_from_log` lo lee como
UTF-8 -> los digitos quedan separados por bytes nulos, el regex de fechas no encaja
y el vigia cae al rango de respaldo hardcoded (388 dias, mal). Si el vigia arranca
con un rango distinto al del lote, matarlo, borrar el log, y relanzar via cmd.
Para reanudar el MISMO lote y que el vigia continue desde su ultimo decil, relanzar
la captura con `--since/--until` del rango del lote (ver `_gdelt_watchdog_summary.md`),
no con `--max-days` (que reescribe el rango a los N faltantes mas recientes).

## Umbrales / notas
- `min_events_fit: 200` — minimo para intentar el MLE de Hawkes (se supera con
  pocos meses; ya deberia estar cubierto).
- Para la ABLACION (M5 vs M4) se necesita el span de ~7 anios (arriba).
- Cuello de botella = rate limit de GDELT sobre IP compartida, NO CPU.
