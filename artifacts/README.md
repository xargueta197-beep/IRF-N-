# artifacts/ — artefactos del pipeline IRF-N

Regla de oro (tras la remediacion de publicacion, 2026-08-16): **`latest/` solo se
escribe por promocion atomica**. Ninguna corrida escribe `latest/` directamente.

## Directorios

| Dir | Que es | Quien escribe |
| :-- | :-- | :-- |
| `runs/<run_id>/` | Salida **inmutable** de una corrida: el set completo (`irfn.json`, `audit.json`, `walkforward.json`, `history.parquet`, y en V3 `hawkes_history.parquet`, `headline_rug.parquet`, `surprise_events.json`) + `manifest.json`. | `run_pipeline.py`, `run_v2.py`, `run_v3.py` (cada una en SU `runs/<run_id>/`). |
| `latest/` | El artefacto **publicado** que lee la app (R9). Un unico `run_id`, con `manifest.json`. | **Solo** `promote_run` (via `scripts/promote.py`), con swap atomico. |
| `analysis/` | Analisis V1 no publicable (`v1_kselect.json`, `v1_ablation.json`, `walkforward_v1.json`). Nunca entra a `latest/`. | `run_v1.py`. |
| `checkpoints/` | Checkpoints reanudables del walk-forward (por bloque, con firma de config). | los orquestadores. |
| `quarantine/` | Respaldos de estados rotos congelados para auditoria. | manual. |
| `btc/` | Misma estructura para la segunda linea de activo (BTC). | orquestadores con `--asset BTC`. |
| `publish_log.jsonl` | Registro de cada promocion (timestamp, run_id, version, forzado, quien). | `promote_run`. |

`runs/`, `latest/`, `checkpoints/`, `analysis/`, `quarantine/`, `btc/` y
`publish_log.jsonl` estan en `.gitignore` (son datos generados). Este README y los
`.gitkeep` si se versionan.

## Como se publica (unico camino)

1. Una corrida escribe su set completo en `artifacts/runs/<run_id>/` (nunca `latest/`).
2. Se promueve:

   ```sh
   python scripts/promote.py artifacts/runs/<run_id>
   # V<3 o corridas provisionales (--quick, R6 sin cumplir) requieren --force-downgrade
   python scripts/promote.py artifacts/runs/<run_id> --slug btc   # linea BTC
   ```

3. `promote_run` (en `src/irfn/outputs/publish.py`):
   - escribe `manifest.json` (run_id + version + sha256 por archivo),
   - corre el **contrato** (`src/irfn/outputs/contract.py`): procedencia unica, set
     completo por version, no look-ahead en la serie, `git_commit` real, R6
     multistart 20-50, frescura de `asof`, un solo `walkforward.json`,
   - aplica el **guardarrail** anti-downgrade (version < V3 o provisional => bloqueado
     salvo `--force-downgrade`, que queda en `publish_log.jsonl`),
   - hace el **swap atomico** (dos renames de destino inexistente + recuperacion desde
     `.latest.trash.*` si el proceso muere a la mitad): `latest/` nunca queda en un
     estado intermedio ni mezclado.

## Red de seguridad (CI / pre-commit)

`scripts/ci_check_latest.py` valida `artifacts/latest/` y falla ante violaciones
duras (mezcla, set incompleto, look-ahead, `nogit`, `--quick`). Instalar el hook:

```sh
cp scripts/git-hooks/pre-commit .git/hooks/pre-commit   # y chmod +x en Unix
```

Asi, un `latest/` corrupto (la regresion que estamos matando) bloquea el commit.
