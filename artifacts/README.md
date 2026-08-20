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

## Runbook completo de re-publicacion (verificado 2026-08-19)

Pasos exactos, en orden, para regenerar y publicar el artefacto **SPY M1** (el
publicado hoy). Ajusta el comando de la corrida para otra linea/spec.

```sh
# 1. Regenerar la corrida (NO toca latest/, escribe en runs/<run_id>/).
#    --no-capture: usa precios en cache (fresco <7d) + corpus GDELT en disco =>
#    reproduce el mismo asof y ajuste; solo cambian los fixes ya commiteados.
python scripts/run_v3.py --publish-m1 --no-capture      # SPY M1
#   variantes: --asset BTC  |  sin --publish-m1 = M2  |  --quick = provisional (R6 no)
# Anota el run_id que imprime al final: "Listo V3. run_id=<NEW>".

# 2. ARBOL DE GIT LIMPIO (contrato R5). promote.py exige `git status --porcelain`
#    VACIO (cuenta tracked + untracked). artifacts/ esta gitignored, asi que
#    publicar no ensucia git; el problema es cualquier OTRO cambio sin commitear.
#    Si el arbol tiene WIP que NO quieres commitear, usa un stash reversible:
git status --porcelain > /tmp/_pre.txt                  # snapshot para verificar
git stash push -u -m "wip-temporal-promover-<NEW>"      # guarda tracked + untracked
git status --porcelain                                   # debe salir VACIO

# 3. Promover atomicamente.
python scripts/promote.py artifacts/runs/<NEW>           # PROMOVIDO: run_id=<NEW>

# 4. Restaurar el WIP INMEDIATAMENTE y verificar que el arbol quedo identico.
git stash pop
diff <(git status --porcelain | sort) <(sort /tmp/_pre.txt) && echo "arbol OK"
git rev-parse HEAD                                        # HEAD no se movio

# 5. Re-exportar el panel publico (lee latest/ -> panel/public/data/).
python scripts/export_panel_data.py
#   Si la validacion formal (validation_v4.md) valida un run ANTERIOR, el
#   guardarrail F6 aborta. El modelo puede ser identico (p.ej. un fix que solo
#   toca presentacion): entonces re-exporta divulgando el desfase --
python scripts/export_panel_data.py --allow-stale-validation   # deja stale=true + ambos run_id

# 6. Verificar coherencia final (todo debe compartir <NEW>).
python - <<'PY'
import json
r=lambda p: json.load(open(p)).get("run_id")
for p in ("artifacts/latest/irfn.json","artifacts/latest/audit.json",
          "artifacts/latest/manifest.json","panel/public/data/irfn.json"):
    print(p, r(p))
PY
```

Notas:
- **No commitear WIP ajeno.** El artefacto queda en `latest/` (gitignored), no
  necesita commit. `git_commit` del artefacto = HEAD al momento de la corrida; el
  modelo publicado es reproducible desde ese commit aunque el arbol tenga cambios
  de UI/reportes sin commitear (no afectan al modelo).
- La resolucion **limpia** de F6 (evitar `stale=true`) es re-correr la validacion
  formal (`validation_v4.md`) sobre el run publicado antes del paso 5.

## Red de seguridad (CI / pre-commit)

`scripts/ci_check_latest.py` valida `artifacts/latest/` y falla ante violaciones
duras (mezcla, set incompleto, look-ahead, `nogit`, `--quick`). Instalar el hook:

```sh
cp scripts/git-hooks/pre-commit .git/hooks/pre-commit   # y chmod +x en Unix
```

Asi, un `latest/` corrupto (la regresion que estamos matando) bloquea el commit.
