"""Contrato del artefacto publicable: un directorio candidato solo es promovible
a `artifacts/latest/` si pasa TODAS las reglas de este modulo.

Esto es lo que impide que vuelva la regresion de publicacion V0 (auditoria
2026-08-15): la promocion deja de ser "copiar pieza por pieza" y pasa por
`validate_artifact`, que rechaza mezclas de procedencia, sets incompletos,
huecos de frescura, corridas que no cumplen R6 y artefactos sin commit.

Uso CLI (nota de layout: en este repo el paquete es `irfn`, asi que el modulo
es `irfn.outputs.contract`, no `outputs.contract`):

    python -m irfn.outputs.contract artifacts/quarantine/2026-08-15_v0_regression/
    python -m irfn.outputs.contract artifacts/runs/<run_id> --repo-root .

Codigo de salida 0 solo si el directorio es PROMOVIBLE; 1 en cualquier otro caso
(con las violaciones enumeradas en stdout).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

# ---------------------------------------------------------------------------
# Parametros del contrato (cero constantes magicas dispersas: viven aqui).
# ---------------------------------------------------------------------------

# Rango de multistart que exige R6 (CLAUDE.md): 20-50 arranques.
R6_MULTISTART_MIN = 20
R6_MULTISTART_MAX = 50

# Frescura por defecto: asof (ultimo dato usado) no puede quedar mas de esto por
# detras de generated_at (cuando se genero el artefacto). 5 tolera fin de semana
# largo/feriados. NO se mide asof-vs-history_end: la serie OOS termina antes que
# asof por la cola natural del walk-forward (el ultimo tramo no forma un bloque de
# test completo), y eso es legitimo; la mezcla vieja-historia/nuevo-asof la impide
# la atomicidad (R1: manifest + un solo run_id).
DEFAULT_TOLERANCE_DAYS = 5

# Archivo que jamas es "ajeno" (marcador de carpeta versionada).
_ALWAYS_ALLOWED = {".gitkeep"}


def _version_rank(version: str | None) -> int:
    """'V0'->0, 'V3'->3. Desconocido -> -1 (se reporta como violacion aparte)."""
    if not version or not version.upper().startswith("V"):
        return -1
    try:
        return int(version.upper()[1:])
    except ValueError:
        return -1


def _expected_files(rank: int, news_active: bool, hawkes_active: bool) -> tuple[set[str], set[str]]:
    """(requeridos, permitidos) para una version dada.

    - requeridos: deben estar presentes; su ausencia es 'set incompleto'.
    - permitidos: superconjunto de requeridos; cualquier archivo fuera de aqui
      es 'archivo ajeno' (procedencia mezclada). Esto atrapa v1_ablation.json,
      v1_kselect.json, walkforward_v1.json y los parquets Hawkes huerfanos en un
      artefacto V0.
    """
    required = {"irfn.json", "audit.json", "walkforward.json", "history.parquet", "manifest.json"}
    allowed = set(required) | set(_ALWAYS_ALLOWED)

    if rank >= 2:  # V2+: capa de sorpresa cableada (aunque este inactiva)
        required.add("surprise_events.json")
        allowed.add("surprise_events.json")
        allowed.add("surprise_history.parquet")
        if news_active:
            required.add("surprise_history.parquet")

    if rank >= 3:  # V3+: capa Hawkes
        allowed.add("hawkes_history.parquet")
        allowed.add("headline_rug.parquet")
        # obligatorios SOLO si la capa Hawkes esta activa (espejo de
        # surprise_history): un V3 con Hawkes inactivo por falta de corpus es
        # legitimo y no debe bloquearse por unos parquets que no existen. En V0
        # (rank 0) estos archivos NO estan en `allowed` => siguen siendo ajenos.
        if hawkes_active:
            required.add("hawkes_history.parquet")
            required.add("headline_rug.parquet")

    if rank >= 4:  # V4: walk-forward economico (opcional, no requerido)
        allowed.add("validation.json")

    return required, allowed


@dataclass
class ContractResult:
    """Resultado de validar un directorio candidato.

    - violations: fallas DURAS. Con una sola, el artefacto no existe (no se
      escribe, no se promueve).
    - provisional_reasons: motivos por los que el artefacto es 'provisional'
      (p.ej. R6 sin cumplir bajo --allow-provisional): valido para inspeccion
      pero PROHIBIDO promover a latest/.
    - version / run_id: leidos del irfn.json, para el log de publicacion.
    """
    violations: list[str] = field(default_factory=list)
    provisional_reasons: list[str] = field(default_factory=list)
    version: str | None = None
    run_id: str | None = None

    @property
    def ok(self) -> bool:
        """Sin fallas duras (puede seguir siendo no-promovible si es provisional)."""
        return not self.violations

    @property
    def promotable(self) -> bool:
        """Promovible a latest/ solo si no hay fallas duras NI marca provisional."""
        return not self.violations and not self.provisional_reasons


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def _load_json(p: Path) -> dict | None:
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _working_tree_dirty(repo_root: Path) -> bool | None:
    """True si hay cambios sin commitear; None si no es repo git o git falla."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], cwd=str(repo_root),
            capture_output=True, text=True, check=True,
        )
    except Exception:
        return None
    return bool(out.stdout.strip())


def validate_artifact(
    candidate_dir: Path,
    *,
    tolerance_days: int = DEFAULT_TOLERANCE_DAYS,
    allow_provisional: bool = False,
    repo_root: Path | None = None,
) -> ContractResult:
    """Valida un directorio candidato contra el contrato. NO escribe nada.

    Reglas (numeradas segun la auditoria):
      1. Procedencia unica: run_id coherente y cero archivos ajenos.
      2. Completitud: exactamente el set que exige la version.
      3. Coherencia temporal: asof - max(history.fecha) <= tolerance_days.
      4. R6: multistart en [20, 50] (duro; provisional bajo allow_provisional).
      5. Reproducibilidad: git_commit != 'nogit' (+ arbol limpio si repo_root).
      6. Coherencia de version: V1+ => tvtp; capa activa => covariables no vacias.
      7. Un solo walk-forward: walkforward.json si, walkforward_v1.json no.
    """
    res = ContractResult()
    d = Path(candidate_dir)
    if not d.is_dir():
        res.violations.append(f"El candidato no es un directorio: {d}")
        return res

    irfn = _load_json(d / "irfn.json")
    if irfn is None:
        res.violations.append("Falta o es ilegible irfn.json: sin el no hay artefacto.")
        return res

    res.version = irfn.get("version")
    res.run_id = irfn.get("run_id")
    rank = _version_rank(res.version)
    if rank < 0:
        res.violations.append(f"Version no reconocida en irfn.json: {res.version!r}.")

    model = irfn.get("model", {}) or {}
    news_active = bool((model.get("news_layer_params") or {}).get("active"))
    hawkes_active = bool((model.get("hawkes_layer_params") or {}).get("active"))

    present = {p.name for p in d.iterdir() if p.is_file()}
    required, allowed = _expected_files(max(rank, 0), news_active, hawkes_active)

    # --- Regla 1: procedencia unica ---
    manifest = _load_json(d / "manifest.json")
    if manifest is None:
        res.violations.append(
            "R1 procedencia: falta manifest.json (lista de archivos + sha256 + "
            "run_id de la corrida). Sin el no se puede afirmar procedencia unica.")
    else:
        m_run = manifest.get("run_id")
        if m_run != res.run_id:
            res.violations.append(
                f"R1 procedencia: manifest.run_id ({m_run}) != irfn.run_id ({res.run_id}).")
        # sha de cada archivo declarado debe coincidir con el de disco
        declared = manifest.get("files", {}) or {}
        for name, meta in declared.items():
            fp = d / name
            if not fp.exists():
                res.violations.append(f"R1 procedencia: manifest declara {name} pero no esta en disco.")
                continue
            want = (meta or {}).get("sha256")
            if want and _sha256(fp) != want:
                res.violations.append(f"R1 procedencia: sha256 de {name} no coincide con el manifiesto.")

    # run_id cruzado entre los JSON que lo llevan (irfn, audit)
    audit = _load_json(d / "audit.json")
    if audit is not None and audit.get("run_id") not in (None, res.run_id):
        res.violations.append(
            f"R1 procedencia: audit.run_id ({audit.get('run_id')}) != irfn.run_id ({res.run_id}).")

    # archivos ajenos (no permitidos para esta version)
    foreign = sorted(present - allowed)
    if foreign:
        res.violations.append(
            "R1 procedencia: archivos ajenos para " + (res.version or "?") +
            " (procedencia mezclada): " + ", ".join(foreign) + ".")

    # --- Regla 2: completitud ---
    missing = sorted(required - present)
    if missing:
        res.violations.append(
            "R2 completitud: faltan archivos que exige " + (res.version or "?") +
            ": " + ", ".join(missing) + ".")
    # V0 (rank 0) NO puede traer los parquets de Hawkes
    if rank == 0:
        forbidden_v0 = sorted({"hawkes_history.parquet", "headline_rug.parquet"} & present)
        if forbidden_v0:
            res.violations.append(
                "R2 completitud: V0 no puede contener capa Hawkes; presentes: " +
                ", ".join(forbidden_v0) + ".")

    # --- Regla 3: coherencia temporal ---
    # (a) look-ahead: history NUNCA puede terminar DESPUES de asof (seria futuro
    #     en la serie OOS). Que termine antes es la cola normal del walk-forward.
    # (b) frescura: asof no puede quedar muy por detras de generated_at (datos
    #     rancios al generar). La mezcla vieja-historia/nuevo-asof la impide R1.
    asof_raw = irfn.get("asof")
    gen_raw = irfn.get("generated_at")
    hist_path = d / "history.parquet"
    asof = None
    if asof_raw:
        try:
            asof = date.fromisoformat(str(asof_raw)[:10])
        except ValueError:
            res.violations.append(f"R3 temporal: asof no parseable ({asof_raw!r}).")
    if asof is not None and hist_path.exists():
        try:
            import pandas as pd
            hist = pd.read_parquet(hist_path, columns=["fecha"])
            last = pd.to_datetime(hist["fecha"]).max().date()
            if last > asof:
                res.violations.append(
                    f"R3 look-ahead: history llega a {last}, DESPUES de asof ({asof}): "
                    "hay futuro en la serie OOS.")
        except Exception as e:  # pragma: no cover - defensivo
            res.violations.append(f"R3 temporal: no se pudo leer history ({e}).")
    if asof is not None and gen_raw:
        try:
            gen = date.fromisoformat(str(gen_raw)[:10])
            lag = (gen - asof).days
            if lag > tolerance_days:
                res.violations.append(
                    f"R3 frescura: asof ({asof}) queda {lag} dias por detras de "
                    f"generated_at ({gen}) (> tolerancia {tolerance_days}): datos rancios.")
        except ValueError:
            pass

    # --- Regla 4: R6 multistart ---
    n_ms = ((model.get("estimation") or {}).get("multistart"))
    if not isinstance(n_ms, int) or not (R6_MULTISTART_MIN <= n_ms <= R6_MULTISTART_MAX):
        msg = (f"R6 multistart: {n_ms} fuera de [{R6_MULTISTART_MIN}, {R6_MULTISTART_MAX}] "
               "(corrida --quick / provisional).")
        if allow_provisional:
            res.provisional_reasons.append(msg)
        else:
            res.violations.append(msg)

    # --- Regla 5: reproducibilidad ---
    git_commit = irfn.get("git_commit")
    if not git_commit or git_commit == "nogit":
        res.violations.append(
            f"R5 reproducibilidad: git_commit={git_commit!r}; run_id = hash(config+commit+"
            "semilla) no es reproducible sin un commit real.")
    if repo_root is not None:
        dirty = _working_tree_dirty(Path(repo_root))
        if dirty:
            res.violations.append(
                "R5 reproducibilidad: el arbol de trabajo tiene cambios sin commitear; "
                "el artefacto no corresponde a un commit limpio.")

    # --- Regla 6: coherencia de version ---
    # Con K=1 NO hay logit de transicion: tvtp=false es correcto por construccion, y
    # una capa activa (Hawkes) se publica como indicador STANDALONE, no como covariable
    # del logit. Por eso las tres sub-reglas solo aplican con K>=2.
    K = int(model.get("K") or 2)
    tvtp = bool(model.get("tvtp"))
    covariates = model.get("covariates") or []
    news_layer = model.get("news_layer") or []
    if rank >= 1 and K >= 2 and not tvtp:
        res.violations.append("R6 version: con K>=2, V1+ exige tvtp=true; el artefacto trae tvtp=false.")
    if K >= 2 and news_active and not covariates:
        res.violations.append(
            "R6 version: surprise_index activa en el logit pero covariates esta vacio.")
    if K >= 2 and news_active and not news_layer:
        res.violations.append(
            "R6 version: news_layer_params.active=true pero model.news_layer esta vacio.")

    # --- Regla 7: un solo walk-forward ---
    wf_variants = sorted(n for n in present if n.startswith("walkforward") and n != "walkforward.json")
    if wf_variants:
        res.violations.append(
            "R7 walk-forward: existen variantes versionadas ademas de walkforward.json: " +
            ", ".join(wf_variants) + " (la pantalla de Validacion no debe poder elegir la favorecedora).")
    if "walkforward.json" not in present:
        res.violations.append("R7 walk-forward: falta walkforward.json.")

    return res


def assert_promotable(candidate_dir: Path, **kwargs) -> ContractResult:
    """Como validate_artifact pero LANZA si el directorio no es promovible.
    Pensado para que publish() lo llame antes del swap atomico (Fase 3)."""
    res = validate_artifact(candidate_dir, **kwargs)
    if not res.promotable:
        detail = "\n".join(
            ["  - " + v for v in res.violations] +
            ["  - (provisional) " + p for p in res.provisional_reasons])
        raise ContractViolation(
            f"Artefacto NO promovible ({candidate_dir}):\n{detail}")
    return res


class ContractViolation(Exception):
    """Se lanza cuando un directorio candidato no cumple el contrato de promocion."""


def _format(res: ContractResult) -> str:
    lines = [f"Artefacto: version={res.version} run_id={res.run_id}"]
    if res.violations:
        lines.append(f"VIOLACIONES ({len(res.violations)}):")
        lines += [f"  - {v}" for v in res.violations]
    if res.provisional_reasons:
        lines.append(f"PROVISIONAL ({len(res.provisional_reasons)}):")
        lines += [f"  - {p}" for p in res.provisional_reasons]
    veredicto = ("PROMOVIBLE" if res.promotable else
                 ("VALIDO-NO-PROMOVIBLE (provisional)" if res.ok else "RECHAZADO"))
    lines.append(f"Veredicto: {veredicto}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Valida un directorio de artefacto contra el contrato de publicacion.")
    ap.add_argument("candidate_dir", type=Path)
    ap.add_argument("--tolerance-days", type=int, default=DEFAULT_TOLERANCE_DAYS)
    ap.add_argument("--allow-provisional", action="store_true",
                    help="Trata R6 como marca provisional en vez de falla dura (sigue prohibida la promocion).")
    ap.add_argument("--repo-root", type=Path, default=None,
                    help="Si se pasa, verifica ademas que el arbol git este limpio (R5).")
    args = ap.parse_args(argv)

    res = validate_artifact(
        args.candidate_dir, tolerance_days=args.tolerance_days,
        allow_provisional=args.allow_provisional, repo_root=args.repo_root)
    print(_format(res))
    return 0 if res.promotable else 1


if __name__ == "__main__":
    raise SystemExit(main())
