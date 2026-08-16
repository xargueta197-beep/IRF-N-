# Diagnostico del regimen degenerado (absorbe-outliers) -- Fase 7

Muestra SPY 2013-01-02..2026-08-14 (3425 obs). Semilla 42. @diagnostic_only: no publica.

Un regimen es 'degenerado' (absorbe-outliers) si E[D]=1/(1-p_kk) < 2.0 dias: aparece un dia suelto para capturar un outlier y no persiste.

## 1) K=2 Normal (50 arranques, > R6)

| regimen | kappa | E[D] (dias) | % dias | vol anual emp | degenerado |
| --: | --: | --: | --: | --: | :--: |
| 0 | 0.9363 | 10.17 | 96.9% | 15.9% | no |
| 1 | 0.9954 | 1.17 | 3.1% | 21.6% | SI |

- loglik=-4078.46, convergieron al optimo 17/50 arranques. Regimen degenerado presente: **SI**.

## 2) K=2 Student-t (colas gordas)

| regimen | kappa | E[D] (dias) | % dias | vol anual emp | degenerado |
| --: | --: | --: | --: | --: | :--: |
| 0 | 0.9421 | 12.76 | 97.8% | 16.1% | no |
| 1 | 0.9972 | 1.10 | 2.2% | 20.0% | SI |

- loglik=-4075.41, convergieron al optimo 1/40 arranques. Regimen degenerado presente: **SI**.

## 3) K=3 Normal (regimen extra)

| regimen | kappa | E[D] (dias) | % dias | vol anual emp | degenerado |
| --: | --: | --: | --: | --: | :--: |
| 0 | 0.9642 | 1.51 | 62.2% | 13.3% | SI |
| 1 | 1.0000 | 1.00 | 2.7% | 14.3% | SI |
| 2 | 0.9925 | 1.38 | 35.2% | 18.6% | SI |

- loglik=-4054.39, convergieron al optimo 1/40 arranques. Regimen degenerado presente: **SI**.

## Conclusiones (con evidencia)

1. **No es un artefacto del multistart.** Con K=2 Normal y **50 arranques** (mas que
   el R6=30 publicado), el ganador sigue teniendo un regimen con E[D]=1.17 dias, y
   **17 de 50 arranques** convergen a ese mismo optimo. El estado absorbe-outliers
   esta en el **optimo global**, no en un maximo local en el que caeria solo un
   arranque desafortunado. Mas arranques NO lo eliminan.

2. **Student-t no lo corrige.** La hipotesis razonable era que las colas gordas
   absorbieran los outliers en la cola de la densidad en vez de en un regimen-pico.
   No ocurre: el segundo regimen sigue degenerado (E[D]=1.10 d, 2.2% de dias). La
   verosimilitud mejora un poco (-4075 vs -4078) pero la estructura de regimenes no
   cambia; ademas la superficie es mas dificil (1/40 arranques al optimo).

3. **K=3 no lo corrige limpiamente.** No libera al segundo regimen de su papel: los
   TRES regimenes salen con E[D]<2 (persistencia baja generalizada) y el ajuste esta
   mal identificado (1/40 arranques, superficie muy multimodal). Es cambiar una
   patologia por otra, no una mejora.

## Contraste con el baseline honesto (climatologia de regimen)

El baseline correcto es la **climatologia** (predecir siempre las frecuencias
marginales del proxy de regimen), calculada por `calibration.summarize` para el
propio modelo publicado -- **nunca** el baseline favorecedor de otra corrida (esa
era la trampa P1-5). Sobre el walk-forward del modelo publicado (M2, K=2 Normal con
TVTP tecnico, `run_id=7773faae4863`): **log-loss 0.243 vs climatologia 0.541** (19
bloques OOS). El modelo **gana** a su climatologia en densidad predictiva. El valor
del modelo (regimen de baja vol bien portado: E[D]=10-13 dias, ~97% de los dias, y
ganancia OOS sobre climatologia) es real y **separable** del regimen-pico.

## Recomendacion

**Reportar tal cual, con el caveat -- que es exactamente lo que la app ya hace**
(banner en 'Retornos condicionales': el estado absorbe-outliers dura ~1 dia y su
retorno condicional NO es un retorno esperado). Justificacion:

- El regimen degenerado es una **limitacion estructural conocida** del MS-GJR-GARCH
  K=2/Normal (ya documentada en CLAUDE.md): con una mezcla de gaussianas, el 2-3% de
  dias mas extremos se modela mejor con un componente de varianza alta y vida corta.
  Esta en el optimo global y captura ~3% de los dias.
- **NO reparametrizar ni restringir K de forma reactiva.** El diagnostico muestra que
  ni Student-t ni K=3 (tal como estan configurados) son una mejora limpia, y K=2 fue
  el elegido por BIC. Cualquier cambio de especificacion (p.ej. un piso de
  persistencia sobre p_kk, un componente explicito de saltos, o una mezcla de colas)
  es una **decision de metodologia del director** y exige re-correr walk-forward
  (R2) y multistart (R6), no un parche silencioso (regla del proyecto: la metodologia
  no se cambia por cuenta propia).
- El caveat en pantalla + los avisos del artefacto ya impiden que alguien lea el
  -139%/año del regimen-pico como una prediccion. Esa es la correccion honesta; el
  numero no se esconde, se explica.

*(Diagnostico reproducible: `scripts/diag_degenerate_regime.py`, @diagnostic_only,
no publica ni toca artifacts/latest/.)*
