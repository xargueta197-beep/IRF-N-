# Ablacion de la capa de noticias — NO APLICA (activo K=1)

generado: 2026-08-21T03:11:20.058751+00:00  |  run_id: `d7dca6e40eeb`  |  K=1, dist=t

El activo se selecciono como **K=1** (un solo regimen) en V1 (BIC + bootstrap LR + diagnostico de estabilidad 2026-08-15, `reports/diag_btc_k2t_stability.md`). Sin regimenes NO hay matriz de transicion que modular: la escalera M0..M5 y el diagnostico M2+H no aplican (no hay transiciones sobre las que medir el aporte de covariables tecnicas, macro, sorpresa o lambda_N_z).

- M3 (macro): ALFRED_API_KEY configurada, pero la ablacion M3 con L1 no consta (reports/ablation_m3_l1_btc.json ausente); correr scripts/run_m3_l1.py para el veredicto macro. El bloqueante real de M5 vs M4 sigue siendo M4 (sorpresa), no la macro.
- El branching ratio de Hawkes (indicador de fragilidad) SI se publica y es independiente de K; ver `reports/validation_v3_btc.md`.
