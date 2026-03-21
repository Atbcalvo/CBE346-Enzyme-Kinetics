# Inhibition analysis summary

## What was standardized
- All four inhibitor conditions were recomputed from raw plate CSVs using the same rate-rebuild mapping (rows A-C as controls, rows D-F as reaction triplicates) and the selected calibration slope from column 10 of `02_pNP_calibration_curve.csv`.
- Selected calibration slope for rate reconstruction: `9.029314 A/mM`.

## Primary inference strategy
- Apparent `Vmax`, `Km`, and `kcat` by inhibitor were fit in native `v` vs `[S]` space with uncertainty-aware nonlinear fitting.
- Report-facing rate outputs are now exported in `mM/min` to match the handout, while the internal fitting logic remains unit-consistent.
- Lineweaver-Burk was retained as a diagnostic view, not the primary estimator.
- Mechanism was compared with global nonlinear fits for competitive, uncompetitive, and mixed inhibition.

## Best current model
- Best model by AIC in the current export: `mixed`.
- Sensitivity check using average `[pNPP]` and `[Pi]` over the run also favored: `mixed`.
- Final report wording should still use `most consistent with` unless the model separation remains strong after you review the residuals and parameter uncertainty.

## Concentration-change reasonableness check
- Maximum estimated `[pNPP]` depletion over a run: `29.98%`.
- Maximum estimated increase in `[Pi]` relative to its starting inhibitor level: `16.95%`.
- Use `concentration_change_check.csv` and `global_model_avg_conc_comparison.csv` in the appendix to justify whether initial concentrations were adequate or whether average concentrations materially change the model ranking.

## Files to cite in the report draft
- `figures_export/rate_table_recomputed_from_raw.csv`
- `figures_export/mm_lb_summary.csv`
- `figures_export/global_model_comparison.csv`
- `figures_export/global_model_parameter_summary.csv`
- `figures_export/concentration_change_check.csv`
- `figures_export/global_model_avg_conc_comparison.csv`
- `figures_export/mm.png`, `lb.png`, `lbzoom.png`, `global_model_overlays.png`, `global_model_residuals.png`

## How to discuss nonlinear versus Lineweaver-Burk
- Use Lineweaver-Burk to show the geometric pattern as inhibitor changes.
- Explain that reciprocal transforms magnify low-rate points and distort error structure.
- State that the inhibition call is based primarily on the global nonlinear comparison, with LB used as a supporting visualization and not as the final estimator.

