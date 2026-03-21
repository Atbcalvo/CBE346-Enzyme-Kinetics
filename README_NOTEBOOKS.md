# CBE346 notebooks — review + local export

## Quick view of all figures (no Colab)

From the `data/` folder:

```bash
python3 export_cbe346_figures.py
```

Requires **Python 3** with **NumPy** and **Matplotlib** only (`pip install numpy matplotlib`). The notebooks on Colab typically also use **pandas** and **scipy**, but the local export script is intentionally self-contained.

This script is now the recommended **source of truth** for the current local analysis. It:

- rebuilds rates for all four inhibitor conditions from the raw plate CSVs in `original_csv/`
- uses a single selected pNP calibration slope across conditions
- exports report-facing rates in `mM/min` while keeping the internal fit logic unit-consistent
- exports apparent Michaelis-Menten fits and Lineweaver-Burk diagnostics
- compares global competitive, uncompetitive, and mixed inhibition models
- checks whether ignoring changing `[pNPP]` and accumulating `[Pi]` is reasonable, and reruns the global comparison using average concentrations
- writes report-facing summaries under `figures_export/`

Outputs go to **`data/figures_export/`**:

| File | What it is |
|------|------------|
| `rate_table_recomputed_from_raw.csv`, `rate_table_recomputed_tidy.csv` | Unified rate table rebuilt from the raw plate CSVs |
| `rate_table_comparison_vs_legacy.csv`, `rate_table_audit.png` | Audit of recomputed rates versus the older master table |
| `representative_pnp_traces.png` | Example baseline-corrected `[pNP](t)` traces used for slope extraction |
| `concentration_change_check.csv`, `concentration_change_summary.csv`, `concentration_change_summary.png` | First-pass check on `[pNPP]` depletion and `[Pi]` accumulation during the run |
| `mm.png`, `lb.png`, `lbzoom.png`, `mmcl.png`, `kmcl.png`, `kcatcl.png` | Apparent MM fits, LB diagnostics, and parameter uncertainty plots |
| `lb_slope_intercept_summary.csv`, `lb_slope_intercept_vs_I.png` | LB slope/intercept diagnostics by inhibitor concentration |
| `global_model_comparison.csv`, `global_model_parameter_summary.csv` | Global competitive / uncompetitive / mixed model comparison and parameter estimates |
| `global_model_avg_conc_comparison.csv`, `global_model_avg_conc_parameter_summary.csv` | Sensitivity analysis using average `[pNPP]` and `[Pi]` over the run |
| `global_model_overlays.png`, `global_model_residuals.png`, `inhibition_call_summary.txt` | Mechanism-comparison figures and summary ranking |
| `global_model_avg_conc_overlays.png`, `global_model_avg_conc_residuals.png`, `global_model_avg_conc_summary.txt` | Sensitivity-analysis figures and summary ranking |
| `mm_lb_summary.csv`, `ki_competitive_from_mm.csv` | Per-condition parameter summary plus competitive-only \(K_i\) back-calculation |
| `pNP_calibration_combined.png`, `pNP_calibration_summary.csv`, `pNP_calibration_slopes.txt` | Calibration plots and selected slope summary |
| `analysis_summary.md` | Short write-up note for how to discuss the results in the report |

Matplotlib cache (if needed) is written under `data/.mplconfig_cache/` (safe to delete).

---

## Notebook status

### `CBE346 Michaelis Menten Fit.ipynb`

- **Cells 0–2:** Google Colab + **gspread** — load from Google Sheets. **Won’t run locally** without auth and the same spreadsheet.
- **Cell 3:** Standalone **local CSV** pipeline (MM grid fit, Lineweaver-Burk, bootstrap CIs, plots).  
  - **Fixed:** `csv_path` now points to `Initial Velocity Data - Sheet1.completelyupdated.csv` (was `updated.csv`).
- **Cell 4:** Competitive \(K_i\) from MM \(K_m\) — run after cell 3.

**Recommendation:** Treat this notebook as exploratory or historical. For the current report workflow, prefer **`export_cbe346_figures.py`** and the files it writes into `figures_export/`.

### `CBE346 [pNP] Absorbance Calibration Curve.ipynb`

- **Cell 0:** Colab + gspread (same limitation).
- **Cells 1–5:** Expect in-memory `data` from the sheet; each cell is a replicate / inhibitor variant.

**Recommendation:** Use the calibration outputs from `export_cbe346_figures.py`, which now also writes a calibration summary CSV and records the selected slope used for rate reconstruction.

### `CBE346 Day 3 vmax Calculators.ipynb`

- **All cells:** Colab + gspread + **hard-coded column indices** (`col[46]`, `col[58]`, …) for one spreadsheet layout.
- **Cell 1 issue:** The same label `Experimental Data` is plotted three times — legend is misleading; should use distinct labels per trace.

**Recommendation:** Not portable. The current export script already rebuilds rates from the cleaned plate CSVs under `original_csv/` using one consistent local workflow.

---

## Data layout reminder

Velocity master table (no header): see `original_csv/README_DATASETS_USED.md` for column indices.
