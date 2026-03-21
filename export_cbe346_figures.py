#!/usr/bin/env python3
"""
Export a reproducible local analysis for the CBE346 enzyme-kinetics dataset.

This script is intended to be the single local source of truth for:
  - pNP calibration summaries
  - raw-plate rate recomputation
  - apparent Michaelis-Menten fits by inhibitor level
  - Lineweaver-Burk diagnostic plots
  - global inhibition-model comparison (competitive / uncompetitive / mixed)
  - uncertainty-aware summary tables and figures for report writing
"""
from __future__ import annotations

import csv
import os
import re
import sys
from pathlib import Path

# Writable matplotlib config (avoids slow cache rebuild in sandbox / read-only home)
_MPL_DIR = Path(__file__).resolve().parent / ".mplconfig_cache"
_MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------
# Paths and dataset constants
# -----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
OUT = HERE / "figures_export"
OUT.mkdir(parents=True, exist_ok=True)

ORIG_DIR = HERE / "original_csv"
LEGACY_VELOCITY_CSV = HERE / "Initial Velocity Data - Sheet1.completelyupdated.csv"
LEGACY_VELOCITY_COPY = ORIG_DIR / "00_Initial_Velocity_Data_finale.csv"
CALIB_CSV = ORIG_DIR / "02_pNP_calibration_curve.csv"

RATE_CALIBRATION_COLUMN = 10
BASELINE_SUB = 0.05679
ENZYME_STOCK_UG_PER_ML = 50.0
ENZYME_ADDED_UL = 10.0
WELL_VOLUME_UL = 200.0
ENZYME_MW_G_PER_MOL = 150000.0
ET_MILLIMOLAR = (
    (ENZYME_STOCK_UG_PER_ML * 1e-6 * 1000.0) * (ENZYME_ADDED_UL / WELL_VOLUME_UL) / ENZYME_MW_G_PER_MOL
) * 1000.0
BOOTSTRAP_SEED = 346
SECONDS_PER_MINUTE = 60.0

CONDITION_SPECS = [
    {
        "label": "0 mM inhibitor",
        "I_mM": 0.0,
        "legacy_v_col": 2,
        "legacy_sd_col": 3,
        "plate_file": ORIG_DIR / "01_plate_0mM_No_Inhibitor.csv",
        "color": "tab:red",
    },
    {
        "label": "1 mM inhibitor",
        "I_mM": 1.0,
        "legacy_v_col": 5,
        "legacy_sd_col": 6,
        "plate_file": ORIG_DIR / "01_plate_2mM_Inhibitor.csv",
        "color": "tab:green",
    },
    {
        "label": "5 mM inhibitor",
        "I_mM": 5.0,
        "legacy_v_col": 7,
        "legacy_sd_col": 8,
        "plate_file": ORIG_DIR / "01_plate_10mM_3-4-26.cleaned.csv",
        "color": "tab:blue",
    },
    {
        "label": "10 mM inhibitor",
        "I_mM": 10.0,
        "legacy_v_col": 9,
        "legacy_sd_col": 10,
        "plate_file": ORIG_DIR / "01_plate_20mM_3-4-26.cleaned.csv",
        "color": "tab:purple",
    },
]


def inhibitor_conc_mM(label: str) -> float:
    m = re.match(r"\s*([0-9]+\.?[0-9]*)", label)
    return float(m.group(1)) if m else float("nan")


def write_csv_dicts(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def per_min(value: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(value) * SECONDS_PER_MINUTE if isinstance(value, np.ndarray) else float(value) * SECONDS_PER_MINUTE


def positive_sigma(values: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    sigma = np.asarray(sigma, dtype=float).copy()
    finite_pos = sigma[np.isfinite(sigma) & (sigma > 0)]
    fallback = float(np.median(finite_pos)) if finite_pos.size else max(float(np.nanmax(values)) * 0.05, 1e-8)
    sigma[~np.isfinite(sigma) | (sigma <= 0)] = fallback
    return sigma


def through_origin_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    denom = float(np.dot(x, x))
    return float(np.dot(x, y) / denom) if denom > 0 else float("nan")


def linear_fit(x: np.ndarray, y: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    p = np.polyfit(x, y, 1)
    slope = float(p[0])
    intercept = float(p[1])
    yhat = slope * x + intercept
    sse = float(np.sum((y - yhat) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    return {"slope": slope, "intercept": intercept, "sse": sse, "r2": r2, "yhat": yhat}


def load_master_layout() -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    for path in (LEGACY_VELOCITY_CSV, LEGACY_VELOCITY_COPY):
        if path.is_file():
            arr = np.loadtxt(path, delimiter=",")
            return arr[:, 1].astype(float), arr[:, 0].astype(float), arr
    s_default = np.array([10.0, 7.5, 5.0, 3.75, 2.5, 2.0, 1.25, 0.625, 0.125, 0.0], dtype=float)
    return s_default, 2.0 * s_default, None


def export_calibration_figures() -> dict:
    if not CALIB_CSV.is_file():
        raise FileNotFoundError(f"Missing calibration CSV: {CALIB_CSV}")

    data = np.loadtxt(CALIB_CSV, delimiter=",")
    pnp = data[:, 0].astype(float)
    replicates = [
        ("rep1_day3_style", 4, 2, 5),
        ("rep2_1mM", 6, 2, 7),
        ("rep3_5mM", 8, 2, 9),
        ("rep4_10mM", 10, 2, 11),
    ]

    rows = []
    plt.figure(figsize=(12, 6))
    plt.title("[pNP] calibration used for local analysis", fontsize=16)
    plt.xlabel("[pNP] (mM)", fontsize=13)
    plt.ylabel("Absorbance at 405 nm", fontsize=13)
    plt.grid(True, alpha=0.3)

    xfit = np.linspace(0.0, max(float(np.nanmax(pnp)), 0.22), 200)
    colors = ["C0", "C1", "C2", "C3"]
    for i, (name, abs_col, xerr_col, yerr_col) in enumerate(replicates):
        if data.shape[1] <= max(abs_col, xerr_col, yerr_col):
            continue
        y_raw = data[:, abs_col].astype(float)
        y_blank_sub = y_raw - BASELINE_SUB
        xerr = data[:, xerr_col].astype(float)
        yerr = data[:, yerr_col].astype(float)
        slope_raw = through_origin_slope(pnp, y_raw)
        slope_blank_sub = through_origin_slope(pnp, y_blank_sub)
        rows.append(
            {
                "replicate": name,
                "abs_col": abs_col,
                "slope_raw_A_per_mM": slope_raw,
                "slope_blank_subtracted_A_per_mM": slope_blank_sub,
                "used_for_rate_rebuild": abs_col == RATE_CALIBRATION_COLUMN,
            }
        )
        c = colors[i % len(colors)]
        label = f"{name} raw fit (slope={slope_raw:.4f})"
        if abs_col == RATE_CALIBRATION_COLUMN:
            label += " [selected]"
        plt.errorbar(
            pnp,
            y_raw,
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            color=c,
            ecolor="black",
            capsize=3,
            label=label,
        )
        plt.plot(xfit, slope_raw * xfit, "-", color=c, alpha=0.85)

    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT / "pNP_calibration_combined.png", dpi=200)
    plt.close()

    write_csv_dicts(
        OUT / "pNP_calibration_summary.csv",
        rows,
        ["replicate", "abs_col", "slope_raw_A_per_mM", "slope_blank_subtracted_A_per_mM", "used_for_rate_rebuild"],
    )

    selected = next((r for r in rows if r["used_for_rate_rebuild"]), None)
    selected_slope = float(selected["slope_raw_A_per_mM"]) if selected else float("nan")
    with (OUT / "pNP_calibration_slopes.txt").open("w") as f:
        for row in rows:
            f.write(
                f"{row['replicate']}\traw={row['slope_raw_A_per_mM']:.6f}\tblank_sub={row['slope_blank_subtracted_A_per_mM']:.6f}\n"
            )
        f.write(f"\nselected_rate_rebuild_column={RATE_CALIBRATION_COLUMN}\n")
        f.write(f"selected_rate_rebuild_slope_raw_A_per_mM={selected_slope:.6f}\n")

    return {
        "rows": rows,
        "selected_rate_slope": selected_slope,
    }


def recompute_plate_rates(plate_path: Path, slope_a_per_mM: float, s_all: np.ndarray) -> tuple[list[dict], dict]:
    if not plate_path.is_file():
        raise FileNotFoundError(f"Missing plate CSV: {plate_path}")

    arr = np.loadtxt(plate_path, delimiter=",")
    time_s = arr[:, 0].astype(float)
    rows = []
    trace_store: dict[int, dict] = {}

    for k, substrate_mM in enumerate(s_all):
        ctrl_indices = [2 + k, 14 + k, 26 + k]
        rxn_indices = [38 + k, 50 + k, 62 + k]
        replicate_rates = []
        replicate_r2 = []
        replicate_pnp = []
        replicate_fits = []

        for ctrl_idx, rxn_idx in zip(ctrl_indices, rxn_indices):
            corrected_abs = arr[:, rxn_idx].astype(float) - arr[:, ctrl_idx].astype(float)
            pnp_mM = corrected_abs / slope_a_per_mM
            fit = linear_fit(time_s, pnp_mM)
            replicate_rates.append(fit["slope"])
            replicate_r2.append(fit["r2"])
            replicate_pnp.append(pnp_mM)
            replicate_fits.append(fit["yhat"])

        replicate_rates_arr = np.asarray(replicate_rates, dtype=float)
        rate_sd = float(np.std(replicate_rates_arr, ddof=1)) if replicate_rates_arr.size > 1 else 0.0
        rows.append(
            {
                "final_S_mM": float(substrate_mM),
                "v_mean_mM_per_s": float(np.mean(replicate_rates_arr)),
                "v_sd_mM_per_s": rate_sd,
                "mean_trace_r2": float(np.mean(replicate_r2)),
                "n_replicates": 3,
            }
        )
        trace_store[k] = {
            "time_s": time_s,
            "replicate_pnp_mM": np.asarray(replicate_pnp, dtype=float),
            "replicate_fit_mM": np.asarray(replicate_fits, dtype=float),
            "replicate_rates_mM_per_s": replicate_rates_arr,
            "mean_pnp_mM": np.mean(np.asarray(replicate_pnp, dtype=float), axis=0),
            "sd_pnp_mM": np.std(np.asarray(replicate_pnp, dtype=float), axis=0, ddof=1),
        }

    return rows, trace_store


def build_analysis_dataset(calibration: dict) -> dict:
    s_all, stock_labels, legacy_arr = load_master_layout()
    selected_slope = float(calibration["selected_rate_slope"])
    wide_rows = [{"stock_S_label": float(stock_labels[i]), "final_S_mM": float(s_all[i])} for i in range(len(s_all))]
    tidy_rows = []
    comparison_rows = []
    trace_store = {}
    series = {}

    for spec in CONDITION_SPECS:
        recomputed_rows, traces = recompute_plate_rates(spec["plate_file"], selected_slope, s_all)
        trace_store[spec["label"]] = traces
        v = np.array([r["v_mean_mM_per_s"] for r in recomputed_rows], dtype=float)
        sd = np.array([r["v_sd_mM_per_s"] for r in recomputed_rows], dtype=float)
        series[spec["label"]] = {
            "v": v,
            "sd": sd,
            "I_mM": float(spec["I_mM"]),
            "color": spec["color"],
            "source": "recomputed_from_raw_plate_csv",
        }

        for i, row in enumerate(recomputed_rows):
            wide_rows[i][f"{int(spec['I_mM'])}mM_v_mean_mM_per_s"] = row["v_mean_mM_per_s"]
            wide_rows[i][f"{int(spec['I_mM'])}mM_v_sd_mM_per_s"] = row["v_sd_mM_per_s"]
            wide_rows[i][f"{int(spec['I_mM'])}mM_v_mean_mM_per_min"] = per_min(row["v_mean_mM_per_s"])
            wide_rows[i][f"{int(spec['I_mM'])}mM_v_sd_mM_per_min"] = per_min(row["v_sd_mM_per_s"])
            tidy_rows.append(
                {
                    "condition": spec["label"],
                    "I_mM": spec["I_mM"],
                    "final_S_mM": row["final_S_mM"],
                    "v_mean_mM_per_s": row["v_mean_mM_per_s"],
                    "v_sd_mM_per_s": row["v_sd_mM_per_s"],
                    "v_mean_mM_per_min": per_min(row["v_mean_mM_per_s"]),
                    "v_sd_mM_per_min": per_min(row["v_sd_mM_per_s"]),
                    "mean_trace_r2": row["mean_trace_r2"],
                    "n_replicates": row["n_replicates"],
                }
            )

        if legacy_arr is not None:
            legacy_v = legacy_arr[:, spec["legacy_v_col"]].astype(float)
            legacy_sd = legacy_arr[:, spec["legacy_sd_col"]].astype(float)
            for i, row in enumerate(recomputed_rows):
                comparison_rows.append(
                    {
                        "condition": spec["label"],
                        "final_S_mM": row["final_S_mM"],
                        "recomputed_v_mM_per_s": row["v_mean_mM_per_s"],
                        "recomputed_v_mM_per_min": per_min(row["v_mean_mM_per_s"]),
                        "legacy_v_mM_per_s": float(legacy_v[i]),
                        "legacy_v_mM_per_min": per_min(float(legacy_v[i])),
                        "delta_v_mM_per_s": row["v_mean_mM_per_s"] - float(legacy_v[i]),
                        "delta_v_mM_per_min": per_min(row["v_mean_mM_per_s"] - float(legacy_v[i])),
                        "relative_delta_v_pct": 100.0
                        * (row["v_mean_mM_per_s"] - float(legacy_v[i]))
                        / max(abs(float(legacy_v[i])), 1e-12),
                        "recomputed_sd_mM_per_s": row["v_sd_mM_per_s"],
                        "recomputed_sd_mM_per_min": per_min(row["v_sd_mM_per_s"]),
                        "legacy_sd_mM_per_s": float(legacy_sd[i]),
                        "legacy_sd_mM_per_min": per_min(float(legacy_sd[i])),
                    }
                )

    write_csv_dicts(
        OUT / "rate_table_recomputed_from_raw.csv",
        wide_rows,
        list(wide_rows[0].keys()),
    )
    write_csv_dicts(
        OUT / "rate_table_recomputed_tidy.csv",
        tidy_rows,
        [
            "condition",
            "I_mM",
            "final_S_mM",
            "v_mean_mM_per_s",
            "v_sd_mM_per_s",
            "v_mean_mM_per_min",
            "v_sd_mM_per_min",
            "mean_trace_r2",
            "n_replicates",
        ],
    )
    if comparison_rows:
        write_csv_dicts(
            OUT / "rate_table_comparison_vs_legacy.csv",
            comparison_rows,
            list(comparison_rows[0].keys()),
        )

    return {
        "S_all": s_all,
        "stock_labels": stock_labels,
        "series": series,
        "trace_store": trace_store,
        "comparison_rows": comparison_rows,
        "selected_rate_slope": selected_slope,
    }


def export_rate_audit_figure(dataset: dict) -> None:
    comparison_rows = dataset["comparison_rows"]
    if not comparison_rows:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.ravel()
    by_label = {}
    for row in comparison_rows:
        by_label.setdefault(row["condition"], []).append(row)

    for ax, spec in zip(axes, CONDITION_SPECS):
        rows = sorted(by_label[spec["label"]], key=lambda r: r["final_S_mM"], reverse=True)
        s = np.array([r["final_S_mM"] for r in rows], dtype=float)
        legacy = np.array([r["legacy_v_mM_per_min"] for r in rows], dtype=float)
        recomputed = np.array([r["recomputed_v_mM_per_min"] for r in rows], dtype=float)
        ax.plot(s, legacy, "o-", label="legacy master table", color="gray")
        ax.plot(s, recomputed, "s--", label="recomputed raw plate", color=spec["color"])
        ax.set_title(spec["label"])
        ax.set_xlabel("[S] (mM)")
        ax.set_ylabel("v (mM/min)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Rate-table audit: legacy values versus recomputed raw-plate rates", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "rate_table_audit.png", dpi=200)
    plt.close(fig)


def export_representative_traces(dataset: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.ravel()
    trace_indices = [0, 4, 8]
    trace_labels = {0: "10 mM S", 4: "2.5 mM S", 8: "0.125 mM S"}
    trace_colors = {0: "tab:blue", 4: "tab:orange", 8: "tab:green"}

    for ax, spec in zip(axes, CONDITION_SPECS):
        traces = dataset["trace_store"][spec["label"]]
        for idx in trace_indices:
            t = traces[idx]["time_s"]
            mean_trace = traces[idx]["mean_pnp_mM"]
            sd_trace = traces[idx]["sd_pnp_mM"]
            fit = linear_fit(t, mean_trace)
            c = trace_colors[idx]
            ax.plot(t, mean_trace, color=c, linewidth=2, label=f"{trace_labels[idx]} mean")
            ax.fill_between(t, mean_trace - sd_trace, mean_trace + sd_trace, color=c, alpha=0.12)
            ax.plot(t, fit["yhat"], "--", color=c, alpha=0.9)
        ax.set_title(spec["label"])
        ax.set_xlabel("time (s)")
        ax.set_ylabel("[pNP] (mM)")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Representative baseline-corrected [pNP](t) traces used for initial-rate extraction", y=0.995, fontsize=14)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT / "representative_pnp_traces.png", dpi=200)
    plt.close(fig)


def export_concentration_change_check(dataset: dict) -> dict:
    rows = []
    avg_rows = []
    for spec in CONDITION_SPECS:
        traces = dataset["trace_store"][spec["label"]]
        series = dataset["series"][spec["label"]]
        for idx, s0 in enumerate(dataset["S_all"]):
            mean_trace = np.asarray(traces[idx]["mean_pnp_mM"], dtype=float)
            final_product = max(float(mean_trace[-1]), 0.0)
            avg_product = max(float(np.mean(mean_trace)), 0.0)
            final_s = max(float(s0) - final_product, 0.0)
            avg_s = max(float(s0) - avg_product, 0.0)
            final_i = float(spec["I_mM"]) + final_product
            avg_i = float(spec["I_mM"]) + avg_product
            percent_s_depleted = 100.0 * final_product / float(s0) if float(s0) > 0 else float("nan")
            percent_i_increase = (
                100.0 * final_product / float(spec["I_mM"]) if float(spec["I_mM"]) > 0 else float("nan")
            )
            row = {
                "condition": spec["label"],
                "I_initial_mM": float(spec["I_mM"]),
                "final_S_initial_mM": float(s0),
                "v_mean_mM_per_s": float(series["v"][idx]),
                "v_mean_mM_per_min": per_min(float(series["v"][idx])),
                "v_sd_mM_per_s": float(series["sd"][idx]),
                "v_sd_mM_per_min": per_min(float(series["sd"][idx])),
                "pNP_final_mM": final_product,
                "pNP_avg_mM": avg_product,
                "S_final_est_mM": final_s,
                "S_avg_est_mM": avg_s,
                "Pi_final_est_mM": final_i,
                "Pi_avg_est_mM": avg_i,
                "percent_S_depleted": percent_s_depleted,
                "percent_I_increase_vs_initial": percent_i_increase,
            }
            rows.append(row)
            avg_rows.append(
                {
                    "condition": spec["label"],
                    "I_mM": avg_i,
                    "final_S_mM": avg_s,
                    "v_mean_mM_per_s": float(series["v"][idx]),
                    "v_sd_mM_per_s": float(series["sd"][idx]),
                }
            )

    write_csv_dicts(OUT / "concentration_change_check.csv", rows, list(rows[0].keys()))

    summary_rows = []
    for spec in CONDITION_SPECS:
        cond_rows = [r for r in rows if r["condition"] == spec["label"]]
        max_depletion = max(float(r["percent_S_depleted"]) for r in cond_rows if np.isfinite(r["percent_S_depleted"]))
        finite_inc = [float(r["percent_I_increase_vs_initial"]) for r in cond_rows if np.isfinite(r["percent_I_increase_vs_initial"])]
        max_i_increase = max(finite_inc) if finite_inc else float("nan")
        summary_rows.append(
            {
                "condition": spec["label"],
                "max_percent_S_depleted": max_depletion,
                "max_percent_I_increase_vs_initial": max_i_increase,
            }
        )
    write_csv_dicts(
        OUT / "concentration_change_summary.csv",
        summary_rows,
        ["condition", "max_percent_S_depleted", "max_percent_I_increase_vs_initial"],
    )

    plt.figure(figsize=(9, 5))
    xpos = np.arange(len(summary_rows))
    plt.bar(xpos - 0.15, [r["max_percent_S_depleted"] for r in summary_rows], width=0.3, label="max % [pNPP] depleted")
    plt.bar(
        xpos + 0.15,
        [0.0 if not np.isfinite(r["max_percent_I_increase_vs_initial"]) else r["max_percent_I_increase_vs_initial"] for r in summary_rows],
        width=0.3,
        label="max % [Pi] increase",
    )
    plt.xticks(xpos, [r["condition"] for r in summary_rows], rotation=20)
    plt.ylabel("Percent change over run")
    plt.title("Reasonableness check for ignoring concentration changes during the run")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "concentration_change_summary.png", dpi=200)
    plt.close()

    return {"rows": rows, "avg_rows": avg_rows, "summary_rows": summary_rows}


def mm_model(s: np.ndarray, vmax: float, km: float) -> np.ndarray:
    return vmax * s / (km + s)


def competitive_model(s: np.ndarray, i_mM: np.ndarray, vmax: float, km: float, ki: float) -> np.ndarray:
    return vmax * s / (((1.0 + i_mM / ki) * km) + s)


def uncompetitive_model(s: np.ndarray, i_mM: np.ndarray, vmax: float, km: float, kip: float) -> np.ndarray:
    return vmax * s / (km + ((1.0 + i_mM / kip) * s))


def mixed_model(s: np.ndarray, i_mM: np.ndarray, vmax: float, km: float, ki: float, kip: float) -> np.ndarray:
    return vmax * s / (((1.0 + i_mM / ki) * km) + ((1.0 + i_mM / kip) * s))


def nelder_mead(fun, x0: np.ndarray, step: float = 0.25, max_iter: int = 400, tol: float = 1e-9) -> tuple[np.ndarray, float]:
    x0 = np.asarray(x0, dtype=float)
    n = len(x0)
    simplex = [x0]
    for j in range(n):
        x = x0.copy()
        x[j] += step
        simplex.append(x)
    simplex = np.asarray(simplex, dtype=float)
    values = np.array([fun(x) for x in simplex], dtype=float)

    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5

    for _ in range(max_iter):
        order = np.argsort(values)
        simplex = simplex[order]
        values = values[order]
        if np.max(np.abs(values - values[0])) < tol:
            break

        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]
        xr = centroid + alpha * (centroid - worst)
        fr = fun(xr)

        if values[0] <= fr < values[-2]:
            simplex[-1] = xr
            values[-1] = fr
            continue

        if fr < values[0]:
            xe = centroid + gamma * (xr - centroid)
            fe = fun(xe)
            if fe < fr:
                simplex[-1] = xe
                values[-1] = fe
            else:
                simplex[-1] = xr
                values[-1] = fr
            continue

        xc = centroid + rho * (worst - centroid)
        fc = fun(xc)
        if fc < values[-1]:
            simplex[-1] = xc
            values[-1] = fc
            continue

        for j in range(1, len(simplex)):
            simplex[j] = simplex[0] + sigma * (simplex[j] - simplex[0])
            values[j] = fun(simplex[j])

    order = np.argsort(values)
    simplex = simplex[order]
    values = values[order]
    return simplex[0], float(values[0])


def optimize_positive_model(
    objective,
    bounds: list[tuple[float, float]],
    seed_points: list[list[float]],
    rng_seed: int,
    n_random: int = 80,
    nm_step: float = 0.35,
    max_iter: int = 350,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(rng_seed)
    log_bounds = np.log10(np.asarray(bounds, dtype=float))
    n_params = len(bounds)

    def bounded_objective(log_params: np.ndarray) -> float:
        log_params = np.asarray(log_params, dtype=float)
        if np.any(~np.isfinite(log_params)):
            return 1e30
        penalty = 0.0
        clipped = log_params.copy()
        for j in range(n_params):
            lo, hi = log_bounds[j]
            if clipped[j] < lo:
                penalty += (lo - clipped[j]) ** 2 * 1e6
                clipped[j] = lo
            elif clipped[j] > hi:
                penalty += (clipped[j] - hi) ** 2 * 1e6
                clipped[j] = hi
        return float(objective(10.0 ** clipped) + penalty)

    candidate_logs = []
    for seed in seed_points:
        candidate_logs.append(np.log10(np.asarray(seed, dtype=float)))
    for _ in range(n_random):
        candidate_logs.append(
            np.array([rng.uniform(lo, hi) for lo, hi in log_bounds], dtype=float)
        )

    scored = sorted(((bounded_objective(c), c) for c in candidate_logs), key=lambda item: item[0])
    best_params = None
    best_value = float("inf")
    for _, cand in scored[: min(10, len(scored))]:
        opt_log, opt_val = nelder_mead(
            bounded_objective,
            cand,
            step=nm_step,
            max_iter=max_iter,
            tol=1e-10,
        )
        params = 10.0 ** opt_log
        if opt_val < best_value:
            best_value = opt_val
            best_params = params

    if best_params is None:
        raise RuntimeError("Optimization failed to produce a valid parameter set.")
    return np.asarray(best_params, dtype=float), float(best_value)


def aic_bic_from_sse(sse: float, n: int, k: int) -> tuple[float, float]:
    sse = max(float(sse), 1e-30)
    aic = n * np.log(sse / n) + 2 * k
    bic = n * np.log(sse / n) + k * np.log(n)
    return float(aic), float(bic)


def flatten_series(series: dict, s_all: np.ndarray) -> dict:
    s_list = []
    i_list = []
    v_list = []
    sd_list = []
    cond_list = []
    for label, d in series.items():
        for s, v, sd in zip(s_all, d["v"], d["sd"]):
            s_list.append(float(s))
            i_list.append(float(d["I_mM"]))
            v_list.append(float(v))
            sd_list.append(float(sd))
            cond_list.append(label)
    v_arr = np.asarray(v_list, dtype=float)
    sd_arr = positive_sigma(v_arr, np.asarray(sd_list, dtype=float))
    return {
        "S": np.asarray(s_list, dtype=float),
        "I": np.asarray(i_list, dtype=float),
        "V": v_arr,
        "SD": sd_arr,
        "condition": cond_list,
    }


def flatten_avg_rows(avg_rows: list[dict]) -> dict:
    s = np.asarray([r["final_S_mM"] for r in avg_rows], dtype=float)
    i_mM = np.asarray([r["I_mM"] for r in avg_rows], dtype=float)
    v = np.asarray([r["v_mean_mM_per_s"] for r in avg_rows], dtype=float)
    sd = positive_sigma(v, np.asarray([r["v_sd_mM_per_s"] for r in avg_rows], dtype=float))
    cond = [r["condition"] for r in avg_rows]
    return {"S": s, "I": i_mM, "V": v, "SD": sd, "condition": cond}


def fit_apparent_mm(s_all: np.ndarray, v: np.ndarray, sd: np.ndarray, rng_seed: int) -> dict:
    mask = np.isfinite(s_all) & np.isfinite(v) & np.isfinite(sd)
    s = s_all[mask]
    v = v[mask]
    sd = positive_sigma(v, sd[mask])

    def objective(params: np.ndarray) -> float:
        vmax, km = params
        pred = mm_model(s, vmax, km)
        return float(np.sum(((v - pred) / sd) ** 2))

    vmax_guess = max(float(np.nanmax(v)), 1e-8)
    positive_s = s[s > 0]
    km_guess = float(np.median(positive_s)) if positive_s.size else 1.0
    params, weighted_sse = optimize_positive_model(
        objective,
        bounds=[(1e-8, 1e-2), (1e-3, 100.0)],
        seed_points=[[vmax_guess, km_guess], [1.2 * vmax_guess, max(km_guess * 0.8, 1e-3)]],
        rng_seed=rng_seed,
        n_random=45,
        nm_step=0.25,
        max_iter=260,
    )
    pred = mm_model(s, *params)
    raw_sse = float(np.sum((v - pred) ** 2))
    sst = float(np.sum((v - np.mean(v)) ** 2))
    r2 = 1.0 - raw_sse / sst if sst > 0 else float("nan")
    return {
        "params": params,
        "weighted_sse": weighted_sse,
        "raw_sse": raw_sse,
        "R2": r2,
    }


def fit_lb(s_all: np.ndarray, v_all: np.ndarray, sd_all: np.ndarray) -> dict:
    mask = np.isfinite(s_all) & np.isfinite(v_all) & np.isfinite(sd_all) & (s_all > 0) & (v_all > 0)
    s = s_all[mask]
    v = v_all[mask]
    sd = positive_sigma(v, sd_all[mask])
    x = 1.0 / s
    y = 1.0 / v
    yerr = sd / np.maximum(v**2, 1e-16)

    m, b = np.polyfit(x, y, 1)
    yhat = m * x + b
    sse = float(np.sum((y - yhat) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    vmax = (1.0 / b) if b != 0 else float("nan")
    km = (m / b) if b != 0 else float("nan")

    X = np.column_stack([x, np.ones_like(x)])
    xtx_inv = np.linalg.inv(X.T @ X)
    dof = max(len(x) - 2, 1)
    sigma2 = sse / dof
    cov = sigma2 * xtx_inv
    se_slope = float(np.sqrt(cov[0, 0]))
    se_intercept = float(np.sqrt(cov[1, 1]))
    return {
        "slope": float(m),
        "intercept": float(b),
        "se_slope": se_slope,
        "se_intercept": se_intercept,
        "vmax_app_mM_per_s": float(vmax),
        "Km_app_mM": float(km),
        "R2": r2,
        "x": x,
        "y": y,
        "yerr": yerr,
    }


def bootstrap_apparent_mm(
    s_all: np.ndarray,
    v_all: np.ndarray,
    sd_all: np.ndarray,
    base_params: np.ndarray,
    rng_seed: int,
    n_boot: int = 120,
) -> dict:
    rng = np.random.default_rng(rng_seed)
    mask = np.isfinite(s_all) & np.isfinite(v_all) & np.isfinite(sd_all)
    s = s_all[mask]
    v = v_all[mask]
    sd = positive_sigma(v, sd_all[mask])
    samples = []
    for b in range(n_boot):
        v_b = np.clip(rng.normal(v, sd), 0.0, None)
        fit_b = fit_apparent_mm(s, v_b, sd, rng_seed + 1000 + b)
        samples.append(fit_b["params"])
    arr = np.asarray(samples, dtype=float)
    return {
        "vmax_ci": np.percentile(arr[:, 0], [2.5, 97.5]),
        "km_ci": np.percentile(arr[:, 1], [2.5, 97.5]),
        "kcat_ci": np.percentile(arr[:, 0] / ET_MILLIMOLAR, [2.5, 97.5]),
    }


def fit_global_inhibition_model(model_name: str, flat: dict, rng_seed: int) -> dict:
    s = flat["S"]
    i_mM = flat["I"]
    v = flat["V"]
    sd = flat["SD"]
    vmax_guess = max(float(np.nanmax(v)), 1e-8)
    positive_s = s[s > 0]
    km_guess = float(np.median(positive_s)) if positive_s.size else 1.0

    if model_name == "competitive":
        model = competitive_model
        bounds = [(1e-8, 1e-2), (1e-3, 100.0), (1e-3, 1000.0)]
        seed_points = [[vmax_guess, km_guess, 5.0], [1.2 * vmax_guess, max(km_guess * 0.8, 1e-3), 10.0]]
        param_names = ["vmax_mM_per_s", "Km_mM", "Ki_mM"]
    elif model_name == "uncompetitive":
        model = uncompetitive_model
        bounds = [(1e-8, 1e-2), (1e-3, 100.0), (1e-3, 1000.0)]
        seed_points = [[vmax_guess, km_guess, 10.0], [1.1 * vmax_guess, km_guess, 20.0]]
        param_names = ["vmax_mM_per_s", "Km_mM", "Ki_prime_mM"]
    elif model_name == "mixed":
        model = mixed_model
        bounds = [(1e-8, 1e-2), (1e-3, 100.0), (1e-3, 1000.0), (1e-3, 1000.0)]
        seed_points = [[vmax_guess, km_guess, 5.0, 20.0], [1.1 * vmax_guess, km_guess, 20.0, 5.0]]
        param_names = ["vmax_mM_per_s", "Km_mM", "Ki_mM", "Ki_prime_mM"]
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    def objective(params: np.ndarray) -> float:
        pred = model(s, i_mM, *params)
        return float(np.sum(((v - pred) / sd) ** 2))

    params, weighted_sse = optimize_positive_model(
        objective,
        bounds=bounds,
        seed_points=seed_points,
        rng_seed=rng_seed,
        n_random=85 if model_name == "mixed" else 60,
        nm_step=0.35,
        max_iter=380,
    )
    pred = model(s, i_mM, *params)
    raw_sse = float(np.sum((v - pred) ** 2))
    sst = float(np.sum((v - np.mean(v)) ** 2))
    r2 = 1.0 - raw_sse / sst if sst > 0 else float("nan")
    aic, bic = aic_bic_from_sse(raw_sse, len(v), len(params))

    return {
        "model": model_name,
        "param_names": param_names,
        "params": params,
        "weighted_sse": weighted_sse,
        "raw_sse": raw_sse,
        "R2": r2,
        "AIC": aic,
        "BIC": bic,
        "pred": pred,
    }


def bootstrap_global_model(
    model_name: str,
    flat: dict,
    base_result: dict,
    rng_seed: int,
    n_boot: int = 40,
) -> dict:
    rng = np.random.default_rng(rng_seed)
    s = flat["S"]
    i_mM = flat["I"]
    v = flat["V"]
    sd = flat["SD"]
    samples = []

    for b in range(n_boot):
        v_b = np.clip(rng.normal(v, sd), 0.0, None)
        boot_flat = {"S": s, "I": i_mM, "V": v_b, "SD": sd, "condition": flat["condition"]}
        fit_b = fit_global_inhibition_model(model_name, boot_flat, rng_seed + 1000 + b)
        samples.append(fit_b["params"])

    arr = np.asarray(samples, dtype=float)
    ci = {}
    for j, name in enumerate(base_result["param_names"]):
        ci[name] = np.percentile(arr[:, j], [2.5, 97.5])
    ci["kcat_per_s"] = np.percentile(arr[:, 0] / ET_MILLIMOLAR, [2.5, 97.5])
    return ci


def export_kinetics_figures(dataset: dict) -> dict:
    s_all = dataset["S_all"]
    series = dataset["series"]
    mm_results = {}
    lb_results = {}
    rows = []
    lb_summary_rows = []

    for idx, spec in enumerate(CONDITION_SPECS):
        label = spec["label"]
        d = series[label]
        mm = fit_apparent_mm(s_all, d["v"], d["sd"], BOOTSTRAP_SEED + idx)
        mm_ci = bootstrap_apparent_mm(
            s_all,
            d["v"],
            d["sd"],
            mm["params"],
            BOOTSTRAP_SEED + idx * 10,
            n_boot=120,
        )
        lb = fit_lb(s_all, per_min(d["v"]), per_min(d["sd"]))
        mm_results[label] = {**mm, **mm_ci}
        lb_results[label] = lb
        vmax, km = mm["params"]
        rows.append(
            {
                "condition": label,
                "I_mM": d["I_mM"],
                "MM_vmax_mM_per_s": vmax,
                "MM_vmax_mM_per_min": per_min(vmax),
                "MM_vmax_CI95_lo": float(mm_ci["vmax_ci"][0]),
                "MM_vmax_CI95_hi": float(mm_ci["vmax_ci"][1]),
                "MM_vmax_CI95_lo_mM_per_min": per_min(float(mm_ci["vmax_ci"][0])),
                "MM_vmax_CI95_hi_mM_per_min": per_min(float(mm_ci["vmax_ci"][1])),
                "MM_Km_mM": km,
                "MM_Km_CI95_lo": float(mm_ci["km_ci"][0]),
                "MM_Km_CI95_hi": float(mm_ci["km_ci"][1]),
                "MM_kcat_per_s": vmax / ET_MILLIMOLAR,
                "MM_kcat_CI95_lo_per_s": float(mm_ci["kcat_ci"][0]),
                "MM_kcat_CI95_hi_per_s": float(mm_ci["kcat_ci"][1]),
                "MM_weighted_SSE": mm["weighted_sse"],
                "MM_raw_SSE": mm["raw_sse"],
                "MM_R2": mm["R2"],
                "LB_vmax_app_mM_per_s": lb["vmax_app_mM_per_s"],
                "LB_vmax_app_mM_per_min": per_min(lb["vmax_app_mM_per_s"]),
                "LB_Km_app_mM": lb["Km_app_mM"],
                "LB_slope": lb["slope"],
                "LB_intercept": lb["intercept"],
                "LB_slope_SE": lb["se_slope"],
                "LB_intercept_SE": lb["se_intercept"],
                "LB_R2": lb["R2"],
                "data_source": d["source"],
            }
        )
        lb_summary_rows.append(
            {
                "condition": label,
                "I_mM": d["I_mM"],
                "slope": lb["slope"],
                "intercept": lb["intercept"],
                "slope_se": lb["se_slope"],
                "intercept_se": lb["se_intercept"],
            }
        )

    write_csv_dicts(OUT / "mm_lb_summary.csv", rows, list(rows[0].keys()))
    write_csv_dicts(
        OUT / "lb_slope_intercept_summary.csv",
        lb_summary_rows,
        list(lb_summary_rows[0].keys()),
    )

    sorted_rows = sorted(rows, key=lambda r: inhibitor_conc_mM(r["condition"]))
    km0 = next(float(r["MM_Km_mM"]) for r in sorted_rows if inhibitor_conc_mM(r["condition"]) == 0.0)
    ki_rows = []
    for r in sorted_rows:
        i_mM = inhibitor_conc_mM(r["condition"])
        km_app = float(r["MM_Km_mM"])
        if i_mM == 0.0:
            ki = float("nan")
            note = "reference (no inhibitor)"
        else:
            denom = (km_app / km0) - 1.0
            if not np.isfinite(denom) or denom <= 0:
                ki = float("nan")
                note = "nonphysical for competitive model (Km_app <= Km0)"
            else:
                ki = i_mM / denom
                note = ""
        ki_rows.append(
            {
                "condition": r["condition"],
                "I_mM": i_mM,
                "Km0_mM": km0,
                "Km_app_mM": km_app,
                "Ki_competitive_mM": ki,
                "note": note,
            }
        )
    write_csv_dicts(
        OUT / "ki_competitive_from_mm.csv",
        ki_rows,
        ["condition", "I_mM", "Km0_mM", "Km_app_mM", "Ki_competitive_mM", "note"],
    )

    s_plot = np.linspace(0.0, float(np.nanmax(s_all)), 400)

    plt.figure(figsize=(12, 6))
    plt.title("Initial velocity versus substrate with apparent Michaelis-Menten fits", fontsize=16)
    plt.xlabel("[Substrate] (mM)", fontsize=13)
    plt.ylabel("Initial velocity (mM/min)", fontsize=13)
    plt.grid(True, alpha=0.3)
    for spec in CONDITION_SPECS:
        label = spec["label"]
        d = series[label]
        vmax, km = mm_results[label]["params"]
        plt.errorbar(
            s_all,
            per_min(d["v"]),
            yerr=per_min(d["sd"]),
            fmt="o",
            color=spec["color"],
            ecolor="black",
            capsize=3,
            label=f"{label} data",
        )
        plt.plot(
            s_plot,
            per_min(mm_model(s_plot, vmax, km)),
            color=spec["color"],
            linewidth=2,
            label=f"{label} fit: Vmax={per_min(vmax):.3e} mM/min, Km={km:.3f}",
        )
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT / "mm.png", dpi=200)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.title("Lineweaver-Burk diagnostic plot with propagated y-error bars", fontsize=16)
    plt.xlabel("1/[Substrate] (1/mM)", fontsize=13)
    plt.ylabel("1/Initial velocity (min/mM)", fontsize=13)
    plt.grid(True, alpha=0.3)
    x_max = max(lb_results[k]["x"].max() for k in lb_results)
    x_line = np.linspace(-0.35 * x_max, 1.05 * x_max, 600)
    y_vals = []
    for spec in CONDITION_SPECS:
        label = spec["label"]
        lb = lb_results[label]
        y_fit = lb["slope"] * x_line + lb["intercept"]
        plt.errorbar(
            lb["x"],
            lb["y"],
            yerr=lb["yerr"],
            fmt="o",
            color=spec["color"],
            ecolor="black",
            capsize=3,
            label=f"{label} data",
        )
        plt.plot(
            x_line,
            y_fit,
            color=spec["color"],
            linewidth=2,
            label=f"{label} fit",
        )
        plt.plot(0, lb["intercept"], marker="x", color=spec["color"], markersize=9, markeredgewidth=2)
        y_vals.extend(lb["y"].tolist())
        y_vals.extend(y_fit.tolist())
    plt.axvline(0, color="k", linewidth=1.2, alpha=0.8)
    plt.axhline(0, color="k", linewidth=1.2, alpha=0.8)
    y_min = min(y_vals)
    y_max = max(y_vals)
    y_pad = 0.1 * (y_max - y_min) if y_max > y_min else 1.0
    plt.ylim(y_min - y_pad, y_max + y_pad)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT / "lb.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.title("Lineweaver-Burk intercept zoom", fontsize=14)
    plt.xlabel("1/[Substrate] (1/mM)", fontsize=12)
    plt.ylabel("1/Initial velocity (min/mM)", fontsize=12)
    plt.grid(True, alpha=0.3)
    x_zoom = np.linspace(-0.18 * x_max, 0.18 * x_max, 400)
    intercepts = []
    for spec in CONDITION_SPECS:
        label = spec["label"]
        lb = lb_results[label]
        plt.plot(x_zoom, lb["slope"] * x_zoom + lb["intercept"], color=spec["color"], linewidth=2, label=label)
        plt.plot(0, lb["intercept"], marker="o", color=spec["color"], markersize=6)
        intercepts.append(lb["intercept"])
    plt.axvline(0, color="k", linewidth=1.2, alpha=0.8)
    plt.axhline(0, color="k", linewidth=1.2, alpha=0.8)
    i_min = min(intercepts)
    i_max = max(intercepts)
    i_pad = 0.2 * (i_max - i_min) if i_max > i_min else 1.0
    plt.ylim(i_min - i_pad, i_max + i_pad)
    plt.xlim(-0.18 * x_max, 0.18 * x_max)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT / "lbzoom.png", dpi=200)
    plt.close()

    conds = [r["condition"] for r in rows]
    xpos = np.arange(len(conds))

    plt.figure(figsize=(8, 5))
    plt.title("Apparent Vmax with 95% bootstrap CI", fontsize=14)
    vmax = np.array([r["MM_vmax_mM_per_min"] for r in rows], dtype=float)
    vmax_lo = np.array([r["MM_vmax_CI95_lo_mM_per_min"] for r in rows], dtype=float)
    vmax_hi = np.array([r["MM_vmax_CI95_hi_mM_per_min"] for r in rows], dtype=float)
    vmax_err = np.vstack([vmax - vmax_lo, vmax_hi - vmax])
    plt.errorbar(xpos, vmax, yerr=vmax_err, fmt="o", capsize=4)
    plt.xticks(xpos, conds, rotation=20)
    plt.ylabel("Vmax (mM/min)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "mmcl.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.title("Apparent Km with 95% bootstrap CI", fontsize=14)
    km = np.array([r["MM_Km_mM"] for r in rows], dtype=float)
    km_lo = np.array([r["MM_Km_CI95_lo"] for r in rows], dtype=float)
    km_hi = np.array([r["MM_Km_CI95_hi"] for r in rows], dtype=float)
    km_err = np.vstack([km - km_lo, km_hi - km])
    plt.errorbar(xpos, km, yerr=km_err, fmt="o", capsize=4)
    plt.xticks(xpos, conds, rotation=20)
    plt.ylabel("Km (mM)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "kmcl.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.title("Apparent kcat with 95% bootstrap CI", fontsize=14)
    kcat = np.array([r["MM_kcat_per_s"] for r in rows], dtype=float)
    kcat_lo = np.array([r["MM_kcat_CI95_lo_per_s"] for r in rows], dtype=float)
    kcat_hi = np.array([r["MM_kcat_CI95_hi_per_s"] for r in rows], dtype=float)
    kcat_err = np.vstack([kcat - kcat_lo, kcat_hi - kcat])
    plt.errorbar(xpos, kcat, yerr=kcat_err, fmt="o", capsize=4)
    plt.xticks(xpos, conds, rotation=20)
    plt.ylabel("kcat (1/s)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "kcatcl.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    slopes = np.array([r["slope"] for r in lb_summary_rows], dtype=float)
    slopes_se = np.array([r["slope_se"] for r in lb_summary_rows], dtype=float)
    intercepts = np.array([r["intercept"] for r in lb_summary_rows], dtype=float)
    intercepts_se = np.array([r["intercept_se"] for r in lb_summary_rows], dtype=float)
    i_mM = np.array([r["I_mM"] for r in lb_summary_rows], dtype=float)
    plt.errorbar(i_mM, slopes, yerr=slopes_se, fmt="o-", capsize=4, label="LB slope")
    plt.errorbar(i_mM, intercepts, yerr=intercepts_se, fmt="s--", capsize=4, label="LB intercept")
    plt.xlabel("Inhibitor concentration (mM)")
    plt.ylabel("LB fit parameter")
    plt.title("Lineweaver-Burk slope and intercept trends", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "lb_slope_intercept_vs_I.png", dpi=200)
    plt.close()

    return {
        "rows": rows,
        "mm_results": mm_results,
        "lb_results": lb_results,
    }


def export_global_model_analysis(dataset: dict, flat: dict | None = None, prefix: str = "global_model", title_suffix: str = "") -> dict:
    flat = flatten_series(dataset["series"], dataset["S_all"]) if flat is None else flat
    model_results = {}
    comparison_rows = []
    param_rows = []

    for j, model_name in enumerate(["competitive", "uncompetitive", "mixed"]):
        fit = fit_global_inhibition_model(model_name, flat, BOOTSTRAP_SEED + 200 + j)
        ci = bootstrap_global_model(model_name, flat, fit, BOOTSTRAP_SEED + 300 + j, n_boot=40)
        model_results[model_name] = {**fit, "ci": ci}
        comparison_rows.append(
            {
                "model": model_name,
                "weighted_SSE": fit["weighted_sse"],
                "raw_SSE": fit["raw_sse"],
                "R2": fit["R2"],
                "AIC": fit["AIC"],
                "BIC": fit["BIC"],
                "k_parameters": len(fit["params"]),
                "vmax_mM_per_s": fit["params"][0],
                "vmax_mM_per_min": per_min(fit["params"][0]),
                "kcat_per_s": fit["params"][0] / ET_MILLIMOLAR,
                "kcat_CI95_lo_per_s": float(ci["kcat_per_s"][0]),
                "kcat_CI95_hi_per_s": float(ci["kcat_per_s"][1]),
            }
        )
        for name, value in zip(fit["param_names"], fit["params"]):
            ci_vals = ci[name]
            row = {"model": model_name, "parameter": name, "estimate": float(value), "CI95_lo": float(ci_vals[0]), "CI95_hi": float(ci_vals[1])}
            param_rows.append(row)
            if name == "vmax_mM_per_s":
                param_rows.append(
                    {
                        "model": model_name,
                        "parameter": "vmax_mM_per_min",
                        "estimate": per_min(float(value)),
                        "CI95_lo": per_min(float(ci_vals[0])),
                        "CI95_hi": per_min(float(ci_vals[1])),
                    }
                )
        param_rows.append(
            {
                "model": model_name,
                "parameter": "kcat_per_s",
                "estimate": float(fit["params"][0] / ET_MILLIMOLAR),
                "CI95_lo": float(ci["kcat_per_s"][0]),
                "CI95_hi": float(ci["kcat_per_s"][1]),
            }
        )

    comparison_rows = sorted(comparison_rows, key=lambda r: r["AIC"])
    best_model = comparison_rows[0]["model"]
    best_aic = comparison_rows[0]["AIC"]
    for row in comparison_rows:
        row["delta_AIC"] = row["AIC"] - best_aic
        row["best_by_AIC"] = row["model"] == best_model

    write_csv_dicts(
        OUT / f"{prefix}_comparison.csv",
        comparison_rows,
        list(comparison_rows[0].keys()),
    )
    write_csv_dicts(
        OUT / f"{prefix}_parameter_summary.csv",
        param_rows,
        ["model", "parameter", "estimate", "CI95_lo", "CI95_hi"],
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    s_plot = np.linspace(0.0, float(np.nanmax(dataset["S_all"])), 400)
    for ax, model_name in zip(axes, ["competitive", "uncompetitive", "mixed"]):
        fit = model_results[model_name]
        for spec in CONDITION_SPECS:
            label = spec["label"]
            d = dataset["series"][label]
            ax.errorbar(
                dataset["S_all"],
                per_min(d["v"]),
                yerr=per_min(d["sd"]),
                fmt="o",
                color=spec["color"],
                ecolor="black",
                capsize=3,
                alpha=0.9,
            )
            if model_name == "competitive":
                curve = competitive_model(s_plot, np.full_like(s_plot, d["I_mM"]), *fit["params"])
            elif model_name == "uncompetitive":
                curve = uncompetitive_model(s_plot, np.full_like(s_plot, d["I_mM"]), *fit["params"])
            else:
                curve = mixed_model(s_plot, np.full_like(s_plot, d["I_mM"]), *fit["params"])
            ax.plot(s_plot, per_min(curve), color=spec["color"], linewidth=2)
        ax.set_title(f"{model_name.capitalize()}{title_suffix}\nAIC={fit['AIC']:.2f}, BIC={fit['BIC']:.2f}")
        ax.set_xlabel("[S] (mM)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Initial velocity (mM/min)")
    fig.suptitle(f"Global inhibition-model overlays{title_suffix}", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / f"{prefix}_overlays.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, model_name in zip(axes, ["competitive", "uncompetitive", "mixed"]):
        fit = model_results[model_name]
        pred = fit["pred"]
        resid = per_min(flat["V"] - pred)
        for spec in CONDITION_SPECS:
            mask = np.isclose(flat["I"], spec["I_mM"])
            ax.errorbar(
                flat["S"][mask],
                resid[mask],
                yerr=per_min(flat["SD"][mask]),
                fmt="o",
                color=spec["color"],
                ecolor="black",
                capsize=3,
                alpha=0.9,
            )
        ax.axhline(0.0, color="k", linewidth=1.0, alpha=0.8)
        ax.set_title(model_name.capitalize())
        ax.set_xlabel("[S] (mM)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Residual (mM/min)")
    fig.suptitle(f"Residual diagnostics for global inhibition models{title_suffix}", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / f"{prefix}_residuals.png", dpi=200)
    plt.close(fig)

    with (OUT / f"{prefix}_summary.txt").open("w") as f:
        f.write(f"best_model_by_AIC\t{best_model}\n")
        for row in comparison_rows:
            f.write(
                f"{row['model']}\tAIC={row['AIC']:.4f}\tBIC={row['BIC']:.4f}\tdelta_AIC={row['delta_AIC']:.4f}\t"
                f"weighted_SSE={row['weighted_SSE']:.4f}\traw_SSE={row['raw_SSE']:.6e}\n"
            )

    return {
        "comparison_rows": comparison_rows,
        "param_rows": param_rows,
        "best_model": best_model,
    }


def write_analysis_summary(
    dataset: dict,
    kinetics: dict,
    global_results: dict,
    calibration: dict,
    concentration_change: dict,
    avg_conc_results: dict,
) -> None:
    best_model = global_results["best_model"]
    best_model_avg = avg_conc_results["best_model"]
    max_depletion = max(r["max_percent_S_depleted"] for r in concentration_change["summary_rows"])
    finite_inc = [r["max_percent_I_increase_vs_initial"] for r in concentration_change["summary_rows"] if np.isfinite(r["max_percent_I_increase_vs_initial"])]
    max_i_increase = max(finite_inc) if finite_inc else float("nan")
    lines = []
    lines.append("# Inhibition analysis summary")
    lines.append("")
    lines.append("## What was standardized")
    lines.append(
        f"- All four inhibitor conditions were recomputed from raw plate CSVs using the same rate-rebuild mapping (rows A-C as controls, rows D-F as reaction triplicates) and the selected calibration slope from column {RATE_CALIBRATION_COLUMN} of `02_pNP_calibration_curve.csv`."
    )
    lines.append(
        f"- Selected calibration slope for rate reconstruction: `{calibration['selected_rate_slope']:.6f} A/mM`."
    )
    lines.append("")
    lines.append("## Primary inference strategy")
    lines.append("- Apparent `Vmax`, `Km`, and `kcat` by inhibitor were fit in native `v` vs `[S]` space with uncertainty-aware nonlinear fitting.")
    lines.append("- Report-facing rate outputs are now exported in `mM/min` to match the handout, while the internal fitting logic remains unit-consistent.")
    lines.append("- Lineweaver-Burk was retained as a diagnostic view, not the primary estimator.")
    lines.append("- Mechanism was compared with global nonlinear fits for competitive, uncompetitive, and mixed inhibition.")
    lines.append("")
    lines.append("## Best current model")
    lines.append(f"- Best model by AIC in the current export: `{best_model}`.")
    lines.append(f"- Sensitivity check using average `[pNPP]` and `[Pi]` over the run also favored: `{best_model_avg}`.")
    lines.append("- Final report wording should still use `most consistent with` unless the model separation remains strong after you review the residuals and parameter uncertainty.")
    lines.append("")
    lines.append("## Concentration-change reasonableness check")
    lines.append(f"- Maximum estimated `[pNPP]` depletion over a run: `{max_depletion:.2f}%`.")
    if np.isfinite(max_i_increase):
        lines.append(f"- Maximum estimated increase in `[Pi]` relative to its starting inhibitor level: `{max_i_increase:.2f}%`.")
    lines.append("- Use `concentration_change_check.csv` and `global_model_avg_conc_comparison.csv` in the appendix to justify whether initial concentrations were adequate or whether average concentrations materially change the model ranking.")
    lines.append("")
    lines.append("## Files to cite in the report draft")
    lines.append("- `figures_export/rate_table_recomputed_from_raw.csv`")
    lines.append("- `figures_export/mm_lb_summary.csv`")
    lines.append("- `figures_export/global_model_comparison.csv`")
    lines.append("- `figures_export/global_model_parameter_summary.csv`")
    lines.append("- `figures_export/concentration_change_check.csv`")
    lines.append("- `figures_export/global_model_avg_conc_comparison.csv`")
    lines.append("- `figures_export/mm.png`, `lb.png`, `lbzoom.png`, `global_model_overlays.png`, `global_model_residuals.png`")
    lines.append("")
    lines.append("## How to discuss nonlinear versus Lineweaver-Burk")
    lines.append("- Use Lineweaver-Burk to show the geometric pattern as inhibitor changes.")
    lines.append("- Explain that reciprocal transforms magnify low-rate points and distort error structure.")
    lines.append("- State that the inhibition call is based primarily on the global nonlinear comparison, with LB used as a supporting visualization and not as the final estimator.")
    lines.append("")
    (OUT / "analysis_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    os.chdir(HERE)
    calibration = export_calibration_figures()
    dataset = build_analysis_dataset(calibration)
    export_rate_audit_figure(dataset)
    export_representative_traces(dataset)
    concentration_change = export_concentration_change_check(dataset)
    kinetics = export_kinetics_figures(dataset)
    global_results = export_global_model_analysis(dataset)
    avg_flat = flatten_avg_rows(concentration_change["avg_rows"])
    avg_conc_results = export_global_model_analysis(
        dataset,
        flat=avg_flat,
        prefix="global_model_avg_conc",
        title_suffix=" (average concentrations)",
    )
    write_analysis_summary(dataset, kinetics, global_results, calibration, concentration_change, avg_conc_results)
    print(f"Analysis exports written to {OUT.resolve()}")


if __name__ == "__main__":
    main()
