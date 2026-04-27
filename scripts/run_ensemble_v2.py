"""V2 ensemble: adds v9 to the base model set, with resolution-aware weighting."""

import numpy as np
import pandas as pd

from hackathon_opti.baselines import SERIES_KEYS, summarize_fold_metrics
from hackathon_opti.config import BASELINE_PREDICTIONS, METRICS_DIR, PREDICTIONS_DIR
from hackathon_opti.ensemble_stacking import (
    MERGE_KEYS,
    _compute_error_cols,
    _mape_objective,
    apply_per_series_selection,
    optimize_horizon_weights_nelder_mead,
    build_nelder_mead_predictions,
)

MODEL_NAME = "ensemble_v2"

print("Loading base model predictions...")
preds = {
    "naive": pd.read_csv(BASELINE_PREDICTIONS),
    "v1": pd.read_csv(PREDICTIONS_DIR / "model_v1_tuned_predictions.csv"),
    "v3": pd.read_csv(PREDICTIONS_DIR / "model_v3_tuned_predictions.csv"),
    "v3d": pd.read_csv(PREDICTIONS_DIR / "model_v3d_tuned_predictions.csv"),
    "v8": pd.read_csv(PREDICTIONS_DIR / "model_v8_tuned_predictions.csv"),
}

# Add v9 (try tuned first, then base)
v9_tuned_path = PREDICTIONS_DIR / "model_v9_tuned_predictions.csv"
v9_base_path = PREDICTIONS_DIR / "model_v9_predictions.csv"
if v9_tuned_path.exists():
    preds["v9"] = pd.read_csv(v9_tuned_path)
    print("  Using v9 tuned predictions")
elif v9_base_path.exists():
    preds["v9"] = pd.read_csv(v9_base_path)
    print("  Using v9 base predictions")
else:
    print("  WARNING: v9 predictions not found, proceeding without v9")

# Add v9 weighted if available
v9w_path = PREDICTIONS_DIR / "model_v9_weighted_predictions.csv"
if v9w_path.exists():
    preds["v9w"] = pd.read_csv(v9w_path)
    print("  Using v9 weighted predictions")

for name, df in preds.items():
    print(f"  {name}: {len(df)} rows")

# Merge
print("\nMerging predictions...")
base = preds["v3"][MERGE_KEYS + ["actual", "system_area", "h3_resolution", "horizon"]].copy()
for name, df in preds.items():
    col = f"pred_{name}"
    base = base.merge(
        df[MERGE_KEYS + ["prediction"]].rename(columns={"prediction": col}),
        on=MERGE_KEYS,
        how="left",
    )
base["system_area_id"] = base["system_area"].map({"PD": 0, "TS": 1})
pred_cols = [f"pred_{n}" for n in preds.keys()]
merged = base.dropna(subset=pred_cols + ["actual"])
print(f"  Merged: {len(merged)} rows, {len(pred_cols)} models: {[c.replace('pred_', '') for c in pred_cols]}")

# --- Approach 1: Standard NM on full data ---
print("\n=== Approach 1: Standard Nelder-Mead (all data) ===")
hw_global = optimize_horizon_weights_nelder_mead(merged, pred_cols)
preds_global = build_nelder_mead_predictions(merged, pred_cols, hw_global)
preds_global = _compute_error_cols(preds_global)
m1_pre = summarize_fold_metrics(preds_global)
m1_pre_mean = m1_pre[m1_pre["fold"] == "mean"].iloc[0]
print(f"  Before selection: MAPE={m1_pre_mean['mape']:.4f}")

preds_global = apply_per_series_selection(preds_global)
m1 = summarize_fold_metrics(preds_global)
m1_mean = m1[m1["fold"] == "mean"].iloc[0]
print(f"  After selection: MAPE={m1_mean['mape']:.4f}")

# --- Approach 2: Resolution-specific weighting ---
print("\n=== Approach 2: Resolution-specific NM weighting ===")
res6_data = merged[merged["h3_resolution"] == 6].copy()
res7_data = merged[merged["h3_resolution"] == 7].copy()

print("  Res 6:")
hw_res6 = optimize_horizon_weights_nelder_mead(res6_data, pred_cols)
preds_res6 = build_nelder_mead_predictions(res6_data, pred_cols, hw_res6)
preds_res6 = _compute_error_cols(preds_res6)

print("  Res 7:")
hw_res7 = optimize_horizon_weights_nelder_mead(res7_data, pred_cols)
preds_res7 = build_nelder_mead_predictions(res7_data, pred_cols, hw_res7)
preds_res7 = _compute_error_cols(preds_res7)

preds_res_split = pd.concat([preds_res6, preds_res7], ignore_index=True)
m2_pre = summarize_fold_metrics(preds_res_split)
m2_pre_mean = m2_pre[m2_pre["fold"] == "mean"].iloc[0]
print(f"  Before selection: MAPE={m2_pre_mean['mape']:.4f}")

preds_res_split = apply_per_series_selection(preds_res_split)
m2 = summarize_fold_metrics(preds_res_split)
m2_mean = m2[m2["fold"] == "mean"].iloc[0]
print(f"  After selection: MAPE={m2_mean['mape']:.4f}")

# Choose best approach
if m1_mean["mape"] <= m2_mean["mape"]:
    print(f"\n  → Global NM wins ({m1_mean['mape']:.4f} vs {m2_mean['mape']:.4f})")
    best_preds = preds_global
    best_metrics = m1
else:
    print(f"\n  → Resolution-specific NM wins ({m2_mean['mape']:.4f} vs {m1_mean['mape']:.4f})")
    best_preds = preds_res_split
    best_metrics = m2

# Save
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
best_preds = best_preds.sort_values(["fold", "period_ym"] + SERIES_KEYS).reset_index(drop=True)
best_preds.to_csv(PREDICTIONS_DIR / f"{MODEL_NAME}_predictions.csv", index=False)
best_metrics.to_csv(METRICS_DIR / f"{MODEL_NAME}_fold_metrics.csv", index=False)

print(f"\n{'='*60}")
print(f"=== {MODEL_NAME} Final Metrics ===")
print(best_metrics.to_string(index=False))

# Compare with original ensemble
print("\n--- vs Original NM5 tuned ---")
try:
    orig = pd.read_csv(METRICS_DIR / "ensemble_nm5_tuned_fold_metrics.csv")
    for _, row in best_metrics.iterrows():
        fold = row["fold"]
        o = orig[orig["fold"] == fold]
        if not o.empty:
            delta = row["mape"] - o.iloc[0]["mape"]
            arrow = "↓" if delta < 0 else "↑"
            print(f"  {fold}: new={row['mape']:.4f}  old={o.iloc[0]['mape']:.4f}  delta={delta:+.4f} {arrow}")
except FileNotFoundError:
    pass
