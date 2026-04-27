"""Ensemble NM5 with Optuna-tuned v1, v3, v3d, and v8 predictions."""

import numpy as np
import pandas as pd

from hackathon_opti.baselines import SERIES_KEYS, summarize_fold_metrics
from hackathon_opti.config import (
    BASELINE_PREDICTIONS,
    METRICS_DIR,
    MODEL_V1_PREDICTIONS,
    MODEL_V3D_PREDICTIONS,
    PREDICTIONS_DIR,
)
from hackathon_opti.ensemble_stacking import (
    MERGE_KEYS,
    _compute_error_cols,
    _mape_objective,
    apply_per_series_selection,
    build_nelder_mead_predictions,
    optimize_horizon_weights_nelder_mead,
)

MODEL_NAME = "ensemble_nm5_tuned"

# Load predictions: naive, v1_tuned, v3_tuned, v3d_tuned, v8_tuned
print("Loading base model predictions...")
preds = {
    "naive": pd.read_csv(BASELINE_PREDICTIONS),
    "v1": pd.read_csv(PREDICTIONS_DIR / "model_v1_tuned_predictions.csv"),
    "v3": pd.read_csv(PREDICTIONS_DIR / "model_v3_tuned_predictions.csv"),
    "v3d": pd.read_csv(PREDICTIONS_DIR / "model_v3d_tuned_predictions.csv"),
    "v8": pd.read_csv(PREDICTIONS_DIR / "model_v8_tuned_predictions.csv"),
}
for name, df in preds.items():
    print(f"  {name}: {len(df)} rows")

# Merge
print("Merging predictions...")
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
print(f"  Merged: {len(merged)} rows")

# Optimize
print("\nOptimizing horizon-band weights (Nelder-Mead, MAPE-direct)...")
horizon_weights = optimize_horizon_weights_nelder_mead(merged, pred_cols)

predictions = build_nelder_mead_predictions(merged, pred_cols, horizon_weights)
predictions = _compute_error_cols(predictions)

print("\n=== Before per-series selection ===")
pre_metrics = summarize_fold_metrics(predictions)
pre_mean = pre_metrics[pre_metrics["fold"] == "mean"].iloc[0]
print(f"  MAPE: {pre_mean['mape']:.4f}")
print(f"  MAE:  {pre_mean['mae']:.4f}")
print(f"  RMSE: {pre_mean['rmse']:.4f}")

print("\nApplying per-series naive selection...")
predictions = apply_per_series_selection(predictions)

metrics = summarize_fold_metrics(predictions)
predictions.to_csv(PREDICTIONS_DIR / f"{MODEL_NAME}_predictions.csv", index=False)
metrics.to_csv(METRICS_DIR / f"{MODEL_NAME}_fold_metrics.csv", index=False)

print(f"\n=== {MODEL_NAME} Final ===")
print(metrics.to_string(index=False))

# Compare with original
print("\n=== Comparison with original NM5 ===")
orig = pd.read_csv(METRICS_DIR / "ensemble_nm5_fold_metrics.csv")
print("Original:")
print(orig.to_string(index=False))
print("\nTuned:")
print(metrics.to_string(index=False))
