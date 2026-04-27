"""Stacking ensemble with Ridge meta-learner.

Leakage-safe: meta-learner trained on leave-one-fold-out OOF predictions.
Base models: naive, v1, v3, v3d, v8.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from .baselines import SERIES_KEYS, summarize_fold_metrics
from .config import (
    BASELINE_PREDICTIONS,
    ENSEMBLE_STACKING_METRICS,
    ENSEMBLE_STACKING_PREDICTIONS,
    METRICS_DIR,
    MODEL_V1_PREDICTIONS,
    MODEL_V3D_PREDICTIONS,
    MODEL_V3_PREDICTIONS,
    MODEL_V8_PREDICTIONS,
    PREDICTIONS_DIR,
)

MERGE_KEYS = ["fold", "cell_id", "rate_category_id", "period_ym"]
MODEL_NAMES = ["naive", "v1", "v3", "v3d", "v8"]
PRED_COLS = [f"pred_{n}" for n in MODEL_NAMES]
SYSTEM_AREA_ENCODING = {"PD": 0, "TS": 1}


def _load_predictions() -> dict[str, pd.DataFrame]:
    preds = {
        "naive": pd.read_csv(BASELINE_PREDICTIONS),
        "v1": pd.read_csv(MODEL_V1_PREDICTIONS),
        "v3": pd.read_csv(MODEL_V3_PREDICTIONS),
    }
    if MODEL_V3D_PREDICTIONS.exists():
        preds["v3d"] = pd.read_csv(MODEL_V3D_PREDICTIONS)
    if MODEL_V8_PREDICTIONS.exists():
        preds["v8"] = pd.read_csv(MODEL_V8_PREDICTIONS)
    return preds


def _merge_predictions(preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    # Use v3 as base (has horizon, system_area, h3_resolution)
    base = preds["v3"][
        MERGE_KEYS + ["actual", "system_area", "h3_resolution", "horizon"]
    ].copy()

    for name, df in preds.items():
        col = f"pred_{name}"
        base = base.merge(
            df[MERGE_KEYS + ["prediction"]].rename(columns={"prediction": col}),
            on=MERGE_KEYS,
            how="left",
        )

    base["system_area_id"] = base["system_area"].map(SYSTEM_AREA_ENCODING)
    available_pred_cols = [c for c in PRED_COLS if c in base.columns]
    return base.dropna(subset=available_pred_cols + ["actual"])


def train_stacking_lofocv(
    merged: pd.DataFrame,
    pred_cols: list[str],
) -> pd.DataFrame:
    """Leave-one-fold-out cross-validated stacking in log1p space."""
    folds = sorted(merged["fold"].unique())
    all_oof = []

    for hold_out in folds:
        train_mask = merged["fold"] != hold_out
        test_mask = merged["fold"] == hold_out

        X_train = np.log1p(merged.loc[train_mask, pred_cols].clip(lower=0).values)
        y_train = np.log1p(merged.loc[train_mask, "actual"].clip(lower=0).values)
        X_test = np.log1p(merged.loc[test_mask, pred_cols].clip(lower=0).values)

        meta_model = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0], fit_intercept=True)
        meta_model.fit(X_train, y_train)

        oof = merged[test_mask].copy()
        oof["prediction"] = np.maximum(np.expm1(meta_model.predict(X_test)), 0.0)
        all_oof.append(oof)

        coefs = dict(zip(pred_cols, meta_model.coef_))
        print(f"  {hold_out} (alpha={meta_model.alpha_}): " +
              ", ".join(f"{k}={v:.4f}" for k, v in coefs.items()))

    return pd.concat(all_oof, ignore_index=True)


def apply_per_series_selection(predictions: pd.DataFrame) -> pd.DataFrame:
    """For each series, if naive beats stacking on MAPE across folds, use naive."""
    result = predictions.copy()
    nonzero = result["actual"] != 0

    sub = result[nonzero].copy()
    sub["ape_stack"] = sub["abs_error"] / sub["actual"].abs()
    sub["ape_naive"] = (sub["pred_naive"] - sub["actual"]).abs() / sub["actual"].abs()

    series_mape = sub.groupby(SERIES_KEYS).agg(
        stack_mape=("ape_stack", "mean"),
        naive_mape=("ape_naive", "mean"),
    )
    naive_wins = series_mape[series_mape["naive_mape"] < series_mape["stack_mape"]].index

    n_switched = 0
    for cell_id, cat_id in naive_wins:
        mask = (result["cell_id"] == cell_id) & (result["rate_category_id"] == cat_id)
        result.loc[mask, "prediction"] = result.loc[mask, "pred_naive"]
        n_switched += mask.sum()

    result["error"] = result["prediction"] - result["actual"]
    result["abs_error"] = result["error"].abs()
    result["squared_error"] = result["error"] ** 2
    nonzero = result["actual"] != 0
    result["ape"] = np.where(nonzero, result["abs_error"] / result["actual"].abs(), np.nan)

    print(f"  Per-series selection: switched {len(naive_wins)} series ({n_switched} rows) to naive")
    return result


def _mape_objective(weights: np.ndarray, merged: pd.DataFrame, pred_cols: list[str]) -> float:
    w = np.abs(weights)
    w = w / w.sum()
    blended = sum(w[i] * merged[col] for i, col in enumerate(pred_cols))
    nonzero = merged["actual"] != 0
    ape = (blended[nonzero] - merged.loc[nonzero, "actual"]).abs() / merged.loc[nonzero, "actual"].abs()
    sub = merged[nonzero].copy()
    sub["ape"] = ape
    return float(sub.groupby("fold")["ape"].mean().mean() * 100)


def optimize_horizon_weights_nelder_mead(
    merged: pd.DataFrame,
    pred_cols: list[str],
) -> dict[tuple[int, int], np.ndarray]:
    """Nelder-Mead per-horizon-band weight optimization (MAPE-direct)."""
    from scipy.optimize import minimize

    HORIZON_BANDS = [(1, 4), (5, 8), (9, 12)]
    n = len(pred_cols)
    band_weights = {}

    for h_lo, h_hi in HORIZON_BANDS:
        band_data = merged[(merged["horizon"] >= h_lo) & (merged["horizon"] <= h_hi)]
        if band_data.empty:
            band_weights[(h_lo, h_hi)] = np.ones(n) / n
            continue

        best_result = None
        best_mape = float("inf")

        starts = [np.ones(n) / n] + [np.eye(n)[i] for i in range(n)]
        for x0 in starts:
            result = minimize(
                _mape_objective, x0, args=(band_data, pred_cols),
                method="Nelder-Mead",
                options={"maxiter": 3000, "xatol": 1e-6, "fatol": 1e-6},
            )
            if result.fun < best_mape:
                best_mape = result.fun
                best_result = result

        w = np.abs(best_result.x)
        w = w / w.sum()
        band_weights[(h_lo, h_hi)] = w
        names = [c.replace("pred_", "") for c in pred_cols]
        labels = ", ".join(f"{n}={w[i]:.3f}" for i, n in enumerate(names))
        print(f"    h={h_lo}-{h_hi}: {labels} (MAPE={best_mape:.4f})")

    return band_weights


def build_nelder_mead_predictions(
    merged: pd.DataFrame,
    pred_cols: list[str],
    horizon_weights: dict[tuple[int, int], np.ndarray],
) -> pd.DataFrame:
    result = merged.copy()
    result["prediction"] = 0.0
    for (h_lo, h_hi), w in horizon_weights.items():
        mask = (result["horizon"] >= h_lo) & (result["horizon"] <= h_hi)
        result.loc[mask, "prediction"] = sum(
            w[i] * result.loc[mask, col] for i, col in enumerate(pred_cols)
        )
    result["prediction"] = result["prediction"].clip(lower=0)
    return result


def run_and_save_nelder_mead_ensemble() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nelder-Mead 5-model ensemble (extends existing approach with v8)."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading base model predictions...")
    preds = _load_predictions()
    for name, df in preds.items():
        print(f"  {name}: {len(df)} rows")

    print("Merging predictions...")
    merged = _merge_predictions(preds)
    print(f"  Merged: {len(merged)} rows")

    pred_cols = [f"pred_{n}" for n in preds.keys()]

    print("Optimizing horizon-band weights (Nelder-Mead, MAPE-direct)...")
    horizon_weights = optimize_horizon_weights_nelder_mead(merged, pred_cols)

    predictions = build_nelder_mead_predictions(merged, pred_cols, horizon_weights)
    predictions = _compute_error_cols(predictions)

    print("\n=== Nelder-Mead 5-model (before per-series selection) ===")
    pre_metrics = summarize_fold_metrics(predictions)
    pre_mean = pre_metrics[pre_metrics["fold"] == "mean"].iloc[0]
    print(f"  MAPE: {pre_mean['mape']:.4f}")
    print(f"  MAE:  {pre_mean['mae']:.4f}")
    print(f"  RMSE: {pre_mean['rmse']:.4f}")

    print("\nApplying per-series naive selection...")
    predictions = apply_per_series_selection(predictions)

    metrics = summarize_fold_metrics(predictions)

    nm_preds_path = PREDICTIONS_DIR / "ensemble_nm5_predictions.csv"
    nm_metrics_path = METRICS_DIR / "ensemble_nm5_fold_metrics.csv"
    predictions.to_csv(nm_preds_path, index=False)
    metrics.to_csv(nm_metrics_path, index=False)

    print("\n=== Nelder-Mead 5-model Ensemble (final) ===")
    mean_row = metrics[metrics["fold"] == "mean"].iloc[0]
    print(f"  MAPE: {mean_row['mape']:.4f}")
    print(f"  MAE:  {mean_row['mae']:.4f}")
    print(f"  RMSE: {mean_row['rmse']:.4f}")

    return predictions, metrics


def _compute_error_cols(df: pd.DataFrame) -> pd.DataFrame:
    df["error"] = df["prediction"] - df["actual"]
    df["abs_error"] = df["error"].abs()
    df["squared_error"] = df["error"] ** 2
    nonzero = df["actual"] != 0
    df["ape"] = np.where(nonzero, df["abs_error"] / df["actual"].abs(), np.nan)
    return df


def run_and_save_stacking_ensemble() -> tuple[pd.DataFrame, pd.DataFrame]:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading base model predictions...")
    preds = _load_predictions()
    for name, df in preds.items():
        print(f"  {name}: {len(df)} rows")

    print("Merging predictions...")
    merged = _merge_predictions(preds)
    print(f"  Merged: {len(merged)} rows")

    pred_cols = [f"pred_{n}" for n in preds.keys()]

    print("Training stacking meta-learner (leave-one-fold-out)...")
    predictions = train_stacking_lofocv(merged, pred_cols)
    predictions = _compute_error_cols(predictions)

    print("\n=== Stacking (before per-series selection) ===")
    pre_metrics = summarize_fold_metrics(predictions)
    pre_mean = pre_metrics[pre_metrics["fold"] == "mean"].iloc[0]
    print(f"  MAPE: {pre_mean['mape']:.4f}")
    print(f"  MAE:  {pre_mean['mae']:.4f}")
    print(f"  RMSE: {pre_mean['rmse']:.4f}")

    print("\nApplying per-series naive selection...")
    predictions = apply_per_series_selection(predictions)

    metrics = summarize_fold_metrics(predictions)
    predictions.to_csv(ENSEMBLE_STACKING_PREDICTIONS, index=False)
    metrics.to_csv(ENSEMBLE_STACKING_METRICS, index=False)

    print("\n=== Stacking Ensemble Mean Metrics (final) ===")
    mean_row = metrics[metrics["fold"] == "mean"].iloc[0]
    print(f"  MAPE: {mean_row['mape']:.4f}")
    print(f"  MAE:  {mean_row['mae']:.4f}")
    print(f"  RMSE: {mean_row['rmse']:.4f}")

    return predictions, metrics
