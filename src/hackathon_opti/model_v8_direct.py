"""Model v8: Enhanced direct multi-horizon LightGBM.

Builds on v3 architecture with additional features:
- Series-level meta-features (mean, CV, trend, ACF12) — computed per fold
- Second-order Fourier harmonics (k=2)
- Exponentially weighted means (span 3, 6)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb

from .baselines import SERIES_KEYS, summarize_fold_metrics
from .config import (
    METRICS_DIR,
    MODEL_V8_IMPORTANCE,
    MODEL_V8_METRICS,
    MODEL_V8_PREDICTIONS,
    PREDICTIONS_DIR,
    REPORTS_DIR,
)
from .features import (
    ALL_FEATURE_COLS,
    EWM_FEATURE_COLS,
    FOURIER2_FEATURE_COLS,
    SERIES_META_COLS,
    TARGET_COL,
    _month_to_angle,
    _period_ym_to_month_index,
    _shift_period_ym,
    build_full_feature_frame,
    build_series_meta_features,
)
from .validation import OFFICIAL_FOLDS, Fold

# ---------------------------------------------------------------------------
# Feature columns
# ---------------------------------------------------------------------------

BASE_FEATURES = ALL_FEATURE_COLS + EWM_FEATURE_COLS + FOURIER2_FEATURE_COLS + SERIES_META_COLS

DIRECT_FEATURES = [
    "horizon",
    "target_month",
    "target_month_sin",
    "target_month_cos",
    "target_month_sin2",
    "target_month_cos2",
]

V8_FEATURE_COLS = BASE_FEATURES + DIRECT_FEATURES

CATEGORICAL_FEATURES = [
    "rate_category_id",
    "system_area_id",
    "h3_resolution",
    "month",
    "horizon",
    "target_month",
]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression",
        metric="mae",
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        min_child_samples=30,
        reg_lambda=0.1,
        reg_alpha=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        n_estimators=1200,
        random_state=42,
        verbose=-1,
    )


# ---------------------------------------------------------------------------
# Direct training/prediction row builders
# ---------------------------------------------------------------------------

def _attach_direct_features(row: dict, target_ym: int, horizon: int) -> None:
    target_month = target_ym % 100
    row["horizon"] = horizon
    row["target_month"] = target_month
    row["target_month_sin"] = float(np.sin(_month_to_angle(target_month)))
    row["target_month_cos"] = float(np.cos(_month_to_angle(target_month)))
    row["target_month_sin2"] = float(np.sin(2 * _month_to_angle(target_month)))
    row["target_month_cos2"] = float(np.cos(2 * _month_to_angle(target_month)))


def build_direct_training_rows(
    feature_frame: pd.DataFrame,
    fold: Fold,
) -> pd.DataFrame:
    train_mask = (
        (feature_frame["period_ym"] >= fold.train_start)
        & (feature_frame["period_ym"] <= fold.train_end)
    )
    train_data = feature_frame[train_mask].copy()
    target_lookup = train_data.set_index(
        ["cell_id", "rate_category_id", "period_ym"]
    )[TARGET_COL].to_dict()

    rows = []
    for _, origin in train_data.iterrows():
        cell = origin["cell_id"]
        cat = origin["rate_category_id"]
        origin_ym = int(origin["period_ym"])

        for h in range(1, 13):
            target_ym = _shift_period_ym(origin_ym, h)
            if target_ym > fold.train_end:
                break

            target_val = target_lookup.get((cell, cat, target_ym))
            if target_val is None or np.isnan(target_val):
                continue

            row = {col: origin[col] for col in BASE_FEATURES if col in origin.index}
            _attach_direct_features(row, target_ym, h)
            row["target"] = target_val
            row["cell_id"] = cell
            row["rate_category_id"] = cat
            row["origin_ym"] = origin_ym
            row["target_ym"] = target_ym
            rows.append(row)

    return pd.DataFrame(rows)


def build_direct_prediction_rows(
    feature_frame: pd.DataFrame,
    canonical_ts: pd.DataFrame,
    fold: Fold,
) -> pd.DataFrame:
    train_mask = (
        (feature_frame["period_ym"] >= fold.train_start)
        & (feature_frame["period_ym"] <= fold.train_end)
    )
    origins = (
        feature_frame[train_mask]
        .sort_values("period_ym")
        .groupby(SERIES_KEYS, sort=False)
        .last()
        .reset_index()
    )

    valid_mask = (
        (canonical_ts["period_ym"] >= fold.valid_start)
        & (canonical_ts["period_ym"] <= fold.valid_end)
    )
    actual_lookup = canonical_ts[valid_mask].set_index(
        ["cell_id", "rate_category_id", "period_ym"]
    )[TARGET_COL].to_dict()

    rows = []
    for _, origin in origins.iterrows():
        cell = origin["cell_id"]
        cat = origin["rate_category_id"]
        origin_ym = int(origin["period_ym"])

        for h in range(1, 13):
            target_ym = _shift_period_ym(origin_ym, h)
            actual = actual_lookup.get((cell, cat, target_ym))
            if actual is None:
                continue

            row = {col: origin[col] for col in BASE_FEATURES if col in origin.index}
            _attach_direct_features(row, target_ym, h)
            row["cell_id"] = cell
            row["rate_category_id"] = cat
            row["origin_ym"] = origin_ym
            row["target_ym"] = target_ym
            row["actual"] = actual
            row["system_area"] = origin.get("system_area", "")
            row["h3_resolution"] = origin.get("h3_resolution", 0)
            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fold evaluation
# ---------------------------------------------------------------------------

def evaluate_fold(
    feature_frame_base: pd.DataFrame,
    canonical_ts: pd.DataFrame,
    fold: Fold,
) -> tuple[pd.DataFrame, lgb.LGBMRegressor | None]:
    # Compute fold-specific series meta-features
    feature_frame = build_series_meta_features(feature_frame_base.copy(), fold.train_end)

    train_df = build_direct_training_rows(feature_frame, fold)
    if train_df.empty:
        raise ValueError(f"No training rows for {fold.name}")

    X_train = train_df[V8_FEATURE_COLS].copy()
    y_train = np.log1p(train_df["target"].clip(lower=0).values)

    for col in CATEGORICAL_FEATURES:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype("category")

    model = build_model()
    model.fit(X_train, y_train)

    pred_df = build_direct_prediction_rows(feature_frame, canonical_ts, fold)
    if pred_df.empty:
        raise ValueError(f"No prediction rows for {fold.name}")

    X_pred = pred_df[V8_FEATURE_COLS].copy()
    for col in CATEGORICAL_FEATURES:
        if col in X_pred.columns:
            X_pred[col] = X_pred[col].astype("category")

    pred_df["prediction"] = np.maximum(np.expm1(model.predict(X_pred)), 0.0)

    result = pd.DataFrame({
        "fold": fold.name,
        "forecast_origin_ym": fold.train_end,
        "horizon": pred_df["horizon"].values,
        "cell_id": pred_df["cell_id"].values,
        "rate_category_id": pred_df["rate_category_id"].values,
        "period_ym": pred_df["target_ym"].values,
        "system_area": pred_df["system_area"].values,
        "h3_resolution": pred_df["h3_resolution"].values,
        "actual": pred_df["actual"].values,
        "prediction": pred_df["prediction"].values,
    })
    result["error"] = result["prediction"] - result["actual"]
    result["abs_error"] = result["error"].abs()
    result["squared_error"] = result["error"] ** 2
    nonzero = result["actual"] != 0
    result["ape"] = np.where(nonzero, result["abs_error"] / result["actual"].abs(), np.nan)

    return result, model


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------

def run_and_save_model_v8(
    canonical_ts: pd.DataFrame,
    adjacency: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Building full feature frame...")
    feature_frame = build_full_feature_frame(canonical_ts, adjacency)
    print(f"  Feature frame: {len(feature_frame)} rows, {len(feature_frame.columns)} columns")

    all_predictions = []
    last_model = None

    for fold in OFFICIAL_FOLDS:
        print(f"\nEvaluating {fold.name}...")
        preds, model = evaluate_fold(feature_frame, canonical_ts, fold)
        all_predictions.append(preds)
        last_model = model
        mape = preds["ape"].mean() * 100
        mae = preds["abs_error"].mean()
        rmse = np.sqrt(preds["squared_error"].mean())
        print(f"  {fold.name}: MAPE={mape:.2f}%, MAE={mae:.2f}, RMSE={rmse:.2f}")

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions = predictions.sort_values(["fold", "period_ym"] + SERIES_KEYS).reset_index(drop=True)
    metrics = summarize_fold_metrics(predictions)

    predictions.to_csv(MODEL_V8_PREDICTIONS, index=False)
    metrics.to_csv(MODEL_V8_METRICS, index=False)

    if last_model is not None:
        importance = pd.DataFrame({
            "feature": V8_FEATURE_COLS,
            "importance_gain": last_model.booster_.feature_importance(importance_type="gain"),
            "importance_split": last_model.booster_.feature_importance(importance_type="split"),
        }).sort_values("importance_gain", ascending=False).reset_index(drop=True)
        importance.to_csv(MODEL_V8_IMPORTANCE, index=False)
        print("\nTop 15 features by gain:")
        print(importance.head(15).to_string(index=False))

    print("\n=== Model v8 Mean Metrics ===")
    mean_row = metrics[metrics["fold"] == "mean"].iloc[0]
    print(f"  MAPE: {mean_row['mape']:.4f}")
    print(f"  MAE:  {mean_row['mae']:.4f}")
    print(f"  RMSE: {mean_row['rmse']:.4f}")

    return predictions, metrics
