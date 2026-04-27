"""Model v3: Direct multi-horizon LightGBM with spatial features.

One global model with horizon (1-12) as a feature.
All features computed at the forecast origin t — no error accumulation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from .baselines import SERIES_KEYS, summarize_fold_metrics
from .config import (
    METRICS_DIR,
    MODEL_V3_IMPORTANCE,
    MODEL_V3_METRICS,
    MODEL_V3_PREDICTIONS,
    PREDICTIONS_DIR,
    REPORTS_DIR,
)
from .features import (
    ALL_FEATURE_COLS,
    SPATIAL_FEATURE_COLS,
    TARGET_COL,
    _month_to_angle,
    _period_ym_to_month_index,
    _shift_period_ym,
    build_full_feature_frame,
)
from .validation import OFFICIAL_FOLDS, Fold


# ---------------------------------------------------------------------------
# Direct training frame: expand each origin into 12 horizon rows
# ---------------------------------------------------------------------------

def build_direct_training_rows(
    feature_frame: pd.DataFrame,
    fold: Fold,
) -> pd.DataFrame:
    """Build training data for the direct model.

    For each series and each month t in the training period (where t+12 is
    also in the training period), create 12 rows — one per horizon h=1..12.
    Target = y_{t+h}. Features = values at origin t.
    """
    train_mask = (
        (feature_frame["period_ym"] >= fold.train_start)
        & (feature_frame["period_ym"] <= fold.train_end)
    )
    train_data = feature_frame[train_mask].copy()

    # Build lookup: (cell_id, rate_category_id, period_ym) -> target value
    target_lookup = train_data.set_index(
        ["cell_id", "rate_category_id", "period_ym"]
    )[TARGET_COL].to_dict()

    # For each origin row, find which horizons have targets within training period
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

            # Target month calendar features
            target_month = target_ym % 100
            row = {col: origin[col] for col in ALL_FEATURE_COLS if col in origin.index}
            row["horizon"] = h
            row["target_month"] = target_month
            row["target_month_sin"] = float(np.sin(_month_to_angle(target_month)))
            row["target_month_cos"] = float(np.cos(_month_to_angle(target_month)))
            row["target"] = target_val
            # Keep keys for grouping
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
    """Build prediction rows for the validation period.

    For each series, use the last training month as the forecast origin.
    Generate 12 rows (h=1..12) targeting the 12 validation months.
    """
    # Origin = last training month per series
    train_mask = (
        (feature_frame["period_ym"] >= fold.train_start)
        & (feature_frame["period_ym"] <= fold.train_end)
    )
    train_data = feature_frame[train_mask]

    origins = (
        train_data.sort_values("period_ym")
        .groupby(SERIES_KEYS, sort=False)
        .last()
        .reset_index()
    )

    # Actual values in validation period
    valid_mask = (
        (canonical_ts["period_ym"] >= fold.valid_start)
        & (canonical_ts["period_ym"] <= fold.valid_end)
    )
    valid_data = canonical_ts[valid_mask]
    actual_lookup = valid_data.set_index(
        ["cell_id", "rate_category_id", "period_ym"]
    )[TARGET_COL].to_dict()

    rows = []
    for _, origin in origins.iterrows():
        cell = origin["cell_id"]
        cat = origin["rate_category_id"]
        origin_ym = int(origin["period_ym"])

        for h in range(1, 13):
            target_ym = _shift_period_ym(origin_ym, h)
            target_month = target_ym % 100

            actual = actual_lookup.get((cell, cat, target_ym))
            if actual is None:
                continue

            row = {col: origin[col] for col in ALL_FEATURE_COLS if col in origin.index}
            row["horizon"] = h
            row["target_month"] = target_month
            row["target_month_sin"] = float(np.sin(_month_to_angle(target_month)))
            row["target_month_cos"] = float(np.cos(_month_to_angle(target_month)))
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
# Feature column list for training
# ---------------------------------------------------------------------------

def _feature_cols() -> list[str]:
    return ALL_FEATURE_COLS + [
        "horizon", "target_month", "target_month_sin", "target_month_cos",
    ]


def _categorical_cols() -> list[str]:
    return ["rate_category_id", "system_area_id", "h3_resolution", "month", "horizon", "target_month"]


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def build_lightgbm_model() -> lgb.LGBMRegressor:
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
        colsample_bytree=0.8,
        n_estimators=1000,
        random_state=42,
        verbose=-1,
    )


# ---------------------------------------------------------------------------
# Fold evaluation
# ---------------------------------------------------------------------------

def evaluate_fold(
    feature_frame: pd.DataFrame,
    canonical_ts: pd.DataFrame,
    fold: Fold,
) -> tuple[pd.DataFrame, lgb.LGBMRegressor | None]:
    """Train and evaluate on one fold. Returns predictions DataFrame and fitted model."""
    # Build training data
    train_df = build_direct_training_rows(feature_frame, fold)
    if train_df.empty:
        raise ValueError(f"No training rows for {fold.name}")

    feat_cols = _feature_cols()
    cat_cols = _categorical_cols()

    X_train = train_df[feat_cols].copy()
    y_train = np.log1p(train_df["target"].values)

    # Mark categorical features
    for col in cat_cols:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype("category")

    # Train
    model = build_lightgbm_model()
    model.fit(X_train, y_train)

    # Build prediction rows
    pred_df = build_direct_prediction_rows(feature_frame, canonical_ts, fold)
    if pred_df.empty:
        raise ValueError(f"No prediction rows for {fold.name}")

    X_pred = pred_df[feat_cols].copy()
    for col in cat_cols:
        if col in X_pred.columns:
            X_pred[col] = X_pred[col].astype("category")

    raw_preds = np.expm1(model.predict(X_pred))
    pred_df["prediction"] = np.maximum(raw_preds, 0.0)

    # Compute error metrics
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

def run_and_save_model_v3(
    canonical_ts: pd.DataFrame,
    adjacency: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run model v3 on all folds and save artifacts."""
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

    # Save
    predictions.to_csv(MODEL_V3_PREDICTIONS, index=False)
    metrics.to_csv(MODEL_V3_METRICS, index=False)

    # Feature importance
    if last_model is not None:
        feat_cols = _feature_cols()
        importance = pd.DataFrame({
            "feature": feat_cols,
            "importance_gain": last_model.booster_.feature_importance(importance_type="gain"),
            "importance_split": last_model.booster_.feature_importance(importance_type="split"),
        }).sort_values("importance_gain", ascending=False).reset_index(drop=True)
        importance.to_csv(MODEL_V3_IMPORTANCE, index=False)
        print("\nTop 10 features by gain:")
        print(importance.head(10).to_string(index=False))

    print("\n=== Model v3 Mean Metrics ===")
    mean_row = metrics[metrics["fold"] == "mean"].iloc[0]
    print(f"  MAPE: {mean_row['mape']:.4f}")
    print(f"  MAE:  {mean_row['mae']:.4f}")
    print(f"  RMSE: {mean_row['rmse']:.4f}")

    return predictions, metrics
