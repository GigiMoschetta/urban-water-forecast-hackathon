"""V9 with sample weighting: temporal decay + data quality multipliers."""

import numpy as np
import pandas as pd
import lightgbm as lgb
from itertools import product

from hackathon_opti.config import CANONICAL_TIMESERIES, CLEAN_ADJACENCY, METRICS_DIR, PREDICTIONS_DIR
from hackathon_opti.baselines import SERIES_KEYS, summarize_fold_metrics
from hackathon_opti.model_v9_enhanced import (
    build_direct_training_rows,
    build_direct_prediction_rows,
    V9_FEATURE_COLS,
    CATEGORICAL_FEATURES,
)
from hackathon_opti.features import (
    build_full_feature_frame,
    build_series_meta_features,
    build_v9_origin_features,
    build_v9_fold_features,
    _period_ym_to_month_index,
)
from hackathon_opti.validation import OFFICIAL_FOLDS

BEST_PARAMS = {
    "learning_rate": 0.0173,
    "num_leaves": 38,
    "max_depth": 4,
    "min_child_samples": 26,
    "reg_lambda": 0.0997,
    "reg_alpha": 0.0677,
    "subsample": 0.754,
    "colsample_bytree": 0.4,
    "n_estimators": 800,
}


def compute_sample_weights(
    train_df: pd.DataFrame,
    fold_train_end: int,
    decay_lambda: float,
    march_boost: float,
    smart_boost: float,
    drought_penalty: float,
    reconciliation_boost: float,
) -> np.ndarray:
    """Compute sample weights combining temporal decay and data quality multipliers."""
    origin_ym = train_df["origin_ym"].values
    month_idx = np.array([_period_ym_to_month_index(pd.Series([ym])).iloc[0] for ym in origin_ym])
    max_idx = _period_ym_to_month_index(pd.Series([fold_train_end])).iloc[0]
    months_ago = max_idx - month_idx

    # Base temporal decay
    weights = np.exp(-decay_lambda * months_ago)

    # Data quality multipliers
    origin_month = origin_ym % 100
    is_march = (origin_month == 3)
    is_reconciliation = np.isin(origin_month, [6, 9, 12])  # Mar already handled
    is_drought = (origin_ym >= 202205) & (origin_ym <= 202211)
    is_smart_era = (origin_ym >= 202301)

    weights = weights * np.where(is_march, march_boost, 1.0)
    weights = weights * np.where(is_reconciliation, reconciliation_boost, 1.0)
    weights = weights * np.where(is_drought, drought_penalty, 1.0)
    weights = weights * np.where(is_smart_era, smart_boost, 1.0)

    return weights / weights.mean()  # normalize so mean weight ≈ 1


print("Loading data...")
canonical_ts = pd.read_csv(CANONICAL_TIMESERIES)
adjacency = pd.read_csv(CLEAN_ADJACENCY)

print("Building feature frame...")
feature_frame_base = build_full_feature_frame(canonical_ts, adjacency)
feature_frame_base = build_v9_origin_features(feature_frame_base, adjacency)

# Grid search configs
WEIGHT_CONFIGS = {
    "no_weights": {"decay_lambda": 0, "march_boost": 1.0, "smart_boost": 1.0, "drought_penalty": 1.0, "reconciliation_boost": 1.0},
    "decay_only_002": {"decay_lambda": 0.02, "march_boost": 1.0, "smart_boost": 1.0, "drought_penalty": 1.0, "reconciliation_boost": 1.0},
    "decay_only_003": {"decay_lambda": 0.03, "march_boost": 1.0, "smart_boost": 1.0, "drought_penalty": 1.0, "reconciliation_boost": 1.0},
    "quality_only": {"decay_lambda": 0, "march_boost": 1.5, "smart_boost": 1.3, "drought_penalty": 0.5, "reconciliation_boost": 1.2},
    "decay02_quality": {"decay_lambda": 0.02, "march_boost": 1.5, "smart_boost": 1.3, "drought_penalty": 0.5, "reconciliation_boost": 1.2},
    "decay03_quality": {"decay_lambda": 0.03, "march_boost": 1.5, "smart_boost": 1.3, "drought_penalty": 0.5, "reconciliation_boost": 1.2},
    "aggressive_quality": {"decay_lambda": 0.02, "march_boost": 2.0, "smart_boost": 1.5, "drought_penalty": 0.3, "reconciliation_boost": 1.3},
    "mild_drought_only": {"decay_lambda": 0, "march_boost": 1.0, "smart_boost": 1.0, "drought_penalty": 0.5, "reconciliation_boost": 1.0},
}

best_mape = float("inf")
best_config_name = None
best_preds = None

for config_name, wparams in WEIGHT_CONFIGS.items():
    fold_mapes = []
    fold_preds_list = []

    for fold in OFFICIAL_FOLDS:
        feature_frame = build_series_meta_features(feature_frame_base.copy(), fold.train_end)
        feature_frame = build_v9_fold_features(feature_frame, fold.train_end)

        train_df = build_direct_training_rows(feature_frame, fold)
        pred_df = build_direct_prediction_rows(feature_frame, canonical_ts, fold)

        X_train = train_df[V9_FEATURE_COLS].copy()
        y_train = np.log1p(train_df["target"].clip(lower=0).values)
        for col in CATEGORICAL_FEATURES:
            if col in X_train.columns:
                X_train[col] = X_train[col].astype("category")

        # Compute sample weights
        weights = compute_sample_weights(train_df, fold.train_end, **wparams)

        model = lgb.LGBMRegressor(
            objective="regression", metric="mae", verbosity=-1, random_state=42,
            **BEST_PARAMS
        )
        model.fit(X_train, y_train, sample_weight=weights)

        X_pred = pred_df[V9_FEATURE_COLS].copy()
        for col in CATEGORICAL_FEATURES:
            if col in X_pred.columns:
                X_pred[col] = X_pred[col].astype("category")

        raw_preds = np.maximum(np.expm1(model.predict(X_pred)), 0.0)

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
            "prediction": raw_preds,
        })
        result["error"] = result["prediction"] - result["actual"]
        result["abs_error"] = result["error"].abs()
        result["squared_error"] = result["error"] ** 2
        nonzero = result["actual"] != 0
        result["ape"] = np.where(nonzero, result["abs_error"] / result["actual"].abs(), np.nan)
        fold_mapes.append(result["ape"].mean() * 100)
        fold_preds_list.append(result)

    mean_mape = np.mean(fold_mapes)
    print(f"{config_name:25s}: {' | '.join(f'{m:.2f}' for m in fold_mapes)} | Mean: {mean_mape:.4f}")

    if mean_mape < best_mape:
        best_mape = mean_mape
        best_config_name = config_name
        best_preds = pd.concat(fold_preds_list, ignore_index=True)

print(f"\n{'='*60}")
print(f"BEST: {best_config_name} (MAPE {best_mape:.4f})")
print(f"{'='*60}")

# Save best weighted v9
if best_preds is not None:
    best_preds = best_preds.sort_values(["fold", "period_ym"] + SERIES_KEYS).reset_index(drop=True)
    metrics = summarize_fold_metrics(best_preds)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    best_preds.to_csv(PREDICTIONS_DIR / "model_v9_weighted_predictions.csv", index=False)
    metrics.to_csv(METRICS_DIR / "model_v9_weighted_fold_metrics.csv", index=False)
    print(metrics.to_string(index=False))
