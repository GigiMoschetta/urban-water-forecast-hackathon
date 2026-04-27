"""Quick parameter sweep for v9 to find better params for the larger feature set."""

import numpy as np
import pandas as pd
import lightgbm as lgb

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
)
from hackathon_opti.validation import OFFICIAL_FOLDS

print("Loading data...")
canonical_ts = pd.read_csv(CANONICAL_TIMESERIES)
adjacency = pd.read_csv(CLEAN_ADJACENCY)

print("Building feature frame...")
feature_frame_base = build_full_feature_frame(canonical_ts, adjacency)
feature_frame_base = build_v9_origin_features(feature_frame_base, adjacency)

# Parameter configs to try
CONFIGS = {
    "v8_params": {
        "learning_rate": 0.0173, "num_leaves": 38, "max_depth": 4,
        "min_child_samples": 26, "reg_lambda": 0.0997, "reg_alpha": 0.0677,
        "subsample": 0.754, "colsample_bytree": 0.558, "n_estimators": 600,
    },
    "more_trees": {
        "learning_rate": 0.0173, "num_leaves": 38, "max_depth": 4,
        "min_child_samples": 26, "reg_lambda": 0.0997, "reg_alpha": 0.0677,
        "subsample": 0.754, "colsample_bytree": 0.558, "n_estimators": 1000,
    },
    "lower_colsample": {
        "learning_rate": 0.0173, "num_leaves": 38, "max_depth": 4,
        "min_child_samples": 26, "reg_lambda": 0.0997, "reg_alpha": 0.0677,
        "subsample": 0.754, "colsample_bytree": 0.4, "n_estimators": 800,
    },
    "more_leaves": {
        "learning_rate": 0.015, "num_leaves": 50, "max_depth": 5,
        "min_child_samples": 20, "reg_lambda": 0.05, "reg_alpha": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.5, "n_estimators": 800,
    },
    "conservative": {
        "learning_rate": 0.01, "num_leaves": 31, "max_depth": 4,
        "min_child_samples": 30, "reg_lambda": 0.2, "reg_alpha": 0.1,
        "subsample": 0.7, "colsample_bytree": 0.45, "n_estimators": 1200,
    },
}

results = {}
best_mape = float("inf")
best_config = None
best_preds = None

for name, params in CONFIGS.items():
    print(f"\n--- Config: {name} ---")
    fold_mapes = []
    fold_preds = []

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

        model = lgb.LGBMRegressor(
            objective="regression", metric="mae", verbosity=-1, random_state=42,
            **params
        )
        model.fit(X_train, y_train)

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
        fold_preds.append(result)

    mean_mape = np.mean(fold_mapes)
    results[name] = {"fold_mapes": fold_mapes, "mean_mape": mean_mape}
    print(f"  Folds: {' | '.join(f'{m:.2f}' for m in fold_mapes)} | Mean: {mean_mape:.4f}")

    if mean_mape < best_mape:
        best_mape = mean_mape
        best_config = name
        best_preds = pd.concat(fold_preds, ignore_index=True)

print(f"\n{'='*60}")
print(f"BEST CONFIG: {best_config} (MAPE {best_mape:.4f})")
print(f"{'='*60}")

# Save best predictions
if best_preds is not None:
    best_preds = best_preds.sort_values(["fold", "period_ym"] + SERIES_KEYS).reset_index(drop=True)
    metrics = summarize_fold_metrics(best_preds)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    best_preds.to_csv(PREDICTIONS_DIR / "model_v9_tuned_predictions.csv", index=False)
    metrics.to_csv(METRICS_DIR / "model_v9_tuned_fold_metrics.csv", index=False)
    print("\nSaved best v9 predictions.")
    print(metrics.to_string(index=False))
