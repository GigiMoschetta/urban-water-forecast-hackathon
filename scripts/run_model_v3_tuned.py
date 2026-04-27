"""Generate model_v3 predictions using Optuna-tuned best params."""

import numpy as np
import pandas as pd
import lightgbm as lgb

from hackathon_opti.config import CANONICAL_TIMESERIES, CLEAN_ADJACENCY, METRICS_DIR, PREDICTIONS_DIR
from hackathon_opti.baselines import SERIES_KEYS, summarize_fold_metrics
from hackathon_opti.model_v3_direct import (
    build_direct_training_rows,
    build_direct_prediction_rows,
    _feature_cols,
    _categorical_cols,
)
from hackathon_opti.features import build_full_feature_frame
from hackathon_opti.validation import OFFICIAL_FOLDS

BEST_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "verbosity": -1,
    "random_state": 42,
    "learning_rate": 0.0124,
    "num_leaves": 58,
    "max_depth": 7,
    "min_child_samples": 16,
    "reg_lambda": 0.0128,
    "reg_alpha": 0.0140,
    "subsample": 0.804,
    "colsample_bytree": 0.543,
    "n_estimators": 1000,
}

print("Loading data...")
canonical_ts = pd.read_csv(CANONICAL_TIMESERIES)
adjacency = pd.read_csv(CLEAN_ADJACENCY)

print("Building feature frame...")
feature_frame = build_full_feature_frame(canonical_ts, adjacency)

feat_cols = _feature_cols()
cat_cols = _categorical_cols()
all_predictions = []

for fold in OFFICIAL_FOLDS:
    print(f"Processing {fold.name}...")
    train_df = build_direct_training_rows(feature_frame, fold)
    pred_df = build_direct_prediction_rows(feature_frame, canonical_ts, fold)

    X_train = train_df[feat_cols].copy()
    y_train = np.log1p(train_df["target"].values)
    for col in cat_cols:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype("category")

    model = lgb.LGBMRegressor(**BEST_PARAMS)
    model.fit(X_train, y_train)

    X_pred = pred_df[feat_cols].copy()
    for col in cat_cols:
        if col in X_pred.columns:
            X_pred[col] = X_pred[col].astype("category")

    raw_preds = np.maximum(np.expm1(model.predict(X_pred)), 0.0)

    result = pd.DataFrame(
        {
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
        }
    )
    result["error"] = result["prediction"] - result["actual"]
    result["abs_error"] = result["error"].abs()
    result["squared_error"] = result["error"] ** 2
    nonzero = result["actual"] != 0
    result["ape"] = np.where(nonzero, result["abs_error"] / result["actual"].abs(), np.nan)
    all_predictions.append(result)

predictions = pd.concat(all_predictions, ignore_index=True)
predictions = predictions.sort_values(["fold", "period_ym"] + SERIES_KEYS).reset_index(drop=True)
metrics = summarize_fold_metrics(predictions)

METRICS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
predictions.to_csv(PREDICTIONS_DIR / "model_v3_tuned_predictions.csv", index=False)
metrics.to_csv(METRICS_DIR / "model_v3_tuned_fold_metrics.csv", index=False)

print("\nFold metrics (tuned v3):")
print(metrics.to_string(index=False))
