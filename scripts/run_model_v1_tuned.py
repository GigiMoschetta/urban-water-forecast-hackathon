"""Generate model_v1 predictions using Optuna-tuned best params (from overnight run)."""

import numpy as np
import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

from hackathon_opti.config import CANONICAL_TIMESERIES, METRICS_DIR, PREDICTIONS_DIR
from hackathon_opti.baselines import SERIES_KEYS, summarize_fold_metrics
from hackathon_opti.model_v1 import (
    build_training_frame,
    feature_columns,
    _series_metadata,
    _build_recursive_feature_row,
    _next_period_ym,
    TARGET_COL,
)
from hackathon_opti.validation import OFFICIAL_FOLDS, split_by_fold

# Best params from Optuna overnight run
BEST_PARAMS = {
    "learning_rate": 0.05887,
    "max_depth": 4,
    "max_iter": 140,
    "min_samples_leaf": 48,
    "l2_regularization": 0.08937,
    "max_bins": 191,
    "early_stopping": False,
    "random_state": 42,
}

print("Loading data...")
canonical_ts = pd.read_csv(CANONICAL_TIMESERIES)
canonical = canonical_ts.sort_values(SERIES_KEYS + ["period_ym"]).reset_index(drop=True)

training_frame = build_training_frame(canonical)
feat_cols = feature_columns()
categorical_columns = {"rate_category_id", "system_area_id", "h3_resolution", "month"}
cat_mask = [col in categorical_columns for col in feat_cols]
BEST_PARAMS["categorical_features"] = cat_mask

all_predictions = []
for fold in OFFICIAL_FOLDS:
    print(f"Processing {fold.name}...")
    fold_train = training_frame[
        (training_frame["period_ym"] >= fold.train_start)
        & (training_frame["period_ym"] <= fold.train_end)
    ].copy()
    train_split, valid_split = split_by_fold(canonical, fold)
    valid_by_series = {
        key: chunk.sort_values("period_ym").reset_index(drop=True)
        for key, chunk in valid_split.groupby(SERIES_KEYS, sort=False)
    }

    regressor = HistGradientBoostingRegressor(**BEST_PARAMS)
    model = TransformedTargetRegressor(
        regressor=regressor, func=np.log1p, inverse_func=np.expm1, check_inverse=False
    )
    model.fit(fold_train[feat_cols], fold_train[TARGET_COL])

    outputs = []
    for key, train_series in train_split.groupby(SERIES_KEYS, sort=False):
        history = train_series.sort_values("period_ym").reset_index(drop=True)
        metadata = _series_metadata(history)
        history_values = history[TARGET_COL].astype(float).tolist()
        target_rows = valid_by_series.get(key)
        if target_rows is None or target_rows.empty:
            continue

        next_period = _next_period_ym(int(history["period_ym"].iloc[-1]))
        for horizon, (_, target_row) in enumerate(target_rows.iterrows(), start=1):
            target_period = int(target_row["period_ym"])
            if target_period != next_period:
                break
            feature_row = _build_recursive_feature_row(metadata, target_period, history_values)
            prediction = float(model.predict(pd.DataFrame([feature_row]))[0])
            prediction = max(prediction, 0.0)
            history_values.append(prediction)
            next_period = _next_period_ym(target_period)

            outputs.append({
                "fold": fold.name,
                "forecast_origin_ym": fold.train_end,
                "horizon": horizon,
                "cell_id": metadata.cell_id,
                "rate_category_id": metadata.rate_category_id,
                "period_ym": target_period,
                "system_area": metadata.system_area,
                "h3_resolution": metadata.h3_resolution,
                "actual": float(target_row[TARGET_COL]),
                "prediction": prediction,
            })

    fold_preds = pd.DataFrame(outputs)
    fold_preds["error"] = fold_preds["prediction"] - fold_preds["actual"]
    fold_preds["abs_error"] = fold_preds["error"].abs()
    fold_preds["squared_error"] = fold_preds["error"] ** 2
    nonzero = fold_preds["actual"] != 0
    fold_preds["ape"] = np.where(nonzero, fold_preds["abs_error"] / fold_preds["actual"].abs(), np.nan)
    all_predictions.append(fold_preds)

predictions = pd.concat(all_predictions, ignore_index=True)
predictions = predictions.sort_values(["fold", "period_ym"] + SERIES_KEYS).reset_index(drop=True)
metrics = summarize_fold_metrics(predictions)

predictions.to_csv(PREDICTIONS_DIR / "model_v1_tuned_predictions.csv", index=False)
metrics.to_csv(METRICS_DIR / "model_v1_tuned_fold_metrics.csv", index=False)

print("\nFold metrics (tuned v1):")
print(metrics.to_string(index=False))

orig = pd.read_csv(METRICS_DIR / "model_v1_fold_metrics.csv")
orig_mean = orig[orig["fold"] == "mean"].iloc[0]
tuned_mean = metrics[metrics["fold"] == "mean"].iloc[0]
print(f"\n  Original: MAPE={orig_mean['mape']:.4f}")
print(f"  Tuned:    MAPE={tuned_mean['mape']:.4f}")
