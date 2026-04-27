"""Optuna hyperparameter tuning for Model v1 (HistGradientBoosting recursive).

Optimizes mean MAPE across 4 official folds.
Recursive prediction makes each trial slower than direct models (~2-3 min/trial).
"""

import numpy as np
import pandas as pd
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

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

# Load data once
print("Loading data...")
canonical_ts = pd.read_csv(CANONICAL_TIMESERIES)
canonical = canonical_ts.sort_values(SERIES_KEYS + ["period_ym"]).reset_index(drop=True)

print("Building training frame...")
training_frame = build_training_frame(canonical)

feat_cols = feature_columns()
categorical_columns = {"rate_category_id", "system_area_id", "h3_resolution", "month"}
cat_mask = [col in categorical_columns for col in feat_cols]

# Pre-build fold splits
print("Pre-building fold data...")
fold_data = []
for fold in OFFICIAL_FOLDS:
    fold_train = training_frame[
        (training_frame["period_ym"] >= fold.train_start)
        & (training_frame["period_ym"] <= fold.train_end)
    ].copy()
    train_split, valid_split = split_by_fold(canonical, fold)
    valid_by_series = {
        key: chunk.sort_values("period_ym").reset_index(drop=True)
        for key, chunk in valid_split.groupby(SERIES_KEYS, sort=False)
    }
    fold_data.append((fold, fold_train, train_split, valid_by_series))
print("Ready for tuning.\n")


def _recursive_predict(model, train_split, valid_by_series, fold):
    """Run recursive 12-step prediction for one fold, return MAPE."""
    ape_values = []
    for key, train_series in train_split.groupby(SERIES_KEYS, sort=False):
        history = train_series.sort_values("period_ym").reset_index(drop=True)
        metadata = _series_metadata(history)
        history_values = history[TARGET_COL].astype(float).tolist()
        target_rows = valid_by_series.get(key)
        if target_rows is None or target_rows.empty:
            continue

        next_period = _next_period_ym(int(history["period_ym"].iloc[-1]))
        for _, target_row in target_rows.iterrows():
            target_period = int(target_row["period_ym"])
            if target_period != next_period:
                continue

            feature_row = _build_recursive_feature_row(metadata, target_period, history_values)
            prediction = float(model.predict(pd.DataFrame([feature_row]))[0])
            prediction = max(prediction, 0.0)
            history_values.append(prediction)
            next_period = _next_period_ym(target_period)

            actual = float(target_row[TARGET_COL])
            if actual != 0:
                ape_values.append(abs(prediction - actual) / abs(actual))

    return np.mean(ape_values) * 100 if ape_values else 100.0


def objective(trial: optuna.Trial) -> float:
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "max_iter": trial.suggest_int("max_iter", 80, 400, step=20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 15, 100),
        "l2_regularization": trial.suggest_float("l2_regularization", 0.01, 2.0, log=True),
        "max_bins": trial.suggest_int("max_bins", 127, 255, step=64),
        "early_stopping": False,
        "categorical_features": cat_mask,
        "random_state": 42,
    }

    regressor = HistGradientBoostingRegressor(**params)
    model = TransformedTargetRegressor(
        regressor=regressor, func=np.log1p, inverse_func=np.expm1, check_inverse=False
    )

    fold_mapes = []
    for fold, fold_train, train_split, valid_by_series in fold_data:
        model.fit(fold_train[feat_cols], fold_train[TARGET_COL])
        mape = _recursive_predict(model, train_split, valid_by_series, fold)
        fold_mapes.append(mape)

    mean_mape = np.mean(fold_mapes)
    for i, m in enumerate(fold_mapes):
        trial.set_user_attr(f"fold_{i+1}_mape", m)

    return mean_mape


study = optuna.create_study(direction="minimize", study_name="v1_tuning")
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"\n{'='*60}")
print(f"Best MAPE: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
for k, v in study.best_trial.user_attrs.items():
    print(f"  {k}: {v:.2f}")

# Save best model predictions
print("\nGenerating predictions with best params...")
best_params = {
    **study.best_params,
    "early_stopping": False,
    "categorical_features": cat_mask,
    "random_state": 42,
}

all_predictions = []
for fold, fold_train, train_split, valid_by_series in fold_data:
    regressor = HistGradientBoostingRegressor(**best_params)
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

METRICS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
predictions.to_csv(PREDICTIONS_DIR / "model_v1_tuned_predictions.csv", index=False)
metrics.to_csv(METRICS_DIR / "model_v1_tuned_fold_metrics.csv", index=False)

print("\nFold metrics (tuned v1):")
print(metrics.to_string(index=False))

# Compare with original
print("\n=== Comparison with original v1 ===")
orig = pd.read_csv(METRICS_DIR / "model_v1_fold_metrics.csv")
orig_mean = orig[orig["fold"] == "mean"].iloc[0]
tuned_mean = metrics[metrics["fold"] == "mean"].iloc[0]
print(f"  Original: MAPE={orig_mean['mape']:.4f}  MAE={orig_mean['mae']:.2f}  RMSE={orig_mean['rmse']:.2f}")
print(f"  Tuned:    MAPE={tuned_mean['mape']:.4f}  MAE={tuned_mean['mae']:.2f}  RMSE={tuned_mean['rmse']:.2f}")
