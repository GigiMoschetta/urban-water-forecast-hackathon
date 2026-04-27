"""Optuna hyperparameter tuning for Model v8 (enhanced direct LightGBM).

Optimizes mean MAPE across 4 official folds.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

from hackathon_opti.config import CANONICAL_TIMESERIES, CLEAN_ADJACENCY, METRICS_DIR, PREDICTIONS_DIR
from hackathon_opti.baselines import SERIES_KEYS, summarize_fold_metrics
from hackathon_opti.model_v8_direct import (
    build_direct_training_rows,
    build_direct_prediction_rows,
    V8_FEATURE_COLS,
    CATEGORICAL_FEATURES,
)
from hackathon_opti.features import build_full_feature_frame, build_series_meta_features
from hackathon_opti.validation import OFFICIAL_FOLDS

# Load data once
print("Loading data...")
canonical_ts = pd.read_csv(CANONICAL_TIMESERIES)
adjacency = pd.read_csv(CLEAN_ADJACENCY)

print("Building feature frame...")
feature_frame_base = build_full_feature_frame(canonical_ts, adjacency)

# Pre-build fold data with per-fold meta features
print("Pre-building fold data...")
fold_data = []
for fold in OFFICIAL_FOLDS:
    ff = build_series_meta_features(feature_frame_base.copy(), fold.train_end)
    train_df = build_direct_training_rows(ff, fold)
    pred_df = build_direct_prediction_rows(ff, canonical_ts, fold)
    fold_data.append((fold, train_df, pred_df))
print("Ready for tuning.\n")


def objective(trial: optuna.Trial) -> float:
    params = {
        "objective": "regression",
        "metric": "mae",
        "verbosity": -1,
        "random_state": 42,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 1.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "n_estimators": trial.suggest_int("n_estimators", 500, 2000, step=100),
    }

    fold_mapes = []
    for fold, train_df, pred_df in fold_data:
        X_train = train_df[V8_FEATURE_COLS].copy()
        y_train = np.log1p(train_df["target"].clip(lower=0).values)
        for col in CATEGORICAL_FEATURES:
            if col in X_train.columns:
                X_train[col] = X_train[col].astype("category")

        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)

        X_pred = pred_df[V8_FEATURE_COLS].copy()
        for col in CATEGORICAL_FEATURES:
            if col in X_pred.columns:
                X_pred[col] = X_pred[col].astype("category")

        preds = np.maximum(np.expm1(model.predict(X_pred)), 0.0)
        actual = pred_df["actual"].values
        nonzero = actual != 0
        ape = np.where(nonzero, np.abs(preds - actual) / np.abs(actual), np.nan)
        fold_mapes.append(np.nanmean(ape) * 100)

    mean_mape = np.mean(fold_mapes)
    for i, m in enumerate(fold_mapes):
        trial.set_user_attr(f"fold_{i+1}_mape", m)

    return mean_mape


study = optuna.create_study(direction="minimize", study_name="v8_tuning")
study.optimize(objective, n_trials=80, show_progress_bar=True)

print(f"\n{'='*60}")
print(f"Best MAPE: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
for k, v in study.best_trial.user_attrs.items():
    print(f"  {k}: {v:.2f}")

# Save best model predictions
print("\nGenerating predictions with best params...")
best_params = {
    "objective": "regression",
    "metric": "mae",
    "verbosity": -1,
    "random_state": 42,
    **study.best_params,
}

all_predictions = []
for fold, train_df, pred_df in fold_data:
    X_train = train_df[V8_FEATURE_COLS].copy()
    y_train = np.log1p(train_df["target"].clip(lower=0).values)
    for col in CATEGORICAL_FEATURES:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype("category")

    model = lgb.LGBMRegressor(**best_params)
    model.fit(X_train, y_train)

    X_pred = pred_df[V8_FEATURE_COLS].copy()
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
    all_predictions.append(result)

predictions = pd.concat(all_predictions, ignore_index=True)
predictions = predictions.sort_values(["fold", "period_ym"] + SERIES_KEYS).reset_index(drop=True)
metrics = summarize_fold_metrics(predictions)

METRICS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
predictions.to_csv(PREDICTIONS_DIR / "model_v8_tuned_predictions.csv", index=False)
metrics.to_csv(METRICS_DIR / "model_v8_tuned_fold_metrics.csv", index=False)

print("\nFold metrics (tuned v8):")
print(metrics.to_string(index=False))
