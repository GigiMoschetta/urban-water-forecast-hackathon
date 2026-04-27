"""Quick Optuna search for v9 model parameters."""

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna

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

# Pre-compute fold-specific feature frames
print("Pre-computing fold features...")
fold_data = {}
for fold in OFFICIAL_FOLDS:
    ff = build_series_meta_features(feature_frame_base.copy(), fold.train_end)
    ff = build_v9_fold_features(ff, fold.train_end)
    train_df = build_direct_training_rows(ff, fold)
    pred_df = build_direct_prediction_rows(ff, canonical_ts, fold)
    fold_data[fold.name] = (train_df, pred_df, fold)
print("  Pre-computation done.")


def objective(trial: optuna.Trial) -> float:
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 1.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.25, 0.7),
        "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
    }

    fold_mapes = []
    for fold_name, (train_df, pred_df, fold) in fold_data.items():
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
        actual = pred_df["actual"].values
        nonzero = actual != 0
        ape = np.where(nonzero, np.abs(raw_preds - actual) / np.abs(actual), np.nan)
        fold_mapes.append(np.nanmean(ape) * 100)

    return np.mean(fold_mapes)


print("\nStarting Optuna search (50 trials)...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"\nBest MAPE: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

# Generate and save best predictions
print("\nGenerating best predictions...")
best_params = study.best_params
all_preds = []
for fold_name, (train_df, pred_df, fold) in fold_data.items():
    X_train = train_df[V9_FEATURE_COLS].copy()
    y_train = np.log1p(train_df["target"].clip(lower=0).values)
    for col in CATEGORICAL_FEATURES:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype("category")

    model = lgb.LGBMRegressor(
        objective="regression", metric="mae", verbosity=-1, random_state=42,
        **best_params
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
    all_preds.append(result)
    mape = result["ape"].mean() * 100
    print(f"  {fold.name}: MAPE={mape:.2f}%")

predictions = pd.concat(all_preds, ignore_index=True)
predictions = predictions.sort_values(["fold", "period_ym"] + SERIES_KEYS).reset_index(drop=True)
metrics = summarize_fold_metrics(predictions)

PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
predictions.to_csv(PREDICTIONS_DIR / "model_v9_optuna_predictions.csv", index=False)
metrics.to_csv(METRICS_DIR / "model_v9_optuna_fold_metrics.csv", index=False)

print("\n=== V9 Optuna-tuned Results ===")
print(metrics.to_string(index=False))
