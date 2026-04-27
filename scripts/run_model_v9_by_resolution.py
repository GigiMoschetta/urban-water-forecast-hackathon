"""Train separate v9 models for resolution 6 and 7, then combine predictions."""

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

# Resolution-specific parameters
# Res 6: 69 series, macro/weather-driven — needs more regularization
RES6_PARAMS = {
    "learning_rate": 0.01,
    "num_leaves": 20,
    "max_depth": 3,
    "min_child_samples": 40,
    "reg_lambda": 0.3,
    "reg_alpha": 0.2,
    "subsample": 0.7,
    "colsample_bytree": 0.35,
    "n_estimators": 800,
}

# Res 7: 286 series, micro/behavior-driven — can afford more complexity
RES7_PARAMS = {
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

print("Loading data...")
canonical_ts = pd.read_csv(CANONICAL_TIMESERIES)
adjacency = pd.read_csv(CLEAN_ADJACENCY)

print("Building feature frame...")
feature_frame_base = build_full_feature_frame(canonical_ts, adjacency)
feature_frame_base = build_v9_origin_features(feature_frame_base, adjacency)

all_fold_preds = []

for fold in OFFICIAL_FOLDS:
    print(f"\n{'='*50}")
    print(f"Processing {fold.name}...")
    feature_frame = build_series_meta_features(feature_frame_base.copy(), fold.train_end)
    feature_frame = build_v9_fold_features(feature_frame, fold.train_end)

    for res, params in [(6, RES6_PARAMS), (7, RES7_PARAMS)]:
        # Filter feature frame by resolution
        ff_res = feature_frame[feature_frame["h3_resolution"] == res].copy()
        ct_res = canonical_ts[canonical_ts["h3_resolution"] == res].copy()

        train_df = build_direct_training_rows(ff_res, fold)
        pred_df = build_direct_prediction_rows(ff_res, ct_res, fold)

        if train_df.empty or pred_df.empty:
            print(f"  Res {res}: skipped (no data)")
            continue

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

        mape = result["ape"].mean() * 100
        n = len(result)
        print(f"  Res {res}: {n} rows, MAPE={mape:.2f}%")
        all_fold_preds.append(result)

# Combine
predictions = pd.concat(all_fold_preds, ignore_index=True)
predictions = predictions.sort_values(["fold", "period_ym"] + SERIES_KEYS).reset_index(drop=True)
metrics = summarize_fold_metrics(predictions)

PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
predictions.to_csv(PREDICTIONS_DIR / "model_v9_by_res_predictions.csv", index=False)
metrics.to_csv(METRICS_DIR / "model_v9_by_res_fold_metrics.csv", index=False)

print(f"\n{'='*60}")
print("=== Combined Resolution-Specific Model ===")
print(metrics.to_string(index=False))

# Per-resolution breakdown
print("\nPer-resolution breakdown:")
for res in [6, 7]:
    res_preds = predictions[predictions["h3_resolution"] == res]
    for fold_name in sorted(res_preds["fold"].unique()):
        fold_data = res_preds[res_preds["fold"] == fold_name]
        mape = fold_data["ape"].mean() * 100
        print(f"  Res {res} {fold_name}: MAPE={mape:.2f}%")
    overall = res_preds["ape"].mean() * 100
    print(f"  Res {res} overall: MAPE={overall:.2f}%")

# Compare with v9 unified
print("\n--- Comparison with v9 unified ---")
try:
    v9_metrics = pd.read_csv(METRICS_DIR / "model_v9_tuned_fold_metrics.csv")
    for _, row in metrics.iterrows():
        fold = row["fold"]
        v9_row = v9_metrics[v9_metrics["fold"] == fold]
        if not v9_row.empty:
            v9_mape = v9_row.iloc[0]["mape"]
            delta = row["mape"] - v9_mape
            arrow = "↓" if delta < 0 else "↑"
            print(f"  {fold}: res_split={row['mape']:.4f}  unified={v9_mape:.4f}  delta={delta:+.4f} {arrow}")
except FileNotFoundError:
    print("  (v9 tuned metrics not found)")
