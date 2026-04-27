"""Generate model_v3d predictions using Optuna-tuned best params."""

import numpy as np
import pandas as pd
import pywt
import lightgbm as lgb

from hackathon_opti.config import CANONICAL_TIMESERIES, CLEAN_ADJACENCY, METRICS_DIR, PREDICTIONS_DIR
from hackathon_opti.baselines import SERIES_KEYS, summarize_fold_metrics
from hackathon_opti.model_v3_direct import (
    build_direct_training_rows,
    build_direct_prediction_rows,
    _feature_cols,
    _categorical_cols,
)
from hackathon_opti.features import build_full_feature_frame, TARGET_COL
from hackathon_opti.validation import OFFICIAL_FOLDS

BEST_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "verbosity": -1,
    "random_state": 42,
    "learning_rate": 0.01260804788175048,
    "num_leaves": 42,
    "max_depth": 4,
    "min_child_samples": 49,
    "reg_lambda": 0.011368196358209922,
    "reg_alpha": 0.5324686798888526,
    "subsample": 0.9317928160765981,
    "colsample_bytree": 0.7694773691387033,
    "n_estimators": 500,
}


def wavelet_denoise_series(values: np.ndarray, wavelet: str = "db4", level: int = None) -> np.ndarray:
    if len(values) < 8 or np.all(values == 0):
        return values.copy()
    coeffs = pywt.wavedec(values, wavelet, level=level)
    detail_coeffs = np.concatenate(coeffs[1:])
    sigma = np.median(np.abs(detail_coeffs)) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(values)))
    denoised_coeffs = [coeffs[0]]
    for c in coeffs[1:]:
        denoised_coeffs.append(pywt.threshold(c, threshold, mode="soft"))
    return pywt.waverec(denoised_coeffs, wavelet)[: len(values)]


def denoise_timeseries(canonical_ts: pd.DataFrame) -> pd.DataFrame:
    df = canonical_ts.sort_values(SERIES_KEYS + ["period_ym"]).copy()
    denoised_vals = df[TARGET_COL].values.copy()

    for _, group in df.groupby(SERIES_KEYS, sort=False):
        idx = group.index
        raw = group[TARGET_COL].values.astype(float)
        denoised_vals[idx] = wavelet_denoise_series(raw)

    df[TARGET_COL] = denoised_vals
    return df


print("Loading data...")
canonical_ts = pd.read_csv(CANONICAL_TIMESERIES)
adjacency = pd.read_csv(CLEAN_ADJACENCY)

print("Applying wavelet denoising...")
canonical_denoised = denoise_timeseries(canonical_ts)

print("Building feature frame from denoised data...")
feature_frame_denoised = build_full_feature_frame(canonical_denoised, adjacency)

original_lookup = canonical_ts.set_index(["cell_id", "rate_category_id", "period_ym"])[TARGET_COL]
feature_frame_denoised[TARGET_COL] = feature_frame_denoised.apply(
    lambda r: original_lookup.get((r["cell_id"], r["rate_category_id"], r["period_ym"]), r[TARGET_COL]),
    axis=1,
)

feat_cols = _feature_cols()
cat_cols = _categorical_cols()
all_predictions = []

for fold in OFFICIAL_FOLDS:
    print(f"Processing {fold.name}...")
    train_df = build_direct_training_rows(feature_frame_denoised, fold)
    pred_df = build_direct_prediction_rows(feature_frame_denoised, canonical_ts, fold)

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
predictions.to_csv(PREDICTIONS_DIR / "model_v3d_tuned_predictions.csv", index=False)
metrics.to_csv(METRICS_DIR / "model_v3d_tuned_fold_metrics.csv", index=False)

print("\nFold metrics (tuned v3d):")
print(metrics.to_string(index=False))
