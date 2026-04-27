from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BASELINE_METRICS, BASELINE_PREDICTIONS, METRICS_DIR, PREDICTIONS_DIR
from .validation import OFFICIAL_FOLDS, split_by_fold

SERIES_KEYS = ["cell_id", "rate_category_id"]


def add_year_month_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["year"] = out["period_ym"] // 100
    out["month"] = out["period_ym"] % 100
    return out


def seasonal_naive_predictions(canonical_ts: pd.DataFrame) -> pd.DataFrame:
    ts = add_year_month_columns(canonical_ts)
    history = ts[SERIES_KEYS + ["year", "month", "noisy_volume_m3"]].rename(
        columns={"year": "history_year", "noisy_volume_m3": "prediction"}
    )

    outputs = []
    for fold in OFFICIAL_FOLDS:
        train, valid = split_by_fold(ts, fold)
        allowed_history = train[SERIES_KEYS + ["year", "month", "noisy_volume_m3"]].rename(
            columns={"year": "history_year", "noisy_volume_m3": "prediction"}
        )
        valid = valid.copy()
        valid["history_year"] = valid["year"] - 1
        pred = valid.merge(
            allowed_history,
            on=SERIES_KEYS + ["history_year", "month"],
            how="left",
            validate="many_to_one",
        )
        pred["fold"] = fold.name
        pred = pred.rename(columns={"noisy_volume_m3": "actual"})
        outputs.append(pred)

    predictions = pd.concat(outputs, ignore_index=True)
    predictions["error"] = predictions["prediction"] - predictions["actual"]
    predictions["abs_error"] = predictions["error"].abs()
    predictions["squared_error"] = predictions["error"] ** 2
    nonzero = predictions["actual"] != 0
    predictions["ape"] = np.where(nonzero, predictions["abs_error"] / predictions["actual"].abs(), np.nan)
    return predictions


def summarize_fold_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold, chunk in predictions.groupby("fold"):
        mape = chunk["ape"].mean() * 100
        mae = chunk["abs_error"].mean()
        rmse = np.sqrt(chunk["squared_error"].mean())
        rows.append(
            {
                "fold": fold,
                "n_predictions": int(len(chunk)),
                "n_mape_samples": int(chunk["ape"].notna().sum()),
                "mape": float(mape),
                "mae": float(mae),
                "rmse": float(rmse),
            }
        )
    summary = pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)
    overall = pd.DataFrame(
        [
            {
                "fold": "mean",
                "n_predictions": int(summary["n_predictions"].sum()),
                "n_mape_samples": int(summary["n_mape_samples"].sum()),
                "mape": float(summary["mape"].mean()),
                "mae": float(summary["mae"].mean()),
                "rmse": float(summary["rmse"].mean()),
            }
        ]
    )
    return pd.concat([summary, overall], ignore_index=True)


def run_and_save_seasonal_naive(canonical_ts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    predictions = seasonal_naive_predictions(canonical_ts)
    metrics = summarize_fold_metrics(predictions)

    predictions.to_csv(BASELINE_PREDICTIONS, index=False)
    metrics.to_csv(BASELINE_METRICS, index=False)
    return predictions, metrics
