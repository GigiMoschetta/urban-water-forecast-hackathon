from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

from .baselines import SERIES_KEYS, summarize_fold_metrics
from .config import METRICS_DIR, MODEL_V1_METRICS, MODEL_V1_PREDICTIONS, PREDICTIONS_DIR
from .validation import OFFICIAL_FOLDS, Fold, split_by_fold

TARGET_COL = "noisy_volume_m3"
STATIC_FEATURES = ["rate_category_id", "system_area_id", "h3_resolution"]
LAG_FEATURES = [1, 2, 3, 6, 12]
ROLLING_MEAN_WINDOWS = [3, 6, 12]
ROLLING_STD_WINDOWS = [6, 12]
SYSTEM_AREA_ENCODING = {"PD": 0, "TS": 1}


@dataclass(frozen=True)
class SeriesMetadata:
    cell_id: str
    rate_category_id: int
    system_area: str
    h3_resolution: int


def _month_to_angle(month: int) -> float:
    return 2 * np.pi * month / 12.0


def _next_period_ym(period_ym: int) -> int:
    return int((pd.Period(str(period_ym), freq="M") + 1).strftime("%Y%m"))


def build_training_frame(canonical_ts: pd.DataFrame) -> pd.DataFrame:
    frame = canonical_ts.sort_values(SERIES_KEYS + ["period_ym"]).copy()
    grouped = frame.groupby(SERIES_KEYS, sort=False)[TARGET_COL]
    frame["system_area_id"] = frame["system_area"].map(SYSTEM_AREA_ENCODING).astype(int)

    for lag in LAG_FEATURES:
        frame[f"lag_{lag}"] = grouped.shift(lag)

    for window in ROLLING_MEAN_WINDOWS:
        frame[f"rolling_mean_{window}"] = grouped.transform(
            lambda series, w=window: series.shift(1).rolling(window=w, min_periods=w).mean()
        )
    for window in ROLLING_STD_WINDOWS:
        frame[f"rolling_std_{window}"] = grouped.transform(
            lambda series, w=window: series.shift(1).rolling(window=w, min_periods=w).std()
        )

    frame["month"] = frame["period_ym"] % 100
    frame["month_sin"] = np.sin(frame["month"].map(_month_to_angle))
    frame["month_cos"] = np.cos(frame["month"].map(_month_to_angle))

    required_columns = feature_columns()
    return frame.dropna(subset=required_columns).reset_index(drop=True)


def feature_columns() -> list[str]:
    lag_columns = [f"lag_{lag}" for lag in LAG_FEATURES]
    rolling_mean_columns = [f"rolling_mean_{window}" for window in ROLLING_MEAN_WINDOWS]
    rolling_std_columns = [f"rolling_std_{window}" for window in ROLLING_STD_WINDOWS]
    return lag_columns + rolling_mean_columns + rolling_std_columns + STATIC_FEATURES + [
        "month",
        "month_sin",
        "month_cos",
    ]


def build_model() -> TransformedTargetRegressor:
    categorical_columns = {"rate_category_id", "system_area_id", "h3_resolution", "month"}
    regressor = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=4,
        max_iter=120,
        min_samples_leaf=50,
        l2_regularization=0.1,
        early_stopping=False,
        categorical_features=[column in categorical_columns for column in feature_columns()],
        random_state=42,
    )
    return TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


def _series_metadata(train_series: pd.DataFrame) -> SeriesMetadata:
    first = train_series.iloc[0]
    return SeriesMetadata(
        cell_id=str(first["cell_id"]),
        rate_category_id=int(first["rate_category_id"]),
        system_area=str(first["system_area"]),
        h3_resolution=int(first["h3_resolution"]),
    )


def _build_recursive_feature_row(
    metadata: SeriesMetadata,
    target_period_ym: int,
    history_values: list[float],
) -> dict[str, float | int | str]:
    month = target_period_ym % 100
    row: dict[str, float | int | str] = {
        "rate_category_id": metadata.rate_category_id,
        "system_area_id": SYSTEM_AREA_ENCODING[metadata.system_area],
        "h3_resolution": metadata.h3_resolution,
        "month": month,
        "month_sin": float(np.sin(_month_to_angle(month))),
        "month_cos": float(np.cos(_month_to_angle(month))),
    }
    history = np.asarray(history_values, dtype=float)

    for lag in LAG_FEATURES:
        row[f"lag_{lag}"] = float(history[-lag])
    for window in ROLLING_MEAN_WINDOWS:
        row[f"rolling_mean_{window}"] = float(history[-window:].mean())
    for window in ROLLING_STD_WINDOWS:
        row[f"rolling_std_{window}"] = float(history[-window:].std(ddof=1))
    return row


def recursive_fold_predictions(
    canonical_ts: pd.DataFrame,
    training_frame: pd.DataFrame,
    fold: Fold,
) -> pd.DataFrame:
    train, valid = split_by_fold(canonical_ts.sort_values(SERIES_KEYS + ["period_ym"]), fold)
    model = build_model()

    fold_train = training_frame[
        (training_frame["period_ym"] >= fold.train_start) & (training_frame["period_ym"] <= fold.train_end)
    ].copy()
    model.fit(fold_train[feature_columns()], fold_train[TARGET_COL])

    outputs: list[dict[str, float | int | str]] = []
    valid_by_series = {
        key: chunk.sort_values("period_ym").reset_index(drop=True)
        for key, chunk in valid.groupby(SERIES_KEYS, sort=False)
    }

    for key, train_series in train.groupby(SERIES_KEYS, sort=False):
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
                raise ValueError(
                    f"Unexpected target period for series {key}: expected {next_period}, found {target_period}"
                )

            feature_row = _build_recursive_feature_row(metadata, target_period, history_values)
            prediction = float(model.predict(pd.DataFrame([feature_row]))[0])
            prediction = max(prediction, 0.0)
            history_values.append(prediction)
            next_period = _next_period_ym(target_period)

            outputs.append(
                {
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
                }
            )

    predictions = pd.DataFrame(outputs).sort_values(["fold", "period_ym"] + SERIES_KEYS).reset_index(drop=True)
    predictions["error"] = predictions["prediction"] - predictions["actual"]
    predictions["abs_error"] = predictions["error"].abs()
    predictions["squared_error"] = predictions["error"] ** 2
    nonzero = predictions["actual"] != 0
    predictions["ape"] = np.where(nonzero, predictions["abs_error"] / predictions["actual"].abs(), np.nan)
    return predictions


def run_and_save_model_v1(canonical_ts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    canonical = canonical_ts.sort_values(SERIES_KEYS + ["period_ym"]).reset_index(drop=True)
    training_frame = build_training_frame(canonical)

    fold_predictions = [recursive_fold_predictions(canonical, training_frame, fold) for fold in OFFICIAL_FOLDS]
    predictions = pd.concat(fold_predictions, ignore_index=True)
    metrics = summarize_fold_metrics(predictions)

    predictions.to_csv(MODEL_V1_PREDICTIONS, index=False)
    metrics.to_csv(MODEL_V1_METRICS, index=False)
    return predictions, metrics
