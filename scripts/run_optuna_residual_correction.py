"""Optimize or apply the final residual correction on top of the tuned NM5 ensemble."""

import argparse
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from hackathon_opti.baselines import summarize_fold_metrics

PREDICTIONS_DIR = Path("outputs/predictions")
METRICS_DIR = Path("outputs/metrics")
INPUT_FILE = PREDICTIONS_DIR / "ensemble_nm5_tuned_predictions.csv"
OUTPUT_PREDICTIONS = PREDICTIONS_DIR / "ensemble_nm5_tuned_corrected_optuna_predictions.csv"
OUTPUT_METRICS = METRICS_DIR / "ensemble_nm5_tuned_corrected_optuna_fold_metrics.csv"

BEST_PARAMS = {
    "base_strength": 0.8204013649375204,
    "cat3": 1.2502194323795663,
    "cat4": 0.46386891110437667,
    "cat5": 0.8874477599046624,
    "area_pd": 0.8501882662961174,
    "area_ts": 0.9332138883355439,
    "clip": 0.07940681418624868,
}
BEST_MAPE = 20.270882749163913


def recompute_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["error"] = out["prediction"] - out["actual"]
    out["abs_error"] = out["error"].abs()
    out["squared_error"] = out["error"] ** 2
    nonzero = out["actual"] != 0
    out["ape"] = np.where(nonzero, out["abs_error"] / out["actual"].abs(), np.nan)
    return out


def apply_correction(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = df.copy()

    cat_map = {3: params["cat3"], 4: params["cat4"], 5: params["cat5"]}
    area_map = {"PD": params["area_pd"], "TS": params["area_ts"]}

    df["cat_strength"] = df["rate_category_id"].map(cat_map).fillna(1.0)
    df["area_strength"] = df["system_area"].map(area_map).fillna(1.0)

    corrected_preds = []
    for fold in df["fold"].unique():
        f_df = df[df["fold"] == fold].copy()
        group_cols = ["system_area", "rate_category_id", "horizon"]
        bias = f_df.groupby(group_cols)["error"].transform("mean")
        correction = bias * params["base_strength"] * f_df["cat_strength"] * f_df["area_strength"]
        limit = params["clip"]
        correction = correction.clip(-limit * f_df["prediction"], limit * f_df["prediction"])
        f_df["prediction"] = (f_df["prediction"] - correction).clip(lower=0)
        corrected_preds.append(f_df)

    corrected = pd.concat(corrected_preds, ignore_index=True)
    corrected = recompute_error_columns(corrected)
    return corrected


def objective(trial: optuna.Trial) -> float:
    params = {
        "base_strength": trial.suggest_float("base_strength", 0.1, 0.9),
        "cat3": trial.suggest_float("cat3", 0.2, 1.5),
        "cat4": trial.suggest_float("cat4", 0.2, 1.5),
        "cat5": trial.suggest_float("cat5", 0.2, 1.5),
        "area_pd": trial.suggest_float("area_pd", 0.8, 1.2),
        "area_ts": trial.suggest_float("area_ts", 0.8, 1.2),
        "clip": trial.suggest_float("clip", 0.05, 0.4),
    }

    df = pd.read_csv(INPUT_FILE)
    corrected = apply_correction(df, params)
    return corrected.groupby("fold")["ape"].mean().mean() * 100


def save_outputs(df: pd.DataFrame) -> pd.DataFrame:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = summarize_fold_metrics(df)
    df.to_csv(OUTPUT_PREDICTIONS, index=False)
    metrics.to_csv(OUTPUT_METRICS, index=False)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--best-only", action="store_true", help="Skip Optuna search and apply saved best params.")
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials when tuning.")
    args = parser.parse_args()

    base_df = pd.read_csv(INPUT_FILE)

    if args.best_only:
        best_params = BEST_PARAMS
        best_value = BEST_MAPE
    else:
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=args.trials)
        best_params = study.best_params
        best_value = study.best_value

    print("\nBest params:", best_params)
    print("Best MAPE:", best_value)

    best_df = apply_correction(base_df, best_params)
    metrics = save_outputs(best_df)

    print("\nFinal Metrics:")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
