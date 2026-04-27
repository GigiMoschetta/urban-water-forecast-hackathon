"""General ablation runner for ensemble compression experiments.

Runs the same core flow used by the current champion family:
- merge chosen base prediction streams
- optimize Nelder-Mead weights globally and by resolution
- keep the better pre-correction ensemble
- apply per-series naive fallback
- optionally run nested residual correction

Example:
PYTHONPATH=src ./.venv/bin/python scripts/run_ensemble_ablation.py \
  --models naive,v1,v3,v3d,v8,v9,v9w \
  --output-name ensemble_ablation_full7 \
  --trials 120
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import optuna  # type: ignore
except Exception:  # pragma: no cover - fallback for lighter local envs
    optuna = None

from hackathon_opti.baselines import SERIES_KEYS, summarize_fold_metrics
from hackathon_opti.config import BASELINE_PREDICTIONS, METRICS_DIR, PREDICTIONS_DIR
from hackathon_opti.ensemble_stacking import (
    MERGE_KEYS,
    _compute_error_cols,
    apply_per_series_selection,
    build_nelder_mead_predictions,
    optimize_horizon_weights_nelder_mead,
)

MODEL_PATHS = {
    "naive": BASELINE_PREDICTIONS,
    "v1": PREDICTIONS_DIR / "model_v1_tuned_predictions.csv",
    "v3": PREDICTIONS_DIR / "model_v3_tuned_predictions.csv",
    "v3d": PREDICTIONS_DIR / "model_v3d_tuned_predictions.csv",
    "v8": PREDICTIONS_DIR / "model_v8_tuned_predictions.csv",
    "v9": PREDICTIONS_DIR / "model_v9_tuned_predictions.csv",
    "v9w": PREDICTIONS_DIR / "model_v9_weighted_predictions.csv",
    "v9o": PREDICTIONS_DIR / "model_v9_optuna_predictions.csv",
    "ridge": PREDICTIONS_DIR / "model_ridge_predictions.csv",
}

BASE_METADATA_COLS = MERGE_KEYS + ["actual", "system_area", "h3_resolution", "horizon"]


def parse_models(raw: str) -> list[str]:
    models = [m.strip() for m in raw.split(",") if m.strip()]
    unknown = [m for m in models if m not in MODEL_PATHS]
    if unknown:
        raise ValueError(f"Unknown models: {unknown}. Allowed: {sorted(MODEL_PATHS)}")
    return models


def load_predictions(models: Iterable[str]) -> dict[str, pd.DataFrame]:
    preds = {}
    for model in models:
        path = MODEL_PATHS[model]
        if not path.exists():
            raise FileNotFoundError(f"Missing predictions for {model}: {path}")
        preds[model] = pd.read_csv(path)
    return preds


def pick_base_df(preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    preferred = ["v3", "v3d", "v8", "v9", "v9w", "v9o", "v1", "naive"]
    for name in preferred:
        df = preds.get(name)
        if df is not None and set(BASE_METADATA_COLS).issubset(df.columns):
            return df[BASE_METADATA_COLS].copy()
    for name, df in preds.items():
        if set(BASE_METADATA_COLS).issubset(df.columns):
            return df[BASE_METADATA_COLS].copy()
    raise ValueError("Could not find a base prediction file with metadata columns")


def merge_predictions(preds: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[str]]:
    base = pick_base_df(preds)
    for name, df in preds.items():
        base = base.merge(
            df[MERGE_KEYS + ["prediction"]].rename(columns={"prediction": f"pred_{name}"}),
            on=MERGE_KEYS,
            how="left",
        )
    base["system_area_id"] = base["system_area"].map({"PD": 0, "TS": 1})
    pred_cols = [f"pred_{name}" for name in preds]
    merged = base.dropna(subset=pred_cols + ["actual"]).copy()
    return merged, pred_cols


def nested_apply_correction(base_df: pd.DataFrame, n_trials: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    optimizer_name = "optuna" if optuna is not None else "random-search"
    print(f"Nested correction backend: {optimizer_name}")
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

        corrected_parts = []
        for fold in sorted(df["fold"].unique()):
            f_df = df[df["fold"] == fold].copy()
            group_cols = ["system_area", "rate_category_id", "horizon"]
            bias = f_df.groupby(group_cols)["error"].transform("mean")
            correction = bias * params["base_strength"] * f_df["cat_strength"] * f_df["area_strength"]
            limit = params["clip"]
            correction = correction.clip(-limit * f_df["prediction"], limit * f_df["prediction"])
            f_df["prediction"] = (f_df["prediction"] - correction).clip(lower=0)
            corrected_parts.append(f_df)

        corrected = pd.concat(corrected_parts, ignore_index=True)
        return recompute_error_columns(corrected)

    def sample_params(rng: np.random.Generator) -> dict:
        return {
            "base_strength": rng.uniform(0.1, 0.9),
            "cat3": rng.uniform(0.2, 1.5),
            "cat4": rng.uniform(0.2, 1.5),
            "cat5": rng.uniform(0.2, 1.5),
            "area_pd": rng.uniform(0.8, 1.2),
            "area_ts": rng.uniform(0.8, 1.2),
            "clip": rng.uniform(0.05, 0.4),
        }

    def score_params(df_train_folds: pd.DataFrame, params: dict) -> float:
        corrected = apply_correction(df_train_folds, params)
        return corrected.groupby("fold")["ape"].mean().mean() * 100

    folds = sorted(base_df["fold"].unique())
    all_corrected = []
    print(f"\nNested correction over folds: {folds}")
    for hold_out in folds:
        train_folds = base_df[base_df["fold"] != hold_out].copy()
        test_fold = base_df[base_df["fold"] == hold_out].copy()

        best_params = None
        best_score = float("inf")

        if optuna is not None:
            def objective(trial: "optuna.Trial") -> float:
                params = {
                    "base_strength": trial.suggest_float("base_strength", 0.1, 0.9),
                    "cat3": trial.suggest_float("cat3", 0.2, 1.5),
                    "cat4": trial.suggest_float("cat4", 0.2, 1.5),
                    "cat5": trial.suggest_float("cat5", 0.2, 1.5),
                    "area_pd": trial.suggest_float("area_pd", 0.8, 1.2),
                    "area_ts": trial.suggest_float("area_ts", 0.8, 1.2),
                    "clip": trial.suggest_float("clip", 0.05, 0.4),
                }
                return score_params(train_folds, params)

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=n_trials)
            best_params = study.best_params
            best_score = float(study.best_value)
        else:
            rng = np.random.default_rng(20260329 + int(str(hold_out).split('_')[-1]))
            seeds = [
                {
                    "base_strength": 0.82,
                    "cat3": 1.25,
                    "cat4": 0.46,
                    "cat5": 0.89,
                    "area_pd": 0.85,
                    "area_ts": 0.93,
                    "clip": 0.079,
                },
                {
                    "base_strength": 0.6,
                    "cat3": 1.0,
                    "cat4": 1.0,
                    "cat5": 1.0,
                    "area_pd": 1.0,
                    "area_ts": 1.0,
                    "clip": 0.1,
                },
            ]
            for params in seeds:
                score = score_params(train_folds, params)
                if score < best_score:
                    best_score = score
                    best_params = params
            for _ in range(n_trials):
                params = sample_params(rng)
                score = score_params(train_folds, params)
                if score < best_score:
                    best_score = score
                    best_params = params

        corrected_test = apply_correction(test_fold, best_params)
        raw_mape = test_fold["ape"].mean() * 100
        corrected_mape = corrected_test["ape"].mean() * 100
        delta = corrected_mape - raw_mape
        arrow = "↓" if delta < 0 else "↑"
        print(
            f"  {hold_out}: {raw_mape:.4f} -> {corrected_mape:.4f} "
            f"({delta:+.4f} {arrow}) | train-best={best_score:.4f}"
        )
        all_corrected.append(corrected_test)

    corrected = pd.concat(all_corrected, ignore_index=True)
    corrected = corrected.sort_values(["fold", "period_ym", "cell_id", "rate_category_id"]).reset_index(drop=True)
    metrics = summarize_fold_metrics(corrected)
    return corrected, metrics


def run_ablation(models: list[str], output_name: str, trials: int, skip_nested: bool) -> None:
    print(f"Loading models: {models}")
    preds = load_predictions(models)
    for name, df in preds.items():
        print(f"  {name}: {len(df)} rows")

    merged, pred_cols = merge_predictions(preds)
    print(f"Merged rows: {len(merged)}")
    print(f"Prediction columns: {pred_cols}")

    print("\n=== Global Nelder-Mead ===")
    hw_global = optimize_horizon_weights_nelder_mead(merged, pred_cols)
    preds_global = build_nelder_mead_predictions(merged, pred_cols, hw_global)
    preds_global = _compute_error_cols(preds_global)
    preds_global = apply_per_series_selection(preds_global)
    metrics_global = summarize_fold_metrics(preds_global)
    mean_global = metrics_global[metrics_global["fold"] == "mean"].iloc[0]["mape"]
    print(f"Global post-selection MAPE: {mean_global:.4f}")

    print("\n=== Resolution-specific Nelder-Mead ===")
    preds_parts = []
    for res in [6, 7]:
        part = merged[merged["h3_resolution"] == res].copy()
        print(f"  Resolution {res}: {len(part)} rows")
        hw = optimize_horizon_weights_nelder_mead(part, pred_cols)
        pred_part = build_nelder_mead_predictions(part, pred_cols, hw)
        pred_part = _compute_error_cols(pred_part)
        preds_parts.append(pred_part)
    preds_res = pd.concat(preds_parts, ignore_index=True)
    preds_res = apply_per_series_selection(preds_res)
    metrics_res = summarize_fold_metrics(preds_res)
    mean_res = metrics_res[metrics_res["fold"] == "mean"].iloc[0]["mape"]
    print(f"Resolution-specific post-selection MAPE: {mean_res:.4f}")

    if mean_global <= mean_res:
        best_preds = preds_global
        best_metrics = metrics_global
        strategy = "global"
    else:
        best_preds = preds_res
        best_metrics = metrics_res
        strategy = "resolution"

    print(f"\nSelected pre-correction strategy: {strategy}")
    print(best_metrics.to_string(index=False))

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    pre_name = f"{output_name}_pre"
    best_preds.to_csv(PREDICTIONS_DIR / f"{pre_name}_predictions.csv", index=False)
    best_metrics.to_csv(METRICS_DIR / f"{pre_name}_fold_metrics.csv", index=False)

    if skip_nested:
        final_preds = best_preds
        final_metrics = best_metrics
    else:
        final_preds, final_metrics = nested_apply_correction(best_preds, n_trials=trials)
        final_preds.to_csv(PREDICTIONS_DIR / f"{output_name}_predictions.csv", index=False)
        final_metrics.to_csv(METRICS_DIR / f"{output_name}_fold_metrics.csv", index=False)

    print(f"\n=== Final metrics: {output_name} ===")
    print(final_metrics.to_string(index=False))
    mean = final_metrics[final_metrics["fold"] == "mean"].iloc[0]
    print(f"FINAL_MEAN_MAPE={mean['mape']:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True, help="Comma-separated model aliases")
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--trials", type=int, default=80)
    parser.add_argument("--skip-nested", action="store_true")
    args = parser.parse_args()

    run_ablation(
        models=parse_models(args.models),
        output_name=args.output_name,
        trials=args.trials,
        skip_nested=args.skip_nested,
    )
