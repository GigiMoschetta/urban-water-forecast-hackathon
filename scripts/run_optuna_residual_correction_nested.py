"""Nested CV residual correction: optimize correction on 3 folds, apply to 4th.

Fixes the leakage in the original correction where params were fit on all 4 folds.
"""

import argparse
import numpy as np
import optuna
import pandas as pd

from hackathon_opti.baselines import summarize_fold_metrics

PREDICTIONS_DIR = "outputs/predictions"
METRICS_DIR = "outputs/metrics"


def recompute_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["error"] = out["prediction"] - out["actual"]
    out["abs_error"] = out["error"].abs()
    out["squared_error"] = out["error"] ** 2
    nonzero = out["actual"] != 0
    out["ape"] = np.where(nonzero, out["abs_error"] / out["actual"].abs(), np.nan)
    return out


def apply_correction(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Apply parameterized residual correction within each fold."""
    df = df.copy()
    cat_map = {3: params["cat3"], 4: params["cat4"], 5: params["cat5"]}
    area_map = {"PD": params["area_pd"], "TS": params["area_ts"]}

    df["cat_strength"] = df["rate_category_id"].map(cat_map).fillna(1.0)
    df["area_strength"] = df["system_area"].map(area_map).fillna(1.0)

    corrected_parts = []
    for fold in df["fold"].unique():
        f_df = df[df["fold"] == fold].copy()
        group_cols = ["system_area", "rate_category_id", "horizon"]
        bias = f_df.groupby(group_cols)["error"].transform("mean")
        correction = bias * params["base_strength"] * f_df["cat_strength"] * f_df["area_strength"]
        limit = params["clip"]
        correction = correction.clip(-limit * f_df["prediction"], limit * f_df["prediction"])
        f_df["prediction"] = (f_df["prediction"] - correction).clip(lower=0)
        corrected_parts.append(f_df)

    corrected = pd.concat(corrected_parts, ignore_index=True)
    corrected = recompute_error_columns(corrected)
    return corrected


def make_objective(df_train_folds: pd.DataFrame):
    """Create objective function for Optuna that operates on training folds only."""
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
        corrected = apply_correction(df_train_folds, params)
        return corrected.groupby("fold")["ape"].mean().mean() * 100
    return objective


def run_nested_correction(
    input_file: str,
    n_trials: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-fold-out nested correction."""
    base_df = pd.read_csv(input_file)
    folds = sorted(base_df["fold"].unique())
    print(f"Loaded {len(base_df)} rows, {len(folds)} folds: {folds}")

    all_corrected = []

    for hold_out in folds:
        print(f"\n--- Hold-out: {hold_out} ---")
        train_folds = base_df[base_df["fold"] != hold_out].copy()
        test_fold = base_df[base_df["fold"] == hold_out].copy()

        # Optimize on train folds
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        study.optimize(make_objective(train_folds), n_trials=n_trials)

        best_params = study.best_params
        best_train_mape = study.best_value
        print(f"  Best train MAPE: {best_train_mape:.4f}")
        print(f"  Params: {best_params}")

        # Apply to hold-out fold
        corrected_test = apply_correction(test_fold, best_params)
        test_mape = corrected_test["ape"].mean() * 100
        uncorrected_mape = test_fold["ape"].mean() * 100
        delta = test_mape - uncorrected_mape
        arrow = "↓" if delta < 0 else "↑"
        print(f"  Hold-out MAPE: {uncorrected_mape:.4f} → {test_mape:.4f} ({delta:+.4f} {arrow})")

        all_corrected.append(corrected_test)

    corrected = pd.concat(all_corrected, ignore_index=True)
    corrected = corrected.sort_values(["fold", "period_ym", "cell_id", "rate_category_id"]).reset_index(drop=True)
    metrics = summarize_fold_metrics(corrected)
    return corrected, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Input predictions CSV")
    parser.add_argument("--output-name", type=str, default="nested_corrected",
                        help="Base name for output files")
    parser.add_argument("--trials", type=int, default=100,
                        help="Optuna trials per fold")
    args = parser.parse_args()

    input_file = args.input or f"{PREDICTIONS_DIR}/ensemble_nm5_tuned_predictions.csv"
    corrected, metrics = run_nested_correction(input_file, n_trials=args.trials)

    output_preds = f"{PREDICTIONS_DIR}/{args.output_name}_predictions.csv"
    output_metrics = f"{METRICS_DIR}/{args.output_name}_fold_metrics.csv"

    import os
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    corrected.to_csv(output_preds, index=False)
    metrics.to_csv(output_metrics, index=False)

    print(f"\n{'='*60}")
    print("=== Nested CV Residual Correction Results ===")
    print(metrics.to_string(index=False))

    # Compare with original non-nested correction
    print("\n--- Comparison with original correction ---")
    try:
        orig = pd.read_csv(f"{METRICS_DIR}/ensemble_nm5_tuned_corrected_optuna_fold_metrics.csv")
        print("Original (non-nested):")
        print(orig.to_string(index=False))
        print("\nNested:")
        print(metrics.to_string(index=False))
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    main()
