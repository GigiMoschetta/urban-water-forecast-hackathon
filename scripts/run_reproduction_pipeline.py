"""Entry point for verification and the documented runnable rebuild paths.

Default behavior verifies the historical submission-reference artifact:
    ensemble_v3_enhanced_nested

Primary runnable rebuild path:
- ensemble_3m_v1_v9o_nested  (recommended 3-model ensemble: naive+v1+v9o)

Historical comparison rebuild path:
- ensemble_v3_nested_corrected       (historical 7-stream near-champion path)

Examples:
    python3 scripts/run_reproduction_pipeline.py
    python3 scripts/run_reproduction_pipeline.py --rebuild-compact5 --trials 150
    python3 scripts/run_reproduction_pipeline.py --rebuild-open-pipeline --trials 150
"""

import argparse
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
OFFICIAL_PREDICTIONS = ROOT / "outputs" / "predictions" / "ensemble_v3_enhanced_nested_predictions.csv"
OFFICIAL_METRICS = ROOT / "outputs" / "metrics" / "ensemble_v3_enhanced_nested_fold_metrics.csv"
OPEN_PREDICTIONS = ROOT / "outputs" / "predictions" / "ensemble_v3_nested_corrected_predictions.csv"
OPEN_METRICS = ROOT / "outputs" / "metrics" / "ensemble_v3_nested_corrected_fold_metrics.csv"
COMPACT_PREDICTIONS = ROOT / "outputs" / "predictions" / "ensemble_3m_v1_v9o_nested_predictions.csv"
COMPACT_METRICS = ROOT / "outputs" / "metrics" / "ensemble_3m_v1_v9o_nested_fold_metrics.csv"

EXPECTED_OFFICIAL = {"mape": 19.95, "mae": 705.0, "rmse": 2244.0}


def run_step(script: str, description: str) -> None:
    print(f"\n>>> STEP: {description}")
    command = [PYTHON, *shlex.split(script)]
    print("Executing:", " ".join(command))
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    subprocess.run(command, cwd=ROOT, check=True, env=env)


def recompute_mean_metrics(predictions_path: Path) -> dict[str, float]:
    df = pd.read_csv(predictions_path)
    per_fold = []
    for _, chunk in df.groupby("fold"):
        nonzero = chunk["actual"] != 0
        per_fold.append(
            {
                "mape": float((chunk.loc[nonzero, "abs_error"] / chunk.loc[nonzero, "actual"].abs()).mean() * 100),
                "mae": float(chunk["abs_error"].mean()),
                "rmse": float(math.sqrt(chunk["squared_error"].mean())),
            }
        )
    mape = float(sum(row["mape"] for row in per_fold) / len(per_fold))
    mae = float(sum(row["mae"] for row in per_fold) / len(per_fold))
    rmse = float(sum(row["rmse"] for row in per_fold) / len(per_fold))
    return {"mape": mape, "mae": mae, "rmse": rmse}


def read_reported_mean(metrics_path: Path) -> dict[str, float]:
    df = pd.read_csv(metrics_path)
    mean_row = df[df["fold"] == "mean"]
    if mean_row.empty:
        raise ValueError(f"No mean row found in {metrics_path}")
    row = mean_row.iloc[0]
    return {"mape": float(row["mape"]), "mae": float(row["mae"]), "rmse": float(row["rmse"])}


def rounded_triplet(metrics: dict[str, float]) -> tuple[float, float, float]:
    return (round(metrics["mape"], 2), round(metrics["mae"]), round(metrics["rmse"]))


def verify_official_artifact() -> None:
    print("==============================================================")
    print("   OFFICIAL CHAMPION VERIFICATION                             ")
    print("==============================================================")
    print(f"Champion: {OFFICIAL_PREDICTIONS.stem.replace('_predictions', '')}")

    missing = [path for path in [OFFICIAL_PREDICTIONS, OFFICIAL_METRICS] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing official artifact(s): {', '.join(str(p) for p in missing)}")

    recomputed = recompute_mean_metrics(OFFICIAL_PREDICTIONS)
    reported = read_reported_mean(OFFICIAL_METRICS)

    print("\nArtifact-backed official files:")
    print(f"  Predictions: {OFFICIAL_PREDICTIONS.relative_to(ROOT)}")
    print(f"  Metrics:     {OFFICIAL_METRICS.relative_to(ROOT)}")

    print("\nRecomputed mean metrics from predictions:")
    print(f"  MAPE: {recomputed['mape']:.6f}")
    print(f"  MAE:  {recomputed['mae']:.6f}")
    print(f"  RMSE: {recomputed['rmse']:.6f}")

    print("\nReported mean metrics from metrics CSV:")
    print(f"  MAPE: {reported['mape']:.6f}")
    print(f"  MAE:  {reported['mae']:.6f}")
    print(f"  RMSE: {reported['rmse']:.6f}")

    if rounded_triplet(recomputed) != rounded_triplet(reported):
        raise ValueError("Recomputed metrics do not match the versioned official metrics file.")
    if rounded_triplet(recomputed) != (
        EXPECTED_OFFICIAL["mape"],
        EXPECTED_OFFICIAL["mae"],
        EXPECTED_OFFICIAL["rmse"],
    ):
        raise ValueError("Official artifact does not match the canonical champion metrics.")

    print("\nResult:")
    print("  Official champion is artifact-backed and internally consistent.")
    print("  Canonical submission metrics: 19.95 MAPE / 705 MAE / 2244 RMSE")


def rebuild_open_pipeline(trials: int) -> None:
    print("\n==============================================================")
    print("   SCRIPT-EXPOSED REBUILD PATH                                ")
    print("==============================================================")
    print("This rebuilds the nearest script-exposed pipeline, not the final 19.95 artifact.")
    print("Target output: ensemble_v3_nested_corrected (~19.96 MAPE)")

    run_step("scripts/run_ensemble_v2.py", "Building the seven-stream V3 ensemble")
    run_step(
        f"scripts/run_optuna_residual_correction_nested.py --input outputs/predictions/ensemble_v3_predictions.csv "
        f"--output-name ensemble_v3_nested_corrected --trials {trials}",
        "Applying nested residual correction",
    )

    if OPEN_PREDICTIONS.exists() and OPEN_METRICS.exists():
        recomputed = recompute_mean_metrics(OPEN_PREDICTIONS)
        print("\nScript-exposed rebuild output:")
        print(f"  Predictions: {OPEN_PREDICTIONS.relative_to(ROOT)}")
        print(f"  Metrics:     {OPEN_METRICS.relative_to(ROOT)}")
        print(f"  Recomputed mean MAPE: {recomputed['mape']:.6f}")
        print(f"  Recomputed mean MAE:  {recomputed['mae']:.6f}")
        print(f"  Recomputed mean RMSE: {recomputed['rmse']:.6f}")


def rebuild_compact5_pipeline(trials: int) -> None:
    print("\n==============================================================")
    print("   3-MODEL ENSEMBLE REBUILD PATH                               ")
    print("==============================================================")
    print("This rebuilds the primary 3-model ensemble (naive+v1+v9o).")
    print("Target output: ensemble_3m_v1_v9o_nested (~19.91 MAPE)")

    run_step(
        f"scripts/run_ensemble_compact5.py --output-name ensemble_3m_v1_v9o_nested --trials {trials}",
        "Building the 3-model nested ensemble",
    )

    if COMPACT_PREDICTIONS.exists() and COMPACT_METRICS.exists():
        recomputed = recompute_mean_metrics(COMPACT_PREDICTIONS)
        print("\nCompact rebuild output:")
        print(f"  Predictions: {COMPACT_PREDICTIONS.relative_to(ROOT)}")
        print(f"  Metrics:     {COMPACT_METRICS.relative_to(ROOT)}")
        print(f"  Recomputed mean MAPE: {recomputed['mape']:.6f}")
        print(f"  Recomputed mean MAE:  {recomputed['mae']:.6f}")
        print(f"  Recomputed mean RMSE: {recomputed['rmse']:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rebuild-open-pipeline",
        action="store_true",
        help="Rebuild the nearest honest script-exposed pipeline (ensemble_v3_nested_corrected) from versioned artifacts.",
    )
    parser.add_argument(
        "--rebuild-compact5",
        action="store_true",
        help="Rebuild the primary 3-model ensemble (naive+v1+v9o).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=150,
        help="Optimization trials per fold for the rebuild paths.",
    )
    args = parser.parse_args()

    verify_official_artifact()

    if args.rebuild_compact5:
        rebuild_compact5_pipeline(trials=args.trials)
    if args.rebuild_open_pipeline:
        rebuild_open_pipeline(trials=args.trials)
    if not args.rebuild_open_pipeline and not args.rebuild_compact5:
        print("\nNo retraining was run.")
        print("Primary runnable path: --rebuild-compact5 (~19.91 MAPE, 3-model ensemble).")
        print("Historical comparison path: --rebuild-open-pipeline (~19.96 MAPE).")


if __name__ == "__main__":
    main()
