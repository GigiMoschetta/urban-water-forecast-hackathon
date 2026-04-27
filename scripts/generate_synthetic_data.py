#!/usr/bin/env python3
"""
Generate synthetic demo data for the public release of this repository.

This script produces deterministic CSVs with the same column shape as the
original hackathon artifacts but values that have NO connection to the real
AcegasApsAmga dataset. Volumes are near-constant (~200 m3) with small seeded
gaussian noise. The point is just to make the demo dashboard and the build
tooling runnable end-to-end without sharing any confidential data.

Run from the repo root:
    python scripts/generate_synthetic_data.py

Then build the dashboard payload:
    python demo/scripts/build_data.py

Outputs (all overwritten on each run):
    data/processed/timeseries_canonical.csv
    outputs/predictions/seasonal_naive_predictions.csv
    outputs/predictions/model_v1_tuned_predictions.csv
    outputs/predictions/model_v9_optuna_predictions.csv
    outputs/predictions/ensemble_v3_enhanced_nested_predictions.csv
    outputs/predictions/ensemble_3m_v1_v9o_nested_predictions.csv
    outputs/metrics/<various>_fold_metrics.csv
"""
import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEED = 42
BASE_VOLUME = 200.0
SIGMA_HIST = 2.0
SIGMA_BY_MODEL = {
    "naive": 6.0,
    "v1": 4.0,
    "v9": 3.5,
    "v3_ensemble": 3.0,
    "primary": 2.5,
}

CELLS = (
    [{"cell_id": f"cell_PD_r7_{i:03d}", "system_area": "PD", "h3_resolution": 7} for i in range(1, 11)]
    + [{"cell_id": f"cell_TS_r7_{i:03d}", "system_area": "TS", "h3_resolution": 7} for i in range(1, 8)]
    + [{"cell_id": f"cell_TS_r6_{i:03d}", "system_area": "TS", "h3_resolution": 6} for i in range(1, 4)]
)
CATEGORIES = [1, 2, 3, 4, 5]
PERIODS = [y * 100 + m for y in range(2020, 2026) for m in range(1, 13)]
FOLDS = {"fold_1": 2022, "fold_2": 2023, "fold_3": 2024, "fold_4": 2025}


def system_area_id(area):
    return 1 if area == "TS" else 2


def make_timeseries(rng):
    rows = []
    for cell in CELLS:
        for cat in CATEGORIES:
            for ym in PERIODS:
                rows.append(
                    {
                        "cell_id": cell["cell_id"],
                        "rate_category_id": cat,
                        "period_ym": ym,
                        "noisy_volume_m3": float(BASE_VOLUME + rng.normal(0, SIGMA_HIST)),
                        "h3_resolution": cell["h3_resolution"],
                        "system_area": cell["system_area"],
                    }
                )
    return pd.DataFrame(rows)


def make_naive_predictions(hist):
    hist_idx = hist.set_index(["cell_id", "rate_category_id", "period_ym"])["noisy_volume_m3"]
    rows = []
    for fold_name, fold_year in FOLDS.items():
        history_year = fold_year - 1
        for cell in CELLS:
            for cat in CATEGORIES:
                for month in range(1, 13):
                    period_ym = fold_year * 100 + month
                    prior_ym = history_year * 100 + month
                    actual_v = float(hist_idx[(cell["cell_id"], cat, period_ym)])
                    prior_v = float(hist_idx[(cell["cell_id"], cat, prior_ym)])
                    error = prior_v - actual_v
                    rows.append(
                        {
                            "cell_id": cell["cell_id"],
                            "rate_category_id": cat,
                            "period_ym": period_ym,
                            "actual": actual_v,
                            "h3_resolution": cell["h3_resolution"],
                            "system_area": cell["system_area"],
                            "year": fold_year,
                            "month": month,
                            "history_year": history_year,
                            "prediction": prior_v,
                            "fold": fold_name,
                            "error": error,
                            "abs_error": abs(error),
                            "squared_error": error ** 2,
                            "ape": abs(error) / actual_v if actual_v else 0.0,
                        }
                    )
    return pd.DataFrame(rows)


def make_model_predictions(hist, sigma, rng):
    hist_idx = hist.set_index(["cell_id", "rate_category_id", "period_ym"])["noisy_volume_m3"]
    rows = []
    for fold_name, fold_year in FOLDS.items():
        forecast_origin_ym = (fold_year - 1) * 100 + 12
        for cell in CELLS:
            for cat in CATEGORIES:
                for month in range(1, 13):
                    period_ym = fold_year * 100 + month
                    actual_v = float(hist_idx[(cell["cell_id"], cat, period_ym)])
                    prediction = actual_v + rng.normal(0, sigma)
                    error = prediction - actual_v
                    rows.append(
                        {
                            "fold": fold_name,
                            "forecast_origin_ym": forecast_origin_ym,
                            "horizon": month,
                            "cell_id": cell["cell_id"],
                            "rate_category_id": cat,
                            "period_ym": period_ym,
                            "system_area": cell["system_area"],
                            "h3_resolution": cell["h3_resolution"],
                            "actual": actual_v,
                            "prediction": prediction,
                            "error": error,
                            "abs_error": abs(error),
                            "squared_error": error ** 2,
                            "ape": abs(error) / actual_v if actual_v else 0.0,
                        }
                    )
    return pd.DataFrame(rows)


def make_ensemble_predictions(hist, sigma, rng, naive_df, v1_df, v9_df):
    hist_idx = hist.set_index(["cell_id", "rate_category_id", "period_ym"])["noisy_volume_m3"]
    naive_idx = naive_df.set_index(["fold", "cell_id", "rate_category_id", "period_ym"])["prediction"]
    v1_idx = v1_df.set_index(["fold", "cell_id", "rate_category_id", "period_ym"])["prediction"]
    v9_idx = v9_df.set_index(["fold", "cell_id", "rate_category_id", "period_ym"])["prediction"]
    rows = []
    for fold_name, fold_year in FOLDS.items():
        for cell in CELLS:
            for cat in CATEGORIES:
                for month in range(1, 13):
                    period_ym = fold_year * 100 + month
                    actual_v = float(hist_idx[(cell["cell_id"], cat, period_ym)])
                    pred_n = float(naive_idx[(fold_name, cell["cell_id"], cat, period_ym)])
                    pred_v1 = float(v1_idx[(fold_name, cell["cell_id"], cat, period_ym)])
                    pred_v9 = float(v9_idx[(fold_name, cell["cell_id"], cat, period_ym)])
                    prediction = actual_v + rng.normal(0, sigma)
                    error = prediction - actual_v
                    rows.append(
                        {
                            "fold": fold_name,
                            "cell_id": cell["cell_id"],
                            "rate_category_id": cat,
                            "period_ym": period_ym,
                            "actual": actual_v,
                            "system_area": cell["system_area"],
                            "h3_resolution": cell["h3_resolution"],
                            "horizon": month,
                            "pred_naive": pred_n,
                            "pred_v1": pred_v1,
                            "pred_v9o": pred_v9,
                            "system_area_id": system_area_id(cell["system_area"]),
                            "prediction": prediction,
                            "error": error,
                            "abs_error": abs(error),
                            "squared_error": error ** 2,
                            "ape": abs(error) / actual_v if actual_v else 0.0,
                            "cat_strength": 1.0,
                            "area_strength": 1.0,
                        }
                    )
    return pd.DataFrame(rows)


def make_metrics_csv(predictions_df):
    rows = []
    for fold_name in sorted(predictions_df["fold"].unique()):
        f = predictions_df[predictions_df["fold"] == fold_name]
        valid = f[f["actual"] != 0]
        rows.append(
            {
                "fold": fold_name,
                "n_predictions": len(f),
                "n_mape_samples": len(valid),
                "mape": float(valid["ape"].mean() * 100),
                "mae": float(f["abs_error"].mean()),
                "rmse": float(np.sqrt(f["squared_error"].mean())),
            }
        )
    return pd.DataFrame(rows)


def main():
    out_proc = ROOT / "data" / "processed"
    out_pred = ROOT / "outputs" / "predictions"
    out_met = ROOT / "outputs" / "metrics"
    out_proc.mkdir(parents=True, exist_ok=True)
    out_pred.mkdir(parents=True, exist_ok=True)
    out_met.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)

    print("Building synthetic timeseries_canonical.csv ...")
    hist = make_timeseries(rng)
    hist.to_csv(out_proc / "timeseries_canonical.csv", index=False)
    print(f"  {len(hist)} rows, {hist['cell_id'].nunique()} cells")

    print("Building seasonal_naive_predictions.csv ...")
    naive = make_naive_predictions(hist)
    naive.to_csv(out_pred / "seasonal_naive_predictions.csv", index=False)
    print(f"  {len(naive)} rows")

    print("Building model_v1_tuned_predictions.csv ...")
    v1 = make_model_predictions(hist, SIGMA_BY_MODEL["v1"], rng)
    v1.to_csv(out_pred / "model_v1_tuned_predictions.csv", index=False)

    print("Building model_v9_optuna_predictions.csv ...")
    v9 = make_model_predictions(hist, SIGMA_BY_MODEL["v9"], rng)
    v9.to_csv(out_pred / "model_v9_optuna_predictions.csv", index=False)

    print("Building ensemble_v3_enhanced_nested_predictions.csv ...")
    ens_v3 = make_ensemble_predictions(hist, SIGMA_BY_MODEL["v3_ensemble"], rng, naive, v1, v9)
    ens_v3.to_csv(out_pred / "ensemble_v3_enhanced_nested_predictions.csv", index=False)

    print("Building ensemble_3m_v1_v9o_nested_predictions.csv ...")
    ens_primary = make_ensemble_predictions(hist, SIGMA_BY_MODEL["primary"], rng, naive, v1, v9)
    ens_primary.to_csv(out_pred / "ensemble_3m_v1_v9o_nested_predictions.csv", index=False)

    print("Building fold metrics CSVs ...")
    metric_specs = [
        ("seasonal_naive_fold_metrics.csv", naive),
        ("model_v1_tuned_fold_metrics.csv", v1),
        ("model_v9_optuna_fold_metrics.csv", v9),
        ("ensemble_v3_enhanced_nested_fold_metrics.csv", ens_v3),
        ("ensemble_3m_v1_v9o_nested_fold_metrics.csv", ens_primary),
        ("ensemble_3m_v1_v9o_nested_pre_fold_metrics.csv", ens_primary),
        ("ensemble_compact5_v3d_v9o_nested_fold_metrics.csv", ens_primary),
        ("model_v3_tuned_fold_metrics.csv", ens_v3),
        ("model_v3d_tuned_fold_metrics.csv", ens_v3),
        ("model_v8_tuned_fold_metrics.csv", v1),
    ]
    for filename, df in metric_specs:
        make_metrics_csv(df).to_csv(out_met / filename, index=False)

    print()
    print("Done. Synthetic placeholder data — not derived from any real dataset.")
    print(f"  {len(CELLS)} cells x {len(CATEGORIES)} categories x {len(PERIODS)} months")
    print(f"  Base volume {BASE_VOLUME} m3 + N(0, {SIGMA_HIST}) noise")


if __name__ == "__main__":
    main()
