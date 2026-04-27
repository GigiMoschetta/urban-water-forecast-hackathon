# Scripts

All runner scripts for training, tuning, and evaluation. Run from the repo root:

```bash
PYTHONPATH=src python scripts/<script>.py
```

## Pipeline

| Script | Description |
|--------|-------------|
| `run_reproduction_pipeline.py` | Verify the official champion artifact or rebuild documented runnable paths |
| `run_ensemble_compact5.py` | Primary 3-model ensemble rebuild (naive+v1+v9o) |
| `run_ensemble_ablation.py` | Generic subset/ablation runner for ensemble compression work |

## Baseline & Data

| Script | Description |
|--------|-------------|
| `run_audit.py` | Data quality audit on the raw dataset |
| `run_baseline.py` | Seasonal naive baseline (predict same month prior year) |

## Individual Models

| Script | Description |
|--------|-------------|
| `run_model_v1.py` | HistGradientBoosting recursive 12-step |
| `run_model_v1_tuned.py` | v1 with Optuna-tuned hyperparameters |
| `run_model_v3_tuned.py` | LightGBM direct + spatial features (tuned) |
| `run_model_v3d_tuned.py` | v3 with wavelet denoising (tuned) |
| `run_model_v8_tuned.py` | Enhanced LightGBM: EWM + Fourier + meta-features (tuned) |
| `run_model_v9.py` | Domain-optimized LightGBM |
| `run_model_v9_tuned.py` | v9 with Optuna-tuned hyperparameters |
| `run_model_v9_by_resolution.py` | v9 with separate models per H3 resolution |
| `run_model_v9_weighted.py` | v9 with sample weighting |

## Ensembles

| Script | Description |
|--------|-------------|
| `run_ensemble_tuned.py` | Historical 5-model Nelder-Mead ensemble (NM5) |
| `run_ensemble_v2.py` | Historical 7-model ensemble with resolution-aware weights |
| `run_ensemble_compact5.py` | Primary 3-model ensemble (naive+v1+v9o) |
| `run_ensemble_ablation.py` | Flexible ensemble subset runner used for compression testing |

## Hyperparameter Optimization

| Script | Description |
|--------|-------------|
| `run_optuna_v1.py` | Optuna search for model v1 |
| `run_optuna_v3.py` | Optuna search for model v3 |
| `run_optuna_v3d.py` | Optuna search for model v3d |
| `run_optuna_v8.py` | Optuna search for model v8 |
| `run_optuna_v9.py` | Optuna search for model v9 |

## Post-Processing

| Script | Description |
|--------|-------------|
| `run_optuna_residual_correction.py` | Category/area residual correction (standalone) |
| `run_optuna_residual_correction_nested.py` | Nested CV residual correction (leak-free) |
