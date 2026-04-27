# Model Architecture

_Last updated: 2026-03-30_

## Primary model: `ensemble_3m_v1_v9o_nested`

**MAPE: 19.91%** — 3-model ensemble with nested CV residual correction.

Each component has a distinct, non-redundant role validated by ablation (2026-03-30):

| Stream | Type | Individual MAPE | Role |
|--------|------|----------------:|------|
| `naive` | Seasonal naive | 27.91 | Per-series fallback for 38 volatile series |
| `v1_tuned` | Recursive HistGBR | 23.57 | Dominates h=1-4 (weight ~0.64) |
| `v9_optuna` | Domain-SOTA LightGBM | 22.16 | Dominates h=5-12 (weight ~0.75) |

Historical reference: `ensemble_compact5_v3d_v9o_nested` (19.87 MAPE, 5-model — v3, v3d, v8 were redundant with v9o).

## Why this architecture

- **72 data points per series** — too short for local models. Global trees with meta-features outperform.
- **Anonymization preserves same-month cross-cell ratios** — spatial features at time t exploit γ_t cancellation.
- **Recursive vs direct** — v1 (recursive) has lowest bias at h=1-4 (no error propagation yet); v9o (direct) scales better at h=5-12.
- **Per-series naive selection** — 38 high-CV series where no ML pattern is learnable.

## Ensemble pipeline

1. **Nelder-Mead** weight optimization per horizon band (h=1-4, h=5-8, h=9-12), global and per-H3-resolution, best wins
2. **Per-series naive fallback** — compare ensemble vs naive across all folds; switch if naive wins
3. **Nested CV residual correction** — 7 params (base_strength, cat3, cat4, cat5, area_pd, area_ts, clip), leave-one-fold-out, Optuna

## Fold-level results

| Fold | Train | Test | Naive | 3-model ensemble |
|------|-------|------|------:|-----------------:|
| 1 | 2020-01 → 2021-12 | 2022 | 24.87 | 17.25 |
| 2 | 2020-01 → 2022-12 | 2023 | 28.40 | 19.44 |
| 3 | 2020-01 → 2023-12 | 2024 | 24.44 | 16.42 |
| 4 | 2020-01 → 2024-12 | 2025 | 33.93 | 26.52 |
| **Mean** | | | **27.91** | **19.91** |

## Feature engineering (v9o)

All features computed at forecast origin t — leakage-safe.

| Group | Features | Rationale |
|-------|----------|-----------|
| Temporal | lags 1,2,3,6,12,24 · rolling mean/std 3,6,12 · EWM 3,6 · YoY delta/ratio | Core time-series signal |
| Calendar | month sin/cos (k=1,2) · trend_index | Seasonality harmonics |
| Static | rate_category_id · system_area_id · h3_resolution | Series identity |
| Spatial | neighbor_mean · spatial_ratio · node_degree · neighbor_std · area_mean | γ_t cancellation in ratios |
| Meta (per-fold) | series_mean · series_cv · trend_strength · ACF12 | Captures cluster behavior without explicit routing |
| Domain (V9) | gamma_t proxy · COVID/drought/smart-meter flags · March anchor · PD academic · TS tourism · volume density | Structural breaks and territory-specific effects |
| Target-level | target_month sin/cos · target_is_march · target_is_drought · target_quarter_end | Calendar info about the target period (not future values) |

## Artifact paths

Primary runnable:
- `outputs/predictions/ensemble_3m_v1_v9o_nested_predictions.csv`
- `outputs/metrics/ensemble_3m_v1_v9o_nested_fold_metrics.csv`

Historical submission reference (artifact-backed):
- `outputs/predictions/ensemble_v3_enhanced_nested_predictions.csv`
- `outputs/metrics/ensemble_v3_enhanced_nested_fold_metrics.csv`

## Source files

| File | Purpose |
|------|---------|
| `src/hackathon_opti/features.py` | All feature engineering |
| `src/hackathon_opti/model_v1.py` | Recursive HistGBR |
| `src/hackathon_opti/model_v9_enhanced.py` | Domain-SOTA LightGBM |
| `src/hackathon_opti/ensemble_stacking.py` | Nelder-Mead + per-series selection |
| `scripts/run_ensemble_compact5.py` | Primary 3-model ensemble rebuild |
| `scripts/run_ensemble_ablation.py` | Generic subset/ablation runner |
| `scripts/run_reproduction_pipeline.py` | Verify + rebuild entry point |
