"""Canonical model names and artifact locations for the current repo story.

Primary recommended model:
    ensemble_3m_v1_v9o_nested  (naive + v1 + v9o, 3-model)

Historical / reference artifacts remain available for verification and comparison.
"""

from __future__ import annotations

from .config import METRICS_DIR, PREDICTIONS_DIR

PRIMARY_MODEL_NAME = "ensemble_3m_v1_v9o_nested"
PRIMARY_MODEL_LABEL = "3-model ensemble (naive+v1+v9o)"
PRIMARY_MODEL_PREDICTIONS = PREDICTIONS_DIR / f"{PRIMARY_MODEL_NAME}_predictions.csv"
PRIMARY_MODEL_METRICS = METRICS_DIR / f"{PRIMARY_MODEL_NAME}_fold_metrics.csv"

OFFICIAL_REFERENCE_MODEL_NAME = "ensemble_v3_enhanced_nested"
OFFICIAL_REFERENCE_MODEL_LABEL = "official artifact-backed submission reference"
OFFICIAL_REFERENCE_PREDICTIONS = PREDICTIONS_DIR / f"{OFFICIAL_REFERENCE_MODEL_NAME}_predictions.csv"
OFFICIAL_REFERENCE_METRICS = METRICS_DIR / f"{OFFICIAL_REFERENCE_MODEL_NAME}_fold_metrics.csv"

HISTORICAL_SCRIPT_REBUILD_NAME = "ensemble_v3_nested_corrected"
HISTORICAL_SCRIPT_REBUILD_LABEL = "historical 7-stream near-champion rebuild"
HISTORICAL_SCRIPT_REBUILD_PREDICTIONS = PREDICTIONS_DIR / f"{HISTORICAL_SCRIPT_REBUILD_NAME}_predictions.csv"
HISTORICAL_SCRIPT_REBUILD_METRICS = METRICS_DIR / f"{HISTORICAL_SCRIPT_REBUILD_NAME}_fold_metrics.csv"
