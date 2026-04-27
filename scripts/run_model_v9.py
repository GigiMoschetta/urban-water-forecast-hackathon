"""Run model v9 (domain-SOTA enhanced) and compare with v8 tuned."""

import pandas as pd

from hackathon_opti.config import CANONICAL_TIMESERIES, CLEAN_ADJACENCY
from hackathon_opti.model_v9_enhanced import run_and_save_model_v9

print("Loading data...")
canonical_ts = pd.read_csv(CANONICAL_TIMESERIES)
adjacency = pd.read_csv(CLEAN_ADJACENCY)

print(f"Canonical TS: {len(canonical_ts)} rows")
print(f"Adjacency: {len(adjacency)} rows")

predictions, metrics = run_and_save_model_v9(canonical_ts, adjacency)

# Compare with v8 tuned
print("\n" + "=" * 60)
print("COMPARISON: v9 vs v8 tuned")
print("=" * 60)
try:
    v8_metrics = pd.read_csv("outputs/metrics/model_v8_tuned_fold_metrics.csv")
    for _, row in metrics.iterrows():
        fold = row["fold"]
        v8_row = v8_metrics[v8_metrics["fold"] == fold]
        if not v8_row.empty:
            v8_mape = v8_row.iloc[0]["mape"]
            delta = row["mape"] - v8_mape
            arrow = "↓" if delta < 0 else "↑"
            print(f"  {fold}: v9={row['mape']:.4f}  v8={v8_mape:.4f}  delta={delta:+.4f} {arrow}")
except FileNotFoundError:
    print("  (v8 tuned metrics not found for comparison)")
