"""Build the primary 3-model ensemble.

Model subset:
- naive  (per-series fallback for volatile series)
- v1     (HistGBR recursive — dominates short horizons h=1-4)
- v9o    (Domain-SOTA LightGBM — dominates long horizons h=5-12)

Each model serves a distinct, non-redundant role.
Ablation study (2026-03-30) showed v3, v3d, v8 are redundant with v9o.

Examples:
    PYTHONPATH=src python scripts/run_ensemble_compact5.py
    PYTHONPATH=src python scripts/run_ensemble_compact5.py --trials 150
"""

from __future__ import annotations

import argparse

from run_ensemble_ablation import run_ablation


DEFAULT_MODELS = ["naive", "v1", "v9o"]
DEFAULT_OUTPUT = "ensemble_3m_v1_v9o_nested"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT,
        help="Base output name for predictions/metrics files.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Nested correction optimization trials per hold-out fold.",
    )
    parser.add_argument(
        "--skip-nested",
        action="store_true",
        help="Stop after blend + naive fallback, without nested residual correction.",
    )
    args = parser.parse_args()

    run_ablation(
        models=DEFAULT_MODELS,
        output_name=args.output_name,
        trials=args.trials,
        skip_nested=args.skip_nested,
    )


if __name__ == "__main__":
    main()
