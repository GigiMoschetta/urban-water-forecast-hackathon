import pandas as pd

from hackathon_opti.baselines import run_and_save_seasonal_naive
from hackathon_opti.config import CANONICAL_TIMESERIES


if __name__ == "__main__":
    canonical = pd.read_csv(CANONICAL_TIMESERIES)
    _, metrics = run_and_save_seasonal_naive(canonical)
    print(metrics.to_string(index=False))
