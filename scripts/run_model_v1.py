import pandas as pd

from hackathon_opti.config import CANONICAL_TIMESERIES
from hackathon_opti.model_v1 import run_and_save_model_v1


if __name__ == "__main__":
    canonical = pd.read_csv(CANONICAL_TIMESERIES)
    _, metrics = run_and_save_model_v1(canonical)
    print(metrics.to_string(index=False))
