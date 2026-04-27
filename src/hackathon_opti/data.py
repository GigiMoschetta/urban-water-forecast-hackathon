from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

from .config import (
    AUDIT_SUMMARY,
    CANONICAL_TIMESERIES,
    CLEAN_ADJACENCY,
    DATASET_XLSX,
    PROCESSED_DIR,
    REPORTS_DIR,
    TIMESERIES_KEY,
)


@dataclass
class DatasetBundle:
    readme: pd.DataFrame
    timeseries: pd.DataFrame
    adjacency: pd.DataFrame
    rate_categories: pd.DataFrame


EXPECTED_SHEETS = ["readme", "timeseries", "adjacency", "rate_categories"]
EXPECTED_TIMESERIES_COLUMNS = [
    "cell_id",
    "rate_category_id",
    "period_ym",
    "noisy_volume_m3",
    "h3_resolution",
    "system_area",
]
EXPECTED_ADJ_COLUMNS = ["cell_id", "neighbour_cell_id"]
EXPECTED_RATE_COLUMNS = ["rate_category_id", "name", "description"]


def load_raw_dataset(path: Path = DATASET_XLSX) -> DatasetBundle:
    excel = pd.ExcelFile(path)
    missing = [sheet for sheet in EXPECTED_SHEETS if sheet not in excel.sheet_names]
    if missing:
        raise ValueError(f"Missing expected sheets: {missing}")

    bundle = DatasetBundle(
        readme=pd.read_excel(path, sheet_name="readme"),
        timeseries=pd.read_excel(path, sheet_name="timeseries"),
        adjacency=pd.read_excel(path, sheet_name="adjacency"),
        rate_categories=pd.read_excel(path, sheet_name="rate_categories"),
    )
    _validate_schema(bundle)
    return bundle


def _validate_schema(bundle: DatasetBundle) -> None:
    if list(bundle.timeseries.columns) != EXPECTED_TIMESERIES_COLUMNS:
        raise ValueError(
            f"Unexpected timeseries columns: {list(bundle.timeseries.columns)}"
        )
    if list(bundle.adjacency.columns) != EXPECTED_ADJ_COLUMNS:
        raise ValueError(f"Unexpected adjacency columns: {list(bundle.adjacency.columns)}")
    if list(bundle.rate_categories.columns) != EXPECTED_RATE_COLUMNS:
        raise ValueError(
            f"Unexpected rate_categories columns: {list(bundle.rate_categories.columns)}"
        )


def build_canonical_timeseries(timeseries: pd.DataFrame) -> pd.DataFrame:
    canonical = (
        timeseries.groupby(TIMESERIES_KEY, as_index=False)
        .agg(
            noisy_volume_m3=("noisy_volume_m3", "sum"),
            h3_resolution=("h3_resolution", "first"),
            system_area=("system_area", "first"),
        )
        .sort_values(TIMESERIES_KEY)
        .reset_index(drop=True)
    )
    return canonical


def clean_adjacency(adjacency: pd.DataFrame) -> pd.DataFrame:
    return adjacency.drop_duplicates().sort_values(["cell_id", "neighbour_cell_id"]).reset_index(drop=True)


def audit_dataset(bundle: DatasetBundle) -> dict:
    ts = bundle.timeseries.copy()
    adj = bundle.adjacency.copy()

    duplicate_mask = ts.duplicated(TIMESERIES_KEY, keep=False)
    duplicate_series = (
        ts.groupby(["cell_id", "rate_category_id"], as_index=False)
        .size()
        .rename(columns={"size": "rows"})
    )

    adjacency_pairs = set(map(tuple, adj[["cell_id", "neighbour_cell_id"]].itertuples(index=False, name=None)))
    symmetric_edges = sum((b, a) in adjacency_pairs for a, b in adjacency_pairs)
    degree = adj.groupby("cell_id").size()

    summary = {
        "timeseries": {
            "rows": int(len(ts)),
            "unique_cells": int(ts["cell_id"].nunique()),
            "unique_categories": int(ts["rate_category_id"].nunique()),
            "unique_periods": int(ts["period_ym"].nunique()),
            "period_min": int(ts["period_ym"].min()),
            "period_max": int(ts["period_ym"].max()),
            "rows_by_area": {k: int(v) for k, v in ts["system_area"].value_counts().sort_index().items()},
            "rows_by_resolution": {str(k): int(v) for k, v in ts["h3_resolution"].value_counts().sort_index().items()},
            "rows_by_category": {str(k): int(v) for k, v in ts["rate_category_id"].value_counts().sort_index().items()},
            "series_count": int(ts.groupby(["cell_id", "rate_category_id"]).ngroups),
            "months_per_series": {
                str(k): int(v)
                for k, v in ts.groupby(["cell_id", "rate_category_id"])["period_ym"].nunique().value_counts().sort_index().items()
            },
            "duplicate_rows": int(duplicate_mask.sum()),
            "duplicate_excess_rows": int(len(ts) - ts.drop_duplicates(TIMESERIES_KEY).shape[0]),
            "duplicate_rows_by_category": {
                str(k): int(v)
                for k, v in ts.loc[duplicate_mask, "rate_category_id"].value_counts().sort_index().items()
            },
            "duplicate_rows_by_resolution": {
                str(k): int(v)
                for k, v in ts.loc[duplicate_mask, "h3_resolution"].value_counts().sort_index().items()
            },
            "series_row_count_distribution": {
                str(k): int(v) for k, v in duplicate_series["rows"].value_counts().sort_index().items()
            },
            "null_counts": {k: int(v) for k, v in ts.isna().sum().items()},
        },
        "adjacency": {
            "rows": int(len(adj)),
            "unique_nodes": int(len(set(adj["cell_id"]).union(adj["neighbour_cell_id"]))),
            "is_fully_symmetric": symmetric_edges == len(adjacency_pairs),
            "degree_min": int(degree.min()),
            "degree_max": int(degree.max()),
            "degree_mean": float(round(degree.mean(), 4)),
            "null_counts": {k: int(v) for k, v in adj.isna().sum().items()},
        },
        "rate_categories": {
            "rows": int(len(bundle.rate_categories)),
            "ids": [int(v) for v in bundle.rate_categories["rate_category_id"].tolist()],
        },
    }
    return summary


def save_processed_artifacts(bundle: DatasetBundle) -> dict:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    canonical = build_canonical_timeseries(bundle.timeseries)
    adjacency = clean_adjacency(bundle.adjacency)
    summary = audit_dataset(bundle)

    canonical.to_csv(CANONICAL_TIMESERIES, index=False)
    adjacency.to_csv(CLEAN_ADJACENCY, index=False)
    AUDIT_SUMMARY.write_text(json.dumps(summary, indent=2))

    return {
        "canonical_timeseries": str(CANONICAL_TIMESERIES),
        "clean_adjacency": str(CLEAN_ADJACENCY),
        "audit_summary": str(AUDIT_SUMMARY),
        "canonical_rows": int(len(canonical)),
    }
