"""Feature engineering module for water demand forecasting.

Builds temporal, calendar, static, and spatial features.
All features are leakage-safe: computed only from data up to the forecast origin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .baselines import SERIES_KEYS

TARGET_COL = "noisy_volume_m3"

# --- Temporal feature config ---
LAG_FEATURES = [1, 2, 3, 6, 12, 24]
ROLLING_MEAN_WINDOWS = [3, 6, 12]
ROLLING_STD_WINDOWS = [6, 12]

# --- Static features ---
SYSTEM_AREA_ENCODING = {"PD": 0, "TS": 1}


def _month_to_angle(month: int | pd.Series) -> float | pd.Series:
    return 2 * np.pi * month / 12.0


def _period_ym_to_month_index(period_ym: pd.Series) -> pd.Series:
    """Convert YYYYMM to a monotonic month index (0 = 202001)."""
    year = period_ym // 100
    month = period_ym % 100
    return (year - 2020) * 12 + (month - 1)


# ---------------------------------------------------------------------------
# Temporal features (lags, rolling, YoY)
# ---------------------------------------------------------------------------

def build_lag_features(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby(SERIES_KEYS, sort=False)[TARGET_COL]
    for lag in LAG_FEATURES:
        frame[f"lag_{lag}"] = grouped.shift(lag)
    return frame


def build_rolling_features(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby(SERIES_KEYS, sort=False)[TARGET_COL]
    for window in ROLLING_MEAN_WINDOWS:
        frame[f"rolling_mean_{window}"] = grouped.transform(
            lambda s, w=window: s.shift(1).rolling(window=w, min_periods=w).mean()
        )
    for window in ROLLING_STD_WINDOWS:
        frame[f"rolling_std_{window}"] = grouped.transform(
            lambda s, w=window: s.shift(1).rolling(window=w, min_periods=w).std()
        )
    return frame


def build_yoy_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Year-over-year delta and ratio vs lag_12."""
    if "lag_1" not in frame.columns or "lag_12" not in frame.columns:
        raise ValueError("Lag features must be built before YoY features")
    frame["yoy_delta"] = frame["lag_1"] - frame["lag_12"]
    safe_denom = frame["lag_12"].replace(0, np.nan)
    frame["yoy_ratio"] = frame["lag_1"] / safe_denom
    return frame


# ---------------------------------------------------------------------------
# Calendar / trend features
# ---------------------------------------------------------------------------

def build_calendar_features(frame: pd.DataFrame, period_col: str = "period_ym") -> pd.DataFrame:
    month = frame[period_col] % 100
    frame["month"] = month
    frame["month_sin"] = np.sin(_month_to_angle(month))
    frame["month_cos"] = np.cos(_month_to_angle(month))
    frame["month_sin2"] = np.sin(2 * _month_to_angle(month))
    frame["month_cos2"] = np.cos(2 * _month_to_angle(month))
    frame["trend_index"] = _period_ym_to_month_index(frame[period_col])
    return frame


# ---------------------------------------------------------------------------
# Static features
# ---------------------------------------------------------------------------

def build_ewm_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Exponentially weighted mean features (shift(1) to avoid leakage)."""
    grouped = frame.groupby(SERIES_KEYS, sort=False)[TARGET_COL]
    for span in [3, 6]:
        frame[f"ewm_mean_{span}"] = grouped.transform(
            lambda s, sp=span: s.shift(1).ewm(span=sp, min_periods=sp).mean()
        )
    return frame


def build_series_meta_features(
    frame: pd.DataFrame,
    train_end_ym: int,
) -> pd.DataFrame:
    """Series-level statistics computed from training data only.

    Must be called per fold with the fold's train_end.
    """
    train_data = frame[frame["period_ym"] <= train_end_ym]

    stats = train_data.groupby(SERIES_KEYS)[TARGET_COL].agg(
        series_mean_volume="mean",
        series_std_volume="std",
    ).reset_index()
    stats["series_cv"] = stats["series_std_volume"] / stats["series_mean_volume"].replace(0, np.nan)
    stats = stats.drop(columns=["series_std_volume"])

    def _trend_slope(sub):
        y = np.log1p(sub[TARGET_COL].values)
        x = _period_ym_to_month_index(sub["period_ym"]).values.astype(float)
        if len(x) < 3:
            return np.nan
        x_c = x - x.mean()
        denom = (x_c ** 2).sum()
        if denom == 0:
            return np.nan
        return float((x_c * (y - y.mean())).sum() / denom)

    trend = train_data.groupby(SERIES_KEYS).apply(_trend_slope).reset_index(name="series_trend_strength")
    stats = stats.merge(trend, on=SERIES_KEYS, how="left")

    def _acf12(sub):
        vals = sub[TARGET_COL].values
        if len(vals) < 24:
            return np.nan
        mean = vals.mean()
        c0 = np.sum((vals - mean) ** 2) / len(vals)
        if c0 == 0:
            return np.nan
        c12 = np.sum((vals[12:] - mean) * (vals[:-12] - mean)) / len(vals)
        return float(c12 / c0)

    acf = (
        train_data.sort_values(SERIES_KEYS + ["period_ym"])
        .groupby(SERIES_KEYS)
        .apply(_acf12)
        .reset_index(name="series_acf12")
    )
    stats = stats.merge(acf, on=SERIES_KEYS, how="left")

    for col in SERIES_META_COLS:
        if col in frame.columns:
            frame = frame.drop(columns=[col])
    return frame.merge(stats, on=SERIES_KEYS, how="left")


def build_static_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame["system_area_id"] = frame["system_area"].map(SYSTEM_AREA_ENCODING).astype(int)
    return frame


# ---------------------------------------------------------------------------
# Spatial features (from adjacency graph)
# ---------------------------------------------------------------------------

def build_node_degree(canonical_ts: pd.DataFrame, adjacency: pd.DataFrame) -> pd.DataFrame:
    """Add node_degree column based on adjacency graph."""
    degree = adjacency.groupby("cell_id").size().reset_index(name="node_degree")
    return canonical_ts.merge(degree, on="cell_id", how="left").fillna({"node_degree": 0})


def _build_neighbour_lookup(adjacency: pd.DataFrame) -> dict[str, list[str]]:
    """Build cell_id -> list of neighbour cell_ids."""
    lookup: dict[str, list[str]] = {}
    for _, row in adjacency.iterrows():
        lookup.setdefault(row["cell_id"], []).append(row["neighbour_cell_id"])
    return lookup


def build_spatial_features(
    frame: pd.DataFrame,
    adjacency: pd.DataFrame,
) -> pd.DataFrame:
    """Add spatial features: neighbour mean at origin, at t-12, and spatial ratio.

    For each (cell_id, rate_category_id, period_ym), compute:
    - neighbour_mean: mean volume of neighbours with same rate_category at same period
    - neighbour_mean_lag12: same but 12 months earlier
    - spatial_ratio: lag_1 / neighbour_mean (gamma_t cancels)
    """
    neighbour_lookup = _build_neighbour_lookup(adjacency)

    # Build a fast lookup: (cell_id, rate_category_id, period_ym) -> volume
    volume_index = frame.set_index(["cell_id", "rate_category_id", "period_ym"])[TARGET_COL]

    # For each row, compute neighbour mean at same period
    neighbour_means = []
    neighbour_means_lag12 = []

    for _, row in frame.iterrows():
        cell = row["cell_id"]
        cat = row["rate_category_id"]
        period = row["period_ym"]

        neighbours = neighbour_lookup.get(cell, [])
        if not neighbours:
            neighbour_means.append(np.nan)
            neighbour_means_lag12.append(np.nan)
            continue

        # Same period neighbour values
        vals = []
        vals_lag12 = []
        for nb in neighbours:
            key = (nb, cat, period)
            if key in volume_index.index:
                vals.append(volume_index.loc[key])
            # 12 months ago
            period_lag12 = _shift_period_ym(period, -12)
            key12 = (nb, cat, period_lag12)
            if key12 in volume_index.index:
                vals_lag12.append(volume_index.loc[key12])

        neighbour_means.append(np.mean(vals) if vals else np.nan)
        neighbour_means_lag12.append(np.mean(vals_lag12) if vals_lag12 else np.nan)

    frame["neighbour_mean"] = neighbour_means
    frame["neighbour_mean_lag12"] = neighbour_means_lag12

    # Spatial ratio (gamma_t cancels)
    safe_denom = frame["neighbour_mean"].replace(0, np.nan)
    frame["spatial_ratio"] = frame.get("lag_1", frame[TARGET_COL]) / safe_denom

    return frame


def _shift_period_ym(period_ym: int, months: int) -> int:
    """Shift a YYYYMM period by a number of months."""
    year = period_ym // 100
    month = (period_ym % 100) - 1  # 0-indexed
    total = year * 12 + month + months
    return (total // 12) * 100 + (total % 12) + 1


# ---------------------------------------------------------------------------
# Vectorized spatial features (much faster than row-by-row)
# ---------------------------------------------------------------------------

def build_spatial_features_fast(
    frame: pd.DataFrame,
    adjacency: pd.DataFrame,
) -> pd.DataFrame:
    """Vectorized spatial features using merge + groupby.

    Much faster than the row-by-row version for large frames.
    """
    # 1) Node degree
    degree = adjacency.groupby("cell_id").size().reset_index(name="node_degree")
    frame = frame.merge(degree, on="cell_id", how="left")
    frame["node_degree"] = frame["node_degree"].fillna(0).astype(int)

    # 2) Neighbour mean at same period (same rate_category)
    # Expand each row's neighbours via adjacency
    # adjacency: (cell_id, neighbour_cell_id) — for each focal cell, get neighbour volumes
    nb_edges = adjacency.rename(columns={"cell_id": "focal_cell", "neighbour_cell_id": "nb_cell"})
    volume_table = frame[["cell_id", "rate_category_id", "period_ym", TARGET_COL]].copy()
    volume_table = volume_table.rename(columns={"cell_id": "nb_cell", TARGET_COL: "nb_volume"})

    neighbour_volumes = nb_edges.merge(volume_table, on="nb_cell", how="inner")

    nb_mean = (
        neighbour_volumes
        .groupby(["focal_cell", "rate_category_id", "period_ym"], as_index=False)["nb_volume"]
        .mean()
        .rename(columns={"focal_cell": "cell_id", "nb_volume": "neighbour_mean"})
    )
    frame = frame.merge(nb_mean, on=["cell_id", "rate_category_id", "period_ym"], how="left")

    # 3) Neighbour mean at t-12
    nb_mean_lag12 = nb_mean.copy()
    nb_mean_lag12["period_ym"] = nb_mean_lag12["period_ym"].apply(lambda p: _shift_period_ym(p, 12))
    nb_mean_lag12 = nb_mean_lag12.rename(columns={"neighbour_mean": "neighbour_mean_lag12"})
    frame = frame.merge(
        nb_mean_lag12, on=["cell_id", "rate_category_id", "period_ym"], how="left"
    )

    # 4) Spatial ratio (lag_1 / neighbour_mean — gamma_t cancels)
    safe_denom = frame["neighbour_mean"].replace(0, np.nan)
    if "lag_1" in frame.columns:
        frame["spatial_ratio"] = frame["lag_1"] / safe_denom
    else:
        frame["spatial_ratio"] = frame[TARGET_COL] / safe_denom

    return frame


# ---------------------------------------------------------------------------
# Full feature pipeline
# ---------------------------------------------------------------------------

TEMPORAL_FEATURE_COLS = (
    [f"lag_{lag}" for lag in LAG_FEATURES]
    + [f"rolling_mean_{w}" for w in ROLLING_MEAN_WINDOWS]
    + [f"rolling_std_{w}" for w in ROLLING_STD_WINDOWS]
    + ["yoy_delta", "yoy_ratio"]
)

CALENDAR_FEATURE_COLS = ["month", "month_sin", "month_cos", "trend_index"]
FOURIER2_FEATURE_COLS = ["month_sin2", "month_cos2"]
EWM_FEATURE_COLS = ["ewm_mean_3", "ewm_mean_6"]
SERIES_META_COLS = ["series_mean_volume", "series_cv", "series_trend_strength", "series_acf12"]

STATIC_FEATURE_COLS = ["rate_category_id", "system_area_id", "h3_resolution"]

SPATIAL_FEATURE_COLS = ["node_degree", "neighbour_mean", "neighbour_mean_lag12", "spatial_ratio"]

ALL_FEATURE_COLS = (
    TEMPORAL_FEATURE_COLS + CALENDAR_FEATURE_COLS + STATIC_FEATURE_COLS + SPATIAL_FEATURE_COLS
)


def build_full_feature_frame(
    canonical_ts: pd.DataFrame,
    adjacency: pd.DataFrame,
) -> pd.DataFrame:
    """Build the complete feature frame from canonical timeseries and adjacency.

    Returns a DataFrame with all features, dropping rows with NaN in required columns.
    """
    frame = canonical_ts.sort_values(SERIES_KEYS + ["period_ym"]).copy()

    # Static
    frame = build_static_features(frame)

    # Temporal
    frame = build_lag_features(frame)
    frame = build_rolling_features(frame)
    frame = build_ewm_features(frame)
    frame = build_yoy_features(frame)

    # Calendar
    frame = build_calendar_features(frame)

    # Spatial
    frame = build_spatial_features_fast(frame, adjacency)

    return frame


def feature_columns_direct() -> list[str]:
    """Feature columns used by the direct multi-horizon model.

    Includes all temporal/calendar/static/spatial features plus horizon and target_month.
    """
    return ALL_FEATURE_COLS + ["horizon", "target_month", "target_month_sin", "target_month_cos"]


# ---------------------------------------------------------------------------
# V9 Domain-informed features
# ---------------------------------------------------------------------------

# Column name constants for V9 feature groups
V9_MARCH_COLS = ["is_march", "months_since_march"]
V9_BILLING_COLS = ["is_quarter_end", "billing_half"]
V9_REGIME_COLS = [
    "is_covid_period", "is_drought_2022", "smart_meter_era", "smart_meter_x_area",
]
V9_AREA_SEASONAL_COLS = [
    "pd_academic_effect", "ts_tourism_summer", "ts_october_event", "pd_cat3_decline",
]
V9_GAMMA_COLS = ["global_monthly_mean", "volume_ratio_to_global"]
V9_DENSITY_COLS = ["volume_density"]
V9_INTERACTION_COLS = ["lag12_x_monthsin"]
V9_SPATIAL_EXTRA_COLS = ["neighbour_std", "spatial_momentum", "area_mean_volume"]
V9_FOLD_COLS = [
    "march_anchor_value", "march_yoy_ratio", "march_correction_magnitude",
    "cell_mean_volume", "volume_x_trend",
]

# All origin-level V9 features (computed once in build_v9_origin_features)
V9_ORIGIN_FEATURE_COLS = (
    V9_MARCH_COLS + V9_BILLING_COLS + V9_REGIME_COLS + V9_AREA_SEASONAL_COLS
    + V9_GAMMA_COLS + V9_DENSITY_COLS + V9_INTERACTION_COLS + V9_SPATIAL_EXTRA_COLS
)

# All new V9 columns (origin + per-fold)
V9_ALL_NEW_COLS = V9_ORIGIN_FEATURE_COLS + V9_FOLD_COLS


def build_v9_origin_features(
    frame: pd.DataFrame,
    adjacency: pd.DataFrame,
) -> pd.DataFrame:
    """Add all V9 origin-level features to the feature frame.

    Call AFTER build_full_feature_frame (needs lags, calendar, spatial, static).
    """
    frame = _build_v9_march_billing(frame)
    frame = _build_v9_regime(frame)
    frame = _build_v9_area_seasonal(frame)
    frame = _build_v9_gamma(frame)
    frame = _build_v9_density(frame)
    frame = _build_v9_interactions(frame)
    frame = _build_v9_spatial_extra(frame, adjacency)
    return frame


def _build_v9_march_billing(frame: pd.DataFrame) -> pd.DataFrame:
    """March anchor + billing cycle features."""
    month = frame["period_ym"] % 100

    # March anchor
    frame["is_march"] = (month == 3).astype(int)
    # Distance from last March: Mar=0, Apr=1, ..., Feb=11
    frame["months_since_march"] = ((month - 3) % 12).astype(int)

    # Billing cycle
    frame["is_quarter_end"] = month.isin([3, 6, 9, 12]).astype(int)
    frame["billing_half"] = (month <= 6).astype(int)
    return frame


def _build_v9_regime(frame: pd.DataFrame) -> pd.DataFrame:
    """Structural regime indicators (COVID, drought, smart meters)."""
    period = frame["period_ym"]
    frame["is_covid_period"] = ((period >= 202003) & (period <= 202106)).astype(int)
    frame["is_drought_2022"] = ((period >= 202205) & (period <= 202211)).astype(int)
    frame["smart_meter_era"] = (period >= 202301).astype(int)
    area_id = frame.get("system_area_id", 0)
    frame["smart_meter_x_area"] = frame["smart_meter_era"] * area_id
    return frame


def _build_v9_area_seasonal(frame: pd.DataFrame) -> pd.DataFrame:
    """Area-specific seasonal interaction features."""
    month = frame["period_ym"] % 100
    is_pd = (frame["system_area"] == "PD").astype(int) if "system_area" in frame.columns else 0
    is_ts = (frame["system_area"] == "TS").astype(int) if "system_area" in frame.columns else 0

    # PD university cycle: peak exam sessions, trough Aug+Dec
    pd_map = {1: 1, 2: 1, 3: 0, 4: 0, 5: 0, 6: 1, 7: 1, 8: -1, 9: 0, 10: 0, 11: 0, 12: -1}
    frame["pd_academic_effect"] = month.map(pd_map).fillna(0).astype(int) * is_pd

    # TS summer tourism (Jun-Sep, 58.6% annual volume)
    frame["ts_tourism_summer"] = month.isin([6, 7, 8, 9]).astype(int) * is_ts

    # TS Barcolana October
    frame["ts_october_event"] = (month == 10).astype(int) * is_ts

    # PD Cat 3 structural decline from April 2024
    frame["pd_cat3_decline"] = (
        (frame["period_ym"] >= 202404)
        & (frame.get("system_area", "") == "PD")
        & (frame["rate_category_id"] == 3)
    ).astype(int)
    return frame


def _build_v9_gamma(frame: pd.DataFrame) -> pd.DataFrame:
    """γ_t proxy features (cross-sectional mean)."""
    global_mean = frame.groupby("period_ym")[TARGET_COL].transform("mean")
    frame["global_monthly_mean"] = global_mean
    safe_global = global_mean.replace(0, np.nan)
    frame["volume_ratio_to_global"] = frame[TARGET_COL] / safe_global
    return frame


def _build_v9_density(frame: pd.DataFrame) -> pd.DataFrame:
    """Volume per km² (normalizes res 6 vs res 7 scale)."""
    area_km2 = frame["h3_resolution"].map({6: 36.12, 7: 5.16})
    frame["volume_density"] = frame[TARGET_COL] / area_km2.replace(0, np.nan)
    return frame


def _build_v9_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Interaction features requiring existing lag + calendar features."""
    if "lag_12" in frame.columns and "month_sin" in frame.columns:
        frame["lag12_x_monthsin"] = frame["lag_12"] * frame["month_sin"]
    else:
        frame["lag12_x_monthsin"] = np.nan
    return frame


def _build_v9_spatial_extra(
    frame: pd.DataFrame,
    adjacency: pd.DataFrame,
) -> pd.DataFrame:
    """Second-order spatial features: neighbour_std, spatial_momentum, area_mean."""
    # Neighbour std via vectorized merge
    nb_edges = adjacency.rename(
        columns={"cell_id": "focal_cell", "neighbour_cell_id": "nb_cell"}
    )
    vol_table = frame[["cell_id", "rate_category_id", "period_ym", TARGET_COL]].copy()
    vol_table = vol_table.rename(columns={"cell_id": "nb_cell", TARGET_COL: "nb_volume"})
    nb_vols = nb_edges.merge(vol_table, on="nb_cell", how="inner")

    nb_std = (
        nb_vols
        .groupby(["focal_cell", "rate_category_id", "period_ym"], as_index=False)["nb_volume"]
        .std()
        .rename(columns={"focal_cell": "cell_id", "nb_volume": "neighbour_std"})
    )
    frame = frame.merge(
        nb_std, on=["cell_id", "rate_category_id", "period_ym"], how="left"
    )

    # Spatial momentum (neighbourhood growth)
    if "neighbour_mean" in frame.columns and "neighbour_mean_lag12" in frame.columns:
        frame["spatial_momentum"] = frame["neighbour_mean"] - frame["neighbour_mean_lag12"]
    else:
        frame["spatial_momentum"] = np.nan

    # Area mean volume (district-level trend proxy)
    if "system_area" in frame.columns:
        frame["area_mean_volume"] = frame.groupby(
            ["system_area", "period_ym"]
        )[TARGET_COL].transform("mean")
    else:
        frame["area_mean_volume"] = np.nan

    return frame


def build_v9_fold_features(
    frame: pd.DataFrame,
    train_end_ym: int,
) -> pd.DataFrame:
    """Per-fold V9 features (leakage-safe: training data only).

    Call per fold AFTER build_series_meta_features.
    """
    train_data = frame[frame["period_ym"] <= train_end_ym]

    # --- March anchor value & YoY ratio & correction magnitude ---
    march_data = train_data[train_data["period_ym"] % 100 == 3]

    if not march_data.empty:
        # Last March value per series
        last_march = (
            march_data.sort_values("period_ym")
            .groupby(SERIES_KEYS)
            .last()
            .reset_index()[SERIES_KEYS + [TARGET_COL]]
            .rename(columns={TARGET_COL: "march_anchor_value"})
        )

        # March YoY ratio (last / second-to-last)
        ratios: list[dict] = []
        for keys, g in march_data.sort_values("period_ym").groupby(SERIES_KEYS):
            vals = g[TARGET_COL].values
            cell, cat = keys
            r = vals[-1] / vals[-2] if len(vals) >= 2 and vals[-2] != 0 else np.nan
            ratios.append({"cell_id": cell, "rate_category_id": cat, "march_yoy_ratio": r})
        march_ratio_df = pd.DataFrame(ratios)

        # Correction magnitude (March - preceding February)
        feb_lookup = train_data.set_index(
            ["cell_id", "rate_category_id", "period_ym"]
        )[TARGET_COL].to_dict()
        corrections: list[dict] = []
        for keys, g in march_data.sort_values("period_ym").groupby(SERIES_KEYS):
            cell, cat = keys
            mar_ym = int(g["period_ym"].iloc[-1])
            mar_vol = g[TARGET_COL].iloc[-1]
            feb_ym = mar_ym - 1  # e.g. 202303 → 202302
            feb_vol = feb_lookup.get((cell, cat, feb_ym))
            mag = mar_vol - feb_vol if feb_vol is not None else np.nan
            corrections.append({
                "cell_id": cell, "rate_category_id": cat,
                "march_correction_magnitude": mag,
            })
        correction_df = pd.DataFrame(corrections)

        # Merge
        for col in ["march_anchor_value", "march_yoy_ratio", "march_correction_magnitude"]:
            if col in frame.columns:
                frame = frame.drop(columns=[col])
        frame = frame.merge(last_march, on=SERIES_KEYS, how="left")
        frame = frame.merge(march_ratio_df, on=SERIES_KEYS, how="left")
        frame = frame.merge(correction_df, on=SERIES_KEYS, how="left")
    else:
        for col in ["march_anchor_value", "march_yoy_ratio", "march_correction_magnitude"]:
            frame[col] = np.nan

    # --- Cell mean volume ---
    cell_mean = (
        train_data.groupby("cell_id")[TARGET_COL]
        .mean()
        .reset_index(name="cell_mean_volume")
    )
    if "cell_mean_volume" in frame.columns:
        frame = frame.drop(columns=["cell_mean_volume"])
    frame = frame.merge(cell_mean, on="cell_id", how="left")

    # --- volume × trend (needs series_mean_volume from build_series_meta_features) ---
    if "series_mean_volume" in frame.columns and "trend_index" in frame.columns:
        frame["volume_x_trend"] = frame["series_mean_volume"] * frame["trend_index"]
    else:
        frame["volume_x_trend"] = np.nan

    return frame
