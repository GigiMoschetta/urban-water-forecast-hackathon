#!/usr/bin/env python3
"""
Comprehensive validation: verify data.json matches source CSVs exactly.
Checks: actuals, ensemble predictions, naive predictions, metrics, KPIs.
"""
import json, pathlib, sys
import pandas as pd
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1].parent
DATA_JSON = pathlib.Path(__file__).resolve().parents[1] / "public" / "data.json"

CAT_NAMES = {1: "Domestic", 2: "Commercial", 3: "Industrial", 4: "Farming", 5: "Other"}

def load_sources():
    hist = pd.read_csv(ROOT / "data" / "processed" / "timeseries_canonical.csv")
    ens = pd.read_csv(ROOT / "outputs/predictions/ensemble_3m_v1_v9o_nested_predictions.csv")
    naive = pd.read_csv(ROOT / "outputs/predictions/seasonal_naive_predictions.csv")
    return hist, ens, naive

def load_dashboard():
    return json.loads(DATA_JSON.read_text())

def agg_sum(df, ym_col, val_col):
    """Aggregate to monthly sum, return dict {ym: value}."""
    agg = df.groupby(ym_col)[val_col].sum()
    return {int(k): round(float(v), 0) for k, v in agg.items()}

errors = []
warnings = []

def check(condition, msg, is_warning=False):
    if not condition:
        if is_warning:
            warnings.append(msg)
        else:
            errors.append(msg)
        return False
    return True

def verify_series(name, dashboard_pts, expected_dict, tolerance=1.0):
    """Verify a series of {ym: v} points matches."""
    dash_dict = {p["ym"]: p["v"] for p in dashboard_pts}
    
    # Check all expected points are present
    for ym, expected_v in sorted(expected_dict.items()):
        if ym not in dash_dict:
            check(False, f"  {name}: missing ym={ym} (expected v={expected_v})")
            continue
        diff = abs(dash_dict[ym] - expected_v)
        if diff > tolerance:
            check(False, f"  {name}: ym={ym} mismatch: dashboard={dash_dict[ym]}, expected={expected_v}, diff={diff}")
    
    # Check for extra points in dashboard that shouldn't be there
    for ym in dash_dict:
        if ym not in expected_dict:
            check(False, f"  {name}: extra ym={ym} in dashboard (v={dash_dict[ym]}) not in source", is_warning=True)

def main():
    print("=" * 70)
    print("FLOW_IT DATA VALIDATION")
    print("=" * 70)
    
    hist, ens_df, naive_df = load_sources()
    dash = load_dashboard()
    cats = dash["categories"]
    
    print(f"\nSource files loaded:")
    print(f"  Historical: {len(hist)} rows, {hist['cell_id'].nunique()} cells")
    print(f"  Ensemble:   {len(ens_df)} rows, folds: {sorted(ens_df['fold'].unique())}")
    print(f"  Naive:      {len(naive_df)} rows")
    print(f"  Dashboard:  {len(cats)} categories in data.json")
    print()
    
    # ═════════════════════════════════════════════════════════════════════
    # 1. VERIFY "ALL CATEGORIES" ACTUALS
    # ═════════════════════════════════════════════════════════════════════
    print("━" * 60)
    print("1. ACTUALS VERIFICATION (historical volumes)")
    print("━" * 60)
    
    # All categories
    expected_actuals = agg_sum(hist, "period_ym", "noisy_volume_m3")
    print(f"\n[All Categories] {len(expected_actuals)} months expected")
    verify_series("All/actuals", cats["all"]["actuals"], expected_actuals)
    
    # Per rate category
    for cat_id, cat_name in CAT_NAMES.items():
        h_sub = hist[hist["rate_category_id"] == cat_id]
        if h_sub.empty:
            continue
        expected = agg_sum(h_sub, "period_ym", "noisy_volume_m3")
        key = str(cat_id)
        if key in cats:
            print(f"\n[Cat {cat_id} / {cat_name}] {len(expected)} months")
            verify_series(f"Cat{cat_id}/actuals", cats[key]["actuals"], expected)
        else:
            check(False, f"Category '{key}' missing from dashboard data")
    
    # Per area
    for area in ["PD", "TS"]:
        h_sub = hist[hist["system_area"] == area]
        expected = agg_sum(h_sub, "period_ym", "noisy_volume_m3")
        key = f"area_{area}"
        if key in cats:
            print(f"\n[Area {area}] {len(expected)} months")
            verify_series(f"Area_{area}/actuals", cats[key]["actuals"], expected)
        else:
            check(False, f"Area '{key}' missing from dashboard data")
    
    # ═════════════════════════════════════════════════════════════════════
    # 2. VERIFY ENSEMBLE PREDICTIONS (CV folds 2022-2025)
    # ═════════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 60}")
    print("2. ENSEMBLE PREDICTIONS VERIFICATION")
    print("━" * 60)
    
    # The build_data.py logic:
    # - For 2020-2021: uses naive proxy (= 2020 actuals, then shifted)
    # - For 2022-2025: uses ensemble CV predictions
    # - For 2026: uses fold_4 predictions shifted by +100
    
    # Check CV portion (2022-2025)
    ens_cv = agg_sum(ens_df, "period_ym", "prediction")
    print(f"\n[All Categories - Ensemble CV] {len(ens_cv)} months from CV folds")
    
    dash_ens_all = {p["ym"]: p["v"] for p in cats["all"]["ensemble"]}
    for ym, expected_v in sorted(ens_cv.items()):
        if ym in dash_ens_all:
            diff = abs(dash_ens_all[ym] - expected_v)
            if diff > 1.0:
                check(False, f"  All/ensemble CV: ym={ym} mismatch: dashboard={dash_ens_all[ym]}, source={expected_v}, diff={diff}")
    
    # Check 2026 forecast = fold_4 shifted by +100
    f4 = ens_df[ens_df["fold"] == "fold_4"].copy()
    f4["period_ym"] = f4["period_ym"] + 100
    ens_2026 = agg_sum(f4, "period_ym", "prediction")
    print(f"\n[All Categories - Ensemble 2026 Forecast] {len(ens_2026)} months from fold_4 shift")
    for ym, expected_v in sorted(ens_2026.items()):
        if ym in dash_ens_all:
            diff = abs(dash_ens_all[ym] - expected_v)
            if diff > 1.0:
                check(False, f"  All/ensemble 2026: ym={ym} mismatch: dashboard={dash_ens_all[ym]}, source={expected_v}, diff={diff}")
        else:
            check(False, f"  All/ensemble 2026: ym={ym} missing from dashboard")
    
    # Per area ensemble
    for area in ["PD", "TS"]:
        e_sub = ens_df[ens_df["system_area"] == area]
        ens_area_cv = agg_sum(e_sub, "period_ym", "prediction")
        key = f"area_{area}"
        if key in cats:
            print(f"\n[Area {area} - Ensemble CV] {len(ens_area_cv)} months")
            dash_area = {p["ym"]: p["v"] for p in cats[key]["ensemble"]}
            for ym, expected_v in sorted(ens_area_cv.items()):
                if ym in dash_area:
                    diff = abs(dash_area[ym] - expected_v)
                    if diff > 1.0:
                        check(False, f"  Area_{area}/ensemble: ym={ym} diff={diff}")
    
    # ═════════════════════════════════════════════════════════════════════
    # 3. VERIFY NAIVE PREDICTIONS
    # ═════════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 60}")
    print("3. NAIVE PREDICTIONS VERIFICATION")
    print("━" * 60)
    
    naive_cv = agg_sum(naive_df, "period_ym", "prediction")
    print(f"\n[All Categories - Naive CV] {len(naive_cv)} months")
    
    dash_naive_all = {p["ym"]: p["v"] for p in cats["all"]["naive"]}
    for ym, expected_v in sorted(naive_cv.items()):
        if ym in dash_naive_all:
            diff = abs(dash_naive_all[ym] - expected_v)
            if diff > 1.0:
                check(False, f"  All/naive CV: ym={ym} mismatch: dashboard={dash_naive_all[ym]}, source={expected_v}, diff={diff}")
    
    # Check 2020 naive anchor = 2020 actuals
    h_2020 = hist[hist["period_ym"] // 100 == 2020]
    naive_2020 = agg_sum(h_2020, "period_ym", "noisy_volume_m3")
    print(f"\n[All Categories - Naive 2020 Anchor] {len(naive_2020)} months (= 2020 actuals)")
    for ym, expected_v in sorted(naive_2020.items()):
        if ym in dash_naive_all:
            diff = abs(dash_naive_all[ym] - expected_v)
            if diff > 1.0:
                check(False, f"  All/naive 2020: ym={ym} mismatch: dashboard={dash_naive_all[ym]}, source={expected_v}")
        else:
            check(False, f"  All/naive 2020: ym={ym} missing from dashboard")
    
    # Check 2021 naive = 2020 actuals shifted
    h_2020_shifted = h_2020.copy()
    h_2020_shifted["period_ym"] = h_2020_shifted["period_ym"] + 100
    naive_2021 = agg_sum(h_2020_shifted, "period_ym", "noisy_volume_m3")
    print(f"\n[All Categories - Naive 2021] {len(naive_2021)} months (= 2020 shifted)")
    for ym, expected_v in sorted(naive_2021.items()):
        if ym in dash_naive_all:
            diff = abs(dash_naive_all[ym] - expected_v)
            if diff > 1.0:
                check(False, f"  All/naive 2021: ym={ym} mismatch: dashboard={dash_naive_all[ym]}, source={expected_v}")
    
    # ═════════════════════════════════════════════════════════════════════
    # 4. VERIFY METRICS (MAPE, MAE, RMSE)
    # ═════════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 60}")
    print("4. METRICS VERIFICATION (MAPE, MAE, RMSE)")
    print("━" * 60)
    
    metrics = dash["metrics"]
    
    # Ensemble overall metrics
    ens_valid = ens_df[ens_df["actual"] != 0]
    expected_mape = round(float(ens_valid["ape"].mean() * 100), 2)
    expected_mae = round(float(ens_df["abs_error"].mean()), 0)
    expected_rmse = round(float(np.sqrt(ens_df["squared_error"].mean())), 0)
    
    print(f"\n[Ensemble Overall]")
    print(f"  MAPE: dashboard={metrics['ensemble']['mape']}%, expected={expected_mape}%")
    check(abs(metrics["ensemble"]["mape"] - expected_mape) < 0.01, 
          f"  Ensemble MAPE mismatch: {metrics['ensemble']['mape']} vs {expected_mape}")
    print(f"  MAE:  dashboard={metrics['ensemble']['mae']}, expected={expected_mae}")
    check(abs(metrics["ensemble"]["mae"] - expected_mae) < 1,
          f"  Ensemble MAE mismatch: {metrics['ensemble']['mae']} vs {expected_mae}")
    print(f"  RMSE: dashboard={metrics['ensemble']['rmse']}, expected={expected_rmse}")
    check(abs(metrics["ensemble"]["rmse"] - expected_rmse) < 1,
          f"  Ensemble RMSE mismatch: {metrics['ensemble']['rmse']} vs {expected_rmse}")
    
    # Naive overall metrics
    naive_valid = naive_df[naive_df["actual"] != 0]
    n_mape = round(float(naive_valid["ape"].mean() * 100), 2)
    n_mae = round(float(naive_df["abs_error"].mean()), 0)
    n_rmse = round(float(np.sqrt(naive_df["squared_error"].mean())), 0)
    
    print(f"\n[Naive Overall]")
    print(f"  MAPE: dashboard={metrics['naive']['mape']}%, expected={n_mape}%")
    check(abs(metrics["naive"]["mape"] - n_mape) < 0.01,
          f"  Naive MAPE mismatch: {metrics['naive']['mape']} vs {n_mape}")
    print(f"  MAE:  dashboard={metrics['naive']['mae']}, expected={n_mae}")
    check(abs(metrics["naive"]["mae"] - n_mae) < 1,
          f"  Naive MAE mismatch: {metrics['naive']['mae']} vs {n_mae}")
    print(f"  RMSE: dashboard={metrics['naive']['rmse']}, expected={n_rmse}")
    check(abs(metrics["naive"]["rmse"] - n_rmse) < 1,
          f"  Naive RMSE mismatch: {metrics['naive']['rmse']} vs {n_rmse}")
    
    # Per-fold metrics
    print(f"\n[Per-Fold Ensemble MAPE]")
    for fold_name in sorted(ens_df["fold"].unique()):
        f = ens_df[ens_df["fold"] == fold_name]
        valid = f[f["actual"] != 0]
        exp_mape = round(float(valid["ape"].mean() * 100), 2)
        
        dash_fold = next((fm for fm in metrics["ensemble"]["folds"] if fm["fold"] == fold_name), None)
        if dash_fold:
            print(f"  {fold_name}: dashboard={dash_fold['mape']}%, expected={exp_mape}%")
            check(abs(dash_fold["mape"] - exp_mape) < 0.01,
                  f"  Fold {fold_name} MAPE mismatch: {dash_fold['mape']} vs {exp_mape}")
        else:
            check(False, f"  Fold {fold_name} missing from dashboard metrics")
    
    # ═════════════════════════════════════════════════════════════════════
    # 5. VERIFY KPIs
    # ═════════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 60}")
    print("5. KPIs VERIFICATION")
    print("━" * 60)
    
    kpis = dash["kpis"]
    MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    
    for key_name, label in [("all", "All"), ("1", "Domestic"), ("area_PD", "Padova"), ("area_TS", "Trieste")]:
        if key_name not in kpis or key_name not in cats:
            continue
        k = kpis[key_name]
        c = cats[key_name]
        
        # Total historical
        total_hist = sum(p["v"] for p in c["actuals"])
        print(f"\n[{label}]")
        print(f"  Total Historical: dashboard={k['totalHistorical']}, computed={total_hist}")
        check(abs(k["totalHistorical"] - total_hist) < 1,
              f"  {label} totalHistorical mismatch: {k['totalHistorical']} vs {total_hist}")
        
        # Total forecast
        forecast_pts = [p for p in c["ensemble"] if p["ym"] // 100 == 2026]
        total_forecast = sum(p["v"] for p in forecast_pts)
        print(f"  Total Forecast:   dashboard={k['totalForecast']}, computed={total_forecast}")
        check(abs(k["totalForecast"] - total_forecast) < 1,
              f"  {label} totalForecast mismatch: {k['totalForecast']} vs {total_forecast}")
        
        # Peak month
        if c["actuals"]:
            peak_pt = max(c["actuals"], key=lambda p: p["v"])
            peak_exp = MONTH_LABELS[(peak_pt["ym"] % 100) - 1] + " " + str(peak_pt["ym"] // 100)
            print(f"  Peak Month:       dashboard='{k['peakMonth']}', computed='{peak_exp}'")
            check(k["peakMonth"] == peak_exp,
                  f"  {label} peakMonth mismatch: '{k['peakMonth']}' vs '{peak_exp}'")
        
        # YoY Growth
        vol_2025 = sum(p["v"] for p in c["actuals"] if p["ym"] // 100 == 2025)
        vol_2024 = sum(p["v"] for p in c["actuals"] if p["ym"] // 100 == 2024)
        exp_yoy = round((vol_2025 / vol_2024 - 1) * 100, 1) if vol_2024 > 0 else 0
        print(f"  YoY Growth:       dashboard={k['yoyGrowth']}%, computed={exp_yoy}%")
        check(abs(k["yoyGrowth"] - exp_yoy) < 0.1,
              f"  {label} YoY mismatch: {k['yoyGrowth']} vs {exp_yoy}")
    
    # ═════════════════════════════════════════════════════════════════════
    # 6. VERIFY SAMPLE CELLS (spot check)
    # ═════════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 60}")
    print("6. CELL-LEVEL SPOT CHECKS (5 random cells)")
    print("━" * 60)
    
    cells = dash.get("cells", [])
    np.random.seed(42)
    sample_cells = np.random.choice([c["id"] for c in cells], size=min(5, len(cells)), replace=False)
    
    for cid in sample_cells:
        key = f"cell_{cid}"
        if key not in cats:
            check(False, f"  Cell {cid[:8]} missing from dashboard")
            continue
        
        h_sub = hist[hist["cell_id"] == cid]
        exp_actuals = agg_sum(h_sub, "period_ym", "noisy_volume_m3")
        
        print(f"\n[Cell {cid[:8]}] {len(exp_actuals)} months")
        verify_series(f"Cell_{cid[:8]}/actuals", cats[key]["actuals"], exp_actuals)
        
        # Check ensemble for this cell
        e_sub = ens_df[ens_df["cell_id"] == cid]
        if not e_sub.empty:
            exp_ens_cv = agg_sum(e_sub, "period_ym", "prediction")
            dash_ens = {p["ym"]: p["v"] for p in cats[key]["ensemble"]}
            mismatches = 0
            for ym, ev in exp_ens_cv.items():
                if ym in dash_ens and abs(dash_ens[ym] - ev) > 1.0:
                    mismatches += 1
            if mismatches > 0:
                check(False, f"  Cell_{cid[:8]}/ensemble: {mismatches} CV point mismatches")
            else:
                print(f"  Ensemble CV: ✓ {len(exp_ens_cv)} points match")
    
    # ═════════════════════════════════════════════════════════════════════
    # 7. VERIFY CELL METADATA
    # ═════════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 60}")
    print("7. CELL METADATA VERIFICATION")
    print("━" * 60)
    
    cell_meta = hist.groupby("cell_id").agg(
        area=("system_area", "first"),
        res=("h3_resolution", "first")
    ).reset_index()
    
    print(f"  Source cells: {len(cell_meta)}")
    print(f"  Dashboard cells: {len(cells)}")
    check(len(cells) == len(cell_meta), f"  Cell count mismatch: {len(cells)} vs {len(cell_meta)}")
    
    for c in cells:
        src = cell_meta[cell_meta["cell_id"] == c["id"]]
        if src.empty:
            check(False, f"  Cell {c['id'][:8]} in dashboard but not in source")
            continue
        check(c["area"] == src.iloc[0]["area"],
              f"  Cell {c['id'][:8]} area mismatch: {c['area']} vs {src.iloc[0]['area']}")
    
    # ═════════════════════════════════════════════════════════════════════
    # 8. VERIFY CATEGORY COUNTS IN CATEGORIES DICT
    # ═════════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 60}")
    print("8. CATEGORY COVERAGE VERIFICATION")
    print("━" * 60)
    
    expected_keys = {"all"} | {str(i) for i in range(1, 6)} | {"area_PD", "area_TS"}
    # Plus cell keys and cell+category keys
    actual_top_keys = {k for k in cats if not k.startswith("cell_")}
    cell_keys = {k for k in cats if k.startswith("cell_")}
    
    print(f"  Top-level keys expected: {sorted(expected_keys)}")
    print(f"  Top-level keys present:  {sorted(actual_top_keys)}")
    for ek in expected_keys:
        check(ek in cats, f"  Missing expected key: {ek}")
    
    print(f"  Cell-level keys: {len(cell_keys)}")
    
    # ═════════════════════════════════════════════════════════════════════
    # 9. VERIFY FORECAST START
    # ═════════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 60}")
    print("9. FORECAST START VERIFICATION")
    print("━" * 60)
    
    fs = dash.get("forecastStart", None)
    print(f"  forecastStart in data.json: {fs}")
    check(fs == 202601, f"  forecastStart should be 202601, got {fs}")
    
    # Verify max actual ym is 202512
    max_actual_ym = max(p["ym"] for p in cats["all"]["actuals"])
    print(f"  Max actual ym: {max_actual_ym}")
    check(max_actual_ym == 202512, f"  Max actual ym should be 202512, got {max_actual_ym}", is_warning=True)
    
    # ═════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═════════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 70}")
    print("VALIDATION SUMMARY")
    print("═" * 70)
    
    if errors:
        print(f"\n❌ ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  {e}")
    else:
        print(f"\n✅ NO ERRORS FOUND")
    
    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  {w}")
    
    if not errors and not warnings:
        print("\n🎉 ALL DATA IS CONSISTENT — dashboard matches source data perfectly!")
    elif not errors:
        print(f"\n✅ Core data is correct. {len(warnings)} minor notes above.")
    else:
        print(f"\n❌ {len(errors)} issues need attention!")
    
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
