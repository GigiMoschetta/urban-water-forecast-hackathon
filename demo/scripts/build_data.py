#!/usr/bin/env python3
"""Build data.json for Flow_IT demo."""
import json, pathlib
import pandas as pd
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1].parent
OUT  = pathlib.Path(__file__).resolve().parents[1] / "public" / "data.json"

CAT_NAMES = {1: "Domestic", 2: "Commercial", 3: "Industrial", 4: "Farming", 5: "Other"}
MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

DOMAIN_EVENTS_RAW = [
    # ── PANDEMIC ──────────────────────────────────────────────────────────────
    {"id": "covid_lock", "m": 3, "endM": 5, "years": [2020], "area": None,
     "icon": "🦠", "title": "COVID-19 National Lockdown",
     "category": "Socio-Economic", "impact": "Structural Break",
     "impactColor": "#f59e0b",
     "relevantCategories": [1, 2, 3],
     "insights": [
         "National lockdown Mar–May 2020",
         "Domestic demand surged +18%, industrial collapsed",
         "Commercial/public facilities closed",
     ]},
    {"id": "covid_rebound", "m": 6, "endM": 12, "years": [2020], "area": None,
     "icon": "↗", "title": "Post-Lockdown Rebound",
     "category": "Socio-Economic", "impact": "Gradual Recovery",
     "impactColor": "#84cc16",
     "relevantCategories": [2, 3, 5],
     "insights": [
         "Phased reopening from June 2020",
         "Industrial demand partially recovering",
         "Tourism absent — ports and cruise terminals closed",
     ]},

    # ── CLIMATE / DROUGHT ────────────────────────────────────────────────────
    {"id": "floods_21", "m": 7, "endM": 7, "years": [2021], "area": None,
     "icon": "🌧", "title": "Flash Floods July 2021",
     "category": "Climate", "impact": "Sudden Demand Shift",
     "impactColor": "#3b82f6",
     "relevantCategories": [1, 4],
     "insights": [
         "Intense localized storms in Veneto (Jul 27)",
         "Temperature anomaly +1.5°C for summer 2021",
         "High summer peaks followed by sudden drops",
     ]},
    {"id": "drought_22", "m": 5, "endM": 10, "years": [2022], "area": None,
     "icon": "🔥", "title": "Water Crisis 2022",
     "category": "Climate", "impact": "Demand Suppressed by Law",
     "impactColor": "#ef4444",
     "relevantCategories": [1, 2, 3, 4, 5],
     "insights": [
         "Veneto state of emergency declared May 3, 2022",
         "FVG declared 'State of Water Suffering' Jun 23",
         "Padova mayoral ordinances: non-essential use banned",
         "Precipitation −30–40% below normal",
     ]},
    {"id": "drought_22_pd", "m": 7, "endM": 10, "years": [2022], "area": "PD",
     "icon": "⚖", "title": "Padova Rationing Ordinances",
     "category": "Regulatory", "impact": "Demand Legally Suppressed",
     "impactColor": "#dc2626",
     "relevantCategories": [1, 4],
     "insights": [
         "Ordinanza n.11 (25/07/2022): urgent water measures",
         "Ordinanza n.13 (03/08/2022): canal health measures",
         "Ordinanza n.31 (18/10/2022): extended restrictions",
     ]},
    {"id": "drought_23_ext", "m": 1, "endM": 11, "years": [2023], "area": None,
     "icon": "⏳", "title": "Emergency Extension 2023",
     "category": "Regulatory", "impact": "Continued Suppression",
     "impactColor": "#f97316",
     "relevantCategories": [1, 2, 3, 4, 5],
     "insights": [
         "National emergency extended 12 months for Veneto/FVG",
         "FVG water suffering state active until Nov 28, 2023",
         "Demand models show 12–15% artificial suppression",
     ]},
    {"id": "fvg_revoke", "m": 11, "endM": 11, "years": [2023], "area": "TS",
     "icon": "✓", "title": "FVG Emergency Revoked",
     "category": "Regulatory", "impact": "Demand Unlocked",
     "impactColor": "#22c55e",
     "relevantCategories": [1, 2, 3, 4, 5],
     "insights": [
         "DPReg n.0194/2023 — official revocation Nov 28",
         "End of ~18 months of water restrictions in Trieste",
         "Baseline consumption expected to return to pre-drought",
     ]},
    {"id": "thermal_q4_23", "m": 10, "endM": 11, "years": [2023], "area": None,
     "icon": "🌡", "title": "Thermal Anomaly Q4 2023",
     "category": "Climate", "impact": "Extended Irrigation",
     "impactColor": "#f59e0b",
     "relevantCategories": [4],
     "insights": [
         "3rd hottest year since 1901",
         "Unseasonably warm October (+1–2°C above norm)",
         "Extended irrigation season into Sep–Oct",
     ]},
    {"id": "rain_surplus_24", "m": 6, "endM": 6, "years": [2024], "area": None,
     "icon": "💧", "title": "2024 Precipitation Surplus",
     "category": "Climate", "impact": "Aquifer Recharge",
     "impactColor": "#06b6d4",
     "relevantCategories": [4],
     "insights": [
         "Precipitation +34% above historical average",
         "Bacchiglione aquifer (PD) at 109% of hist. levels",
         "Reduced outdoor draw from natural precipitation",
     ]},
    {"id": "heatwave_24", "m": 7, "endM": 9, "years": [2024], "area": None,
     "icon": "🔥", "title": "Record Heatwave 2024",
     "category": "Climate", "impact": "Record Bio-Stress",
     "impactColor": "#ef4444",
     "relevantCategories": [1, 4],
     "insights": [
         "Hottest year on record (1992–2024)",
         "Trieste sea temps reached 27°C in July",
         "February 2024 record heat anomaly",
         "Sep 1 record: highest temperature ever for that month",
     ]},

    # ── ACADEMIC EXODUS (PD only) ─────────────────────────────────────────────
    {"id": "acad_aug", "m": 8, "endM": 8, "years": range(2020, 2027), "area": "PD",
     "icon": "☀", "title": "Academic Exodus (Summer)",
     "category": "Demographic", "impact": "20–30% Demand Drop",
     "impactColor": "#fbbf24",
     "relevantCategories": [1, 2],
     "insights": [
         "University of Padova closes ~2 weeks mid-August",
         "60,000+ students leave — Portello district empties",
         "Domestic & Commercial demand reaches annual nadir",
     ]},
    {"id": "acad_dec", "m": 12, "endM": 12, "years": range(2020, 2027), "area": "PD",
     "icon": "❄", "title": "Academic Exodus (Winter)",
     "category": "Demographic", "impact": ">70% Student Drop",
     "impactColor": "#7dd3fc",
     "relevantCategories": [1, 2],
     "insights": [
         "University closed Dec 24 – Jan 6",
         "Major reduction in Domestic & Commercial demand",
         "Combined with general holiday population exodus",
     ]},

    # ── CRUISE TOURISM (TS only) ──────────────────────────────────────────────
    {"id": "cruise", "m": 5, "endM": 10, "years": range(2022, 2027), "area": "TS",
     "icon": "🚢", "title": "Cruise Season Peak",
     "category": "Tourism", "impact": "Maritime Water Spikes",
     "impactColor": "#06b6d4",
     "relevantCategories": [2, 5],
     "insights": [
         "Turnaround ships load freshwater pierside (standard practice)",
         "MSC Fantasia (homeport TS), Costa Deliziosa, Mein Schiff 6",
         "Bi-weekly TA operations May–Oct at Stazione Marittima",
     ]},

    # ── TOURISM (TS) ─────────────────────────────────────────────────────────
    {"id": "tourism_ts_23", "m": 7, "endM": 8, "years": [2023], "area": "TS",
     "icon": "🏖", "title": "Trieste Tourism Record 2023",
     "category": "Tourism", "impact": "Baseline Demand Increase",
     "impactColor": "#14b8a6",
     "relevantCategories": [2, 5],
     "insights": [
         "Italy: record 447M tourist presences in 2023",
         "FVG: +8.5% presences vs 2022 (12.6M total)",
         "58.6% of annual tourism concentrated in summer",
     ]},
    {"id": "tourism_ts_25", "m": 6, "endM": 9, "years": [2025], "area": "TS",
     "icon": "🏖", "title": "FVG Tourism Growth 2025",
     "category": "Tourism", "impact": "Continued Baseline Elevation",
     "impactColor": "#14b8a6",
     "relevantCategories": [2, 5],
     "insights": [
         "FVG tourism on sustained post-COVID growth trajectory",
         "GO!2025 ECoC centered on Gorizia/Nova Gorica (FVG)",
         "Indirect regional visibility boost for summer TS demand",
     ]},

    # ── OPERATIONAL ──────────────────────────────────────────────────────────
    {"id": "smartmeter_ts", "m": 1, "endM": 12, "years": [2023], "area": "TS",
     "icon": "📡", "title": "Smart Meter Surge (TS)",
     "category": "Operational", "impact": "Data Resolution Improved",
     "impactColor": "#a3e635",
     "relevantCategories": [1, 2, 3, 4, 5],
     "insights": [
         "Smart water meters in Trieste: 690 → 2,800 (~4x)",
         "Padova: 339 → 3,219 (~9.5x) — PNRR-funded rollout",
         "Reduced billing estimation errors across all categories",
     ]},

    # ── PORT / INDUSTRY (TS) ─────────────────────────────────────────────────
    {"id": "port_crisis", "m": 8, "endM": 12, "years": [2025], "area": "TS",
     "icon": "⚓", "title": "Trieste Port Green Transition",
     "category": "Industrial", "impact": "Industrial Reconversion",
     "impactColor": "#64748b",
     "relevantCategories": [3],
     "insights": [
         "Area di Crisi Industriale Complessa designation (confirmed)",
         "Aug 2025 agreement: green port transition signed",
         "Phased decommission of water-intensive port industry",
     ]},
]

def expand_events():
    events = []
    for e in DOMAIN_EVENTS_RAW:
        for y in e["years"]:
            ev = e.copy()
            ev["ym"] = y * 100 + e["m"]
            ev["id"] = f"{e['id']}_{y}"
            del ev["years"], ev["m"], ev["endM"]
            # relevantCategories is preserved (not deleted)
            events.append(ev)
    return events


def build():
    print("Building data.json...")
    hist = pd.read_csv(ROOT / "data" / "processed" / "timeseries_canonical.csv")
    ens_df = pd.read_csv(ROOT / "outputs/predictions/ensemble_3m_v1_v9o_nested_predictions.csv")
    naive_df = pd.read_csv(ROOT / "outputs/predictions/seasonal_naive_predictions.csv")

    cell_meta = hist.groupby("cell_id").agg(area=("system_area", "first"), res=("h3_resolution", "first")).reset_index()
    cell_cats = hist.groupby("cell_id")["rate_category_id"].unique().to_dict()

    def agg_points(df, ym_col, val_col):
        if df.empty:
            return []
        agg = df.groupby(ym_col)[val_col].sum().reset_index().sort_values(ym_col)
        return [{"ym": int(r[ym_col]), "v": round(float(r[val_col]), 0)} for _, r in agg.iterrows()]

    def assemble(h_df, e_df, n_df):
        # ACTUALS from original dataset (2020-2025)
        actuals = agg_points(h_df, "period_ym", "noisy_volume_m3")

        # ENSEMBLE: CV predictions (2022-2025) + 2026 forecast (shifted fold_4)
        ens_cv = agg_points(e_df, "period_ym", "prediction")
        e_f4 = e_df[e_df["fold"] == "fold_4"].copy()
        if not e_f4.empty:
            e_f4 = e_f4.copy()
            e_f4["period_ym"] = e_f4["period_ym"] + 100
            ens_2026 = agg_points(e_f4, "period_ym", "prediction")
        else:
            ens_2026 = []

        # NAIVE: 2020 anchor + 2021 from 2020 actuals + CV predictions (2022-2025)
        #        + 2026 forecast from 2025 actuals
        naive_cv = agg_points(n_df, "period_ym", "prediction")

        # 2020 naive anchor: no 2019 data available → use 2020 actuals as persistence baseline
        # This anchors the naive line at the start of observations (clearly labelled in tooltip)
        h_2020_raw = h_df[h_df["period_ym"] // 100 == 2020]
        naive_2020 = agg_points(h_2020_raw, "period_ym", "noisy_volume_m3") if not h_2020_raw.empty else []

        # 2021 naive = 2020 actual values shifted by +100 (standard same-month-prior-year)
        h_2020 = h_2020_raw.copy()
        if not h_2020.empty:
            h_2020["period_ym"] = h_2020["period_ym"] + 100
            naive_2021 = agg_points(h_2020, "period_ym", "noisy_volume_m3")
        else:
            naive_2021 = []

        # 2026 naive = 2025 actual values shifted by +100
        h_2025 = h_df[h_df["period_ym"] // 100 == 2025].copy()
        if not h_2025.empty:
            h_2025["period_ym"] = h_2025["period_ym"] + 100
            naive_2026 = agg_points(h_2025, "period_ym", "noisy_volume_m3")
        else:
            naive_2026 = []

        naive = sorted(naive_2020 + naive_2021 + naive_cv + naive_2026, key=lambda p: p["ym"])

        # ENSEMBLE: extend back to 2020–2021 using naive proxy values
        # (CV folds only cover 2022–2025; pre-CV period uses naive as stand-in so all lines
        # share the same X-axis start for visual comparability)
        ensemble = sorted(naive_2020 + naive_2021 + ens_cv + ens_2026, key=lambda p: p["ym"])

        return {
            "actuals": actuals,
            "ensemble": ensemble,
            "naive": naive,
        }

    categories = {}
    print("Processing categories...")
    for key, name in [("all", "All Categories")] + [(str(i), CAT_NAMES[i]) for i in range(1, 6)]:
        h_sub = hist if key == "all" else hist[hist["rate_category_id"] == int(key)]
        if h_sub.empty:
            continue
        e_sub = ens_df if key == "all" else ens_df[ens_df["rate_category_id"] == int(key)]
        n_sub = naive_df if key == "all" else naive_df[naive_df["rate_category_id"] == int(key)]
        categories[key] = {"name": name, **assemble(h_sub, e_sub, n_sub)}

    print("Processing areas...")
    for area in ["PD", "TS"]:
        h_sub = hist[hist["system_area"] == area]
        e_sub = ens_df[ens_df["system_area"] == area]
        n_sub = naive_df[naive_df["system_area"] == area]
        categories[f"area_{area}"] = {"name": "Padova" if area == "PD" else "Trieste", **assemble(h_sub, e_sub, n_sub)}

    print(f"Processing {len(cell_meta)} cells...")
    for _, row in cell_meta.iterrows():
        cid = row["cell_id"]
        h_sub = hist[hist["cell_id"] == cid]
        e_sub = ens_df[ens_df["cell_id"] == cid]
        n_sub = naive_df[naive_df["cell_id"] == cid]
        categories[f"cell_{cid}"] = {"name": f"{row['area']} | Res{row['res']} | {cid[:8]}", **assemble(h_sub, e_sub, n_sub)}
        for cat_id in cell_cats[cid]:
            h_c = h_sub[h_sub["rate_category_id"] == cat_id]
            e_c = e_sub[e_sub["rate_category_id"] == cat_id]
            n_c = n_sub[n_sub["rate_category_id"] == cat_id]
            categories[f"cell_{cid}_{cat_id}"] = {"name": f"{CAT_NAMES[int(cat_id)]} ({cid[:8]})", **assemble(h_c, e_c, n_c)}

    # Metrics
    ens_valid = ens_df[ens_df["actual"] != 0]
    naive_valid = naive_df[naive_df["actual"] != 0]

    fold_metrics = []
    for fold_name in sorted(ens_df["fold"].unique()):
        f = ens_df[ens_df["fold"] == fold_name]
        valid = f[f["actual"] != 0]
        fold_metrics.append({"fold": fold_name, "mape": round(float(valid["ape"].mean() * 100), 2), "mae": round(float(f["abs_error"].mean()), 0), "rmse": round(float(np.sqrt(f["squared_error"].mean())), 0)})
    naive_fold_metrics = []
    for fold_name in sorted(naive_df["fold"].unique()):
        f = naive_df[naive_df["fold"] == fold_name]
        valid = f[f["actual"] != 0]
        naive_fold_metrics.append({"fold": fold_name, "mape": round(float(valid["ape"].mean() * 100), 2), "mae": round(float(f["abs_error"].mean()), 0), "rmse": round(float(np.sqrt(f["squared_error"].mean())), 0)})

    # KPIs
    kpis = {}
    for k, v in categories.items():
        total_hist = sum(p["v"] for p in v["actuals"])
        forecast_pts = [p for p in v["ensemble"] if p["ym"] // 100 == 2026]
        total_forecast = sum(p["v"] for p in forecast_pts)
        if v["actuals"]:
            peak_pt = max(v["actuals"], key=lambda p: p["v"])
            trough_pt = min(v["actuals"], key=lambda p: p["v"])
            peak_month = MONTH_LABELS[(peak_pt["ym"] % 100) - 1] + " " + str(peak_pt["ym"] // 100)
            trough_month = MONTH_LABELS[(trough_pt["ym"] % 100) - 1] + " " + str(trough_pt["ym"] // 100)
        else:
            peak_month = "-"
            trough_month = "-"
        vol_2025 = sum(p["v"] for p in v["actuals"] if p["ym"] // 100 == 2025)
        vol_2024 = sum(p["v"] for p in v["actuals"] if p["ym"] // 100 == 2024)
        yoy = round((vol_2025 / vol_2024 - 1) * 100, 1) if vol_2024 > 0 else 0
        kpis[k] = {"totalHistorical": total_hist, "totalForecast": total_forecast, "peakMonth": peak_month, "troughMonth": trough_month, "yoyGrowth": yoy}

    cell_list = [{"id": row["cell_id"], "short": row["cell_id"][:8], "area": row["area"], "res": int(row["res"]), "categories": [int(c) for c in cell_cats[row["cell_id"]]], "totalVolume": float(hist[hist["cell_id"] == row["cell_id"]]["noisy_volume_m3"].sum())} for _, row in cell_meta.iterrows()]

    payload = {
        "categories": categories,
        "metrics": {
            "ensemble": {"mape": round(float(ens_valid["ape"].mean() * 100), 2), "mae": round(float(ens_df["abs_error"].mean()), 0), "rmse": round(float(np.sqrt(ens_df["squared_error"].mean())), 0), "folds": fold_metrics},
            "naive": {"mape": round(float(naive_valid["ape"].mean() * 100), 2), "mae": round(float(naive_df["abs_error"].mean()), 0), "rmse": round(float(np.sqrt(naive_df["squared_error"].mean())), 0), "folds": naive_fold_metrics},
        },
        "kpis": kpis,
        "domainEvents": expand_events(),
        "cells": cell_list,
        "forecastStart": 202601,
    }

    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"Done. {len(cell_list)} cells, {OUT.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    build()
