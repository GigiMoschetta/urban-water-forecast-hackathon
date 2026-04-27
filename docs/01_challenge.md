# Challenge Requirements

_Last updated: 2026-03-29_

## Challenge title

**From Bills to Plans: Predicting 12 Months of Water Demand at Urban Scale**
- Company: AcegasApsAmga
- Domain: Forecasting & Data Analytics
- Hackathon: March 25 — April 1, 2026

## Business problem

AcegasApsAmga manages water distribution across Padova (PD) and Trieste (TS) metropolitan areas. Monthly demand varies by geography, season, tariff/user category, and local context. Forecasts support capacity planning, infrastructure planning, anomaly detection, and revenue estimation.

## Required task

Build a **multivariate time series model** that forecasts **12 months ahead** for each (cell_id, rate_category_id) pair using the anonymized billing dataset.

## Required evaluation

- **Walk-forward cross-validation**, 4 folds
- **Minimum training window**: 24 months
- **Forecast horizon**: 12 months
- **Ranking metric**: mean MAPE across 4 folds
- Also report: MAE, RMSE
- For zero-target observations, the challenge material allows either excluding the sample from MAPE or using sMAPE; our current project convention is to exclude zero-target samples from MAPE and report that rule explicitly

## Required baseline comparison

**Seasonal naive**: predict month t+h with the value from the same month in the prior year. A model is considered meaningful only if it clearly outperforms this.

## Mandatory deliverables (due April 1, 13:00)

| # | Deliverable | Format | Notes |
|---|---|---|---|
| 1 | Presentation | PDF/PPTX, max 10 slides, English | Mandatory |
| 2 | Code repository | GitHub/GitLab, public link | Mandatory. README with reproduction instructions |
| 3 | Live demo | Shown live during pitch | Mandatory. Must run live on the day |
| 4 | Technical report | PDF or .ipynb | Optional |

Late submission = scoring penalty. After 13:30 = rejected.

## Mandatory repository requirements

- **MIT License** file (Art. 14)
- **AI tool disclosure** in README (Art. 13): which tools, how they contributed
- **Compute resource disclosure** in README (Art. 13): anything beyond free tier

## Dataset handling constraints

From the challenge statement:
- the dataset is confidential and must not be shared outside the hackathon
- derived outputs may be shared only if they do not enable reconstruction of the raw dataset
- local copies, extracts, and caches must be deleted within 7 days after the event ends

## Packaging implication for this repo

Operational interpretation for submission prep:
- the mandatory public repository requirement applies to the code/documentation deliverable, not as blanket permission to publish confidential challenge inputs
- for any public mirror, exclude `data/original/`, organizer PDFs/handouts, and copied excerpts unless redistribution is explicitly allowed
- review derived artifacts such as `data/processed/` and `outputs/` case by case before publication

## Presentation format

- **5 min**: pitch + demo (demo max 3 min per Art. 8)
- **5 min**: Q&A with jury
- "Five slides or fewer is often better than ten"
- Jury asks about: key choices, baseline comparison, what you'd do differently

## Optional / bonus elements

- Exploit spatial correlation via adjacency
- Dashboard of forecast patterns across the territorial grid
- Dashboard surfacing anomalies or trend shifts
