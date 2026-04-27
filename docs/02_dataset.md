# Dataset Analysis

_Last updated: 2026-03-25_

## Source

Workbook: `units_hackathon_challenge_dataset.xlsx` — 4 sheets: readme, timeseries, adjacency, rate_categories.

**Note:** the challenge statement PDF uses different column names (area_id, tariff_id, year, month, volume_m3) than the actual dataset. The names below are from the real data.

## Timeseries schema

| Column | Type | Description |
|---|---|---|
| cell_id | string (64 char hex) | HMAC-SHA256 of original H3 index |
| rate_category_id | int | Tariff macro-group (1-5) |
| period_ym | int | YYYYMM format |
| noisy_volume_m3 | float | Aggregated monthly volume after noise scaling |
| h3_resolution | int | 6 (~36 km2) or 7 (~5.16 km2) |
| system_area | string | PD (Padova) or TS (Trieste) |

## Rate categories

| ID | Name | Description |
|---|---|---|
| 1 | DOMESTIC | Residential, non-residential, promiscuous/condominium |
| 2 | COMMERCIAL | Artisanal and commercial |
| 3 | INDUSTRIAL | Industrial, high water-intensity |
| 4 | FARMING | Agricultural and zoological |
| 5 | OTHER | Public, firefighting, internal, miscellaneous |

## Adjacency

- Columns: `cell_id, neighbour_cell_id`
- Fully symmetric: every `(A, B)` edge has its `(B, A)` counterpart
- Used by the spatial features (neighbor mean, ratio, node degree)

---

## Duplicate key rows

The natural key `(cell_id, rate_category_id, period_ym)` is **not unique** in the delivered workbook for some series at H3 resolution 7. Per Data Card Section 1.5.2, Phase 1, 17 source tariff categories were consolidated into 5 macro-groups — the most plausible reading is that the duplicated keys correspond to un-reaggregated sub-components inside those macro-groups (only DOMESTIC, COMMERCIAL and OTHER are affected, INDUSTRIAL/FARMING are not).

### Project handling rule

```python
ts = ts.groupby(['cell_id', 'rate_category_id', 'period_ym'], as_index=False).agg({
    'noisy_volume_m3': 'sum',
    'h3_resolution': 'first',
    'system_area': 'first'
})
```

We use **sum** as the canonical aggregation rule, consistent with the tariff-consolidation description in the Data Card.

---

## Anonymization model

### Noise formula (Data Card Section 1.5.2, Phase 4)

```
y' = alpha_c * gamma_t * y + epsilon
```

| Factor | Distribution | Scope |
|---|---|---|
| alpha_c | Uniform(0.7, 1.3) | Fixed per (cell, rate_category) series |
| gamma_t | Uniform(0.9, 1.1) | **Global per calendar month — same for ALL cells** |
| epsilon | N(0, sigma^2), sigma = 0.03 * alpha_c * gamma_t * y | Small additive noise |

### Critical insight: gamma_t is global

All cells share the same gamma_t in any given month. Therefore:
- **Same-month cross-series comparisons are highly reliable** (gamma_t cancels in ratios)
- Spatial features at the same time t are higher quality than cross-temporal ones
- Temporal features across months carry additional gamma_{t1}/gamma_{t2} ratio noise

### What's preserved

- Seasonality (alpha_c constant, gamma_t narrow +-10%)
- Relative temporal dynamics
- Spatial correlation across adjacent cells (especially within same month)
- Cross-series relationships

### What's destroyed

- Absolute scale (alpha_c in [0.7, 1.3])
- True coordinates (HMAC-SHA256 hashing)
- Original H3 indices
- Full tariff taxonomy (17 -> 5)
- Some volume due to suppression/sampling
