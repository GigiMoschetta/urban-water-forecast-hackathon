# Flow_IT Dashboard — Predictive Water Intelligence

This is a Next.js application for visualizing water consumption forecasts for Padova and Trieste.

## 🚀 Getting Started

### 1. Data Preparation
Before running the dashboard, ensure the data is generated:

```bash
python3 scripts/build_data.py
```

This script processes the raw timeseries and model predictions from the project root and outputs `public/data.json`.

### 2. Run Development Server
Install dependencies and start the app:

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to see the interactive dashboard.

## 📊 Features

- **Interactive Hero Chart**: Visualize historical data (2020-2025) and forecasts (2026).
- **Model Comparison**: Switch between Ensemble 3-Model and Seasonal Naïve baselines.
- **Drill-down**: View metrics by Category (Domestic, Industrial, etc.), Area (Padova/Trieste), or individual H3 Cells.
- **Overlays**: Toggle Seasonality bands, Domain Events (Lockdowns, Droughts), and Confidence Bands.
- **Parallax Background**: Immersive 3D control room aesthetic with mouse-reactive parallax.

## 🛠️ Tech Stack

- **Framework**: Next.js 15 (App Router)
- **Styling**: Tailwind CSS 4
- **Animations**: Framer Motion
- **Icons**: Lucide React
- **Data Visualization**: Custom SVG with D3-like scaling logic.
