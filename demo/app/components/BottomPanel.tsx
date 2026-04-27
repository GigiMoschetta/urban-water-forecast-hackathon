"use client";
import { useMemo } from "react";

interface KPIs {
  totalHistorical: number;
  totalForecast: number;
  peakMonth: string;
  troughMonth: string;
  yoyGrowth: number;
}

interface CellInfo {
  id: string;
  short: string;
  area: string;
  res: number;
  categories: number[];
  totalVolume: number;
}

interface DataPoint { ym: number; v: number; }

interface BottomPanelProps {
  kpis: KPIs;
  metrics: { mape: number; mae: number; rmse: number };
  model: string;
  categoryName: string;
  activeCategory: string;
  cells: CellInfo[];
  activeCell: string | null;
  onCellChange: (cellKey: string | null) => void;
  actuals: DataPoint[];
}

function fmt(n: number) {
  if (n === undefined || n === null) return "0 m\u00B3";
  if (n >= 1e6) return (n / 1e6).toFixed(2) + "M m\u00B3";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "K m\u00B3";
  return n.toFixed(0) + " m\u00B3";
}

function Sparkline({ data, color = "#22d3ee", height = 22, width = 88 }: {
  data: number[]; color?: string; height?: number; width?: number;
}) {
  if (data.length < 2) return <div style={{ height, width }} />;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const pts = data.map((v, i) => {
    const x = (i / (data.length - 1)) * width;
    const y = height - ((v - min) / range) * (height - 2) - 1;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} aria-hidden="true" className="opacity-55">
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.5"
        strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function HexSVG({ size = 68 }: { size?: number }) {
  const pts = Array.from({ length: 6 }, (_, i) => {
    const a = (Math.PI / 180) * (60 * i - 30);
    const r = size / 2 - 2;
    return `${(size / 2 + r * Math.cos(a)).toFixed(1)},${(size / 2 + r * Math.sin(a)).toFixed(1)}`;
  }).join(" ");
  const innerPts = Array.from({ length: 6 }, (_, i) => {
    const a = (Math.PI / 180) * (60 * i - 30);
    const r = (size / 2 - 2) * 0.55;
    return `${(size / 2 + r * Math.cos(a)).toFixed(1)},${(size / 2 + r * Math.sin(a)).toFixed(1)}`;
  }).join(" ");
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}
      className="drop-shadow-[0_0_10px_rgba(34,211,238,0.3)]" aria-hidden="true">
      <defs>
        <radialGradient id="hexGrad" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.12" />
          <stop offset="100%" stopColor="#22d3ee" stopOpacity="0.02" />
        </radialGradient>
      </defs>
      <polygon points={pts} fill="url(#hexGrad)" stroke="#22d3ee" strokeWidth="1.5" strokeOpacity="0.65" />
      <polygon points={innerPts} fill="none" stroke="#22d3ee" strokeWidth="0.5" strokeOpacity="0.22" />
    </svg>
  );
}

function KPICard({ icon, label, value, sub, color, sparkData, sparkColor }: {
  icon: React.ReactNode; label: string; value: string;
  sub: string; color: string; sparkData: number[]; sparkColor: string;
}) {
  return (
    <div className="bg-white/[0.025] border border-cyan-500/10 rounded-xl p-3 flex flex-col gap-1 hover:border-cyan-500/22 transition-colors">
      <div className="flex items-center gap-2">
        <div className="text-white/30 flex-shrink-0">{icon}</div>
        <p className="text-[9px] text-cyan-400/50 uppercase tracking-[1.5px] font-bold truncate">{label}</p>
      </div>
      <p className={`text-[17px] font-bold tabular-nums tracking-tight leading-none ${color}`}>{value}</p>
      <Sparkline data={sparkData} color={sparkColor} width={90} height={20} />
      <p className="text-[9px] text-white/40">{sub}</p>
    </div>
  );
}

// Minimal SVG icons
const HistoryIcon = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
  </svg>
);
const ForecastIcon = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
  </svg>
);
const PeakIcon = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/>
  </svg>
);
const TrendIcon = ({ positive }: { positive: boolean }) => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    {positive
      ? <><line x1="12" y1="20" x2="12" y2="4"/><polyline points="6 10 12 4 18 10"/></>
      : <><line x1="12" y1="4" x2="12" y2="20"/><polyline points="18 14 12 20 6 14"/></>}
  </svg>
);

export default function BottomPanel({
  kpis, categoryName, activeCategory,
  cells = [], activeCell, onCellChange, actuals = []
}: BottomPanelProps) {
  const k = kpis || { totalHistorical: 0, totalForecast: 0, peakMonth: "-", troughMonth: "-", yoyGrowth: 0 };

  const activeCellIndex = useMemo(
    () => activeCell ? cells.findIndex(c => `cell_${c.id}` === activeCell) : -1,
    [cells, activeCell]
  );
  const activeCellData = useMemo(
    () => activeCell ? cells.find(c => `cell_${c.id}` === activeCell) : null,
    [cells, activeCell]
  );

  const prevCell = () => {
    if (cells.length === 0) return;
    const nextIdx = activeCellIndex <= 0 ? cells.length - 1 : activeCellIndex - 1;
    onCellChange(`cell_${cells[nextIdx].id}`);
  };
  const nextCell = () => {
    if (cells.length === 0) return;
    const nextIdx = activeCellIndex >= cells.length - 1 ? 0 : activeCellIndex + 1;
    onCellChange(`cell_${cells[nextIdx].id}`);
  };

  // Sparkline data from actuals
  const last24 = actuals.slice(-24).map(p => p.v);
  const last12 = actuals.slice(-12).map(p => p.v);
  const yoySparkData = actuals.slice(-24).map((p, i, arr) =>
    i === 0 ? 0 : ((p.v - arr[i - 1].v) / (arr[i - 1].v || 1)) * 100
  );

  return (
    <div className="panel-container items-stretch p-4 gap-4 h-full" style={{ flexDirection: "row" }}>

      {/* ── LEFT: Hex Selector ── */}
      <div className="flex flex-col items-center justify-center gap-2 min-w-[180px] border-r border-cyan-500/8 pr-4">
        <p className="text-[9px] text-cyan-400/40 uppercase tracking-[1.5px] font-bold">H3 Navigator</p>

        <div className="flex items-center gap-3">
          <button
            onClick={prevCell}
            className="w-7 h-7 rounded-lg bg-white/[0.04] border border-cyan-500/15 text-cyan-400/60 hover:border-cyan-500/40 hover:text-cyan-300 hover:bg-white/[0.08] transition-all flex items-center justify-center font-bold text-base leading-none"
            aria-label="Previous cell"
          >&#8249;</button>

          <HexSVG size={52} />

          <button
            onClick={nextCell}
            className="w-7 h-7 rounded-lg bg-white/[0.04] border border-cyan-500/15 text-cyan-400/60 hover:border-cyan-500/40 hover:text-cyan-300 hover:bg-white/[0.08] transition-all flex items-center justify-center font-bold text-base leading-none"
            aria-label="Next cell"
          >&#8250;</button>
        </div>

        <div className="text-center">
          <p className="text-[9px] text-white/30 font-mono mb-0.5">
            {activeCellIndex >= 0 ? `${activeCellIndex + 1} / ${cells.length}` : `— / ${cells.length}`}
          </p>
          <p className="text-[13px] font-mono font-bold text-cyan-300 tracking-tight">
            {activeCellData?.short ?? "SELECT CELL"}
          </p>
        </div>

        {activeCellData ? (
          <div className="grid grid-cols-3 gap-2 text-center w-full">
            <div>
              <p className="text-[8px] text-white/25 uppercase font-bold">Res</p>
              <p className="text-[11px] font-bold text-white/65">H3-{activeCellData.res}</p>
            </div>
            <div>
              <p className="text-[8px] text-white/25 uppercase font-bold">Area</p>
              <p className="text-[11px] font-bold text-sky-400/80">{activeCellData.area}</p>
            </div>
            <div>
              <p className="text-[8px] text-white/25 uppercase font-bold">Cats</p>
              <p className="text-[11px] font-bold text-white/65">{activeCellData.categories.length}</p>
            </div>
          </div>
        ) : (
          <p className="text-[9px] text-white/20 text-center">Use ‹ › to browse cells</p>
        )}
      </div>

      {/* ── RIGHT: KPI Cards ── */}
      <div className="flex-1 grid grid-cols-4 gap-3 items-center">
        <KPICard
          icon={<HistoryIcon />}
          label="Total Historical"
          value={fmt(k.totalHistorical)}
          sub="6-year aggregated volume"
          color="text-sky-300"
          sparkData={last24}
          sparkColor="#38bdf8"
        />
        <KPICard
          icon={<ForecastIcon />}
          label="2026 Forecast"
          value={fmt(k.totalForecast)}
          sub="12-month ensemble projection"
          color="text-emerald-400"
          sparkData={last12}
          sparkColor="#10b981"
        />
        <KPICard
          icon={<PeakIcon />}
          label="Peak Demand"
          value={k.peakMonth || "—"}
          sub="Historical maximum month"
          color="text-amber-400"
          sparkData={last12}
          sparkColor="#f59e0b"
        />
        <KPICard
          icon={<TrendIcon positive={k.yoyGrowth > 0} />}
          label="YoY Growth"
          value={`${k.yoyGrowth > 0 ? "+" : ""}${k.yoyGrowth}%`}
          sub="2025 vs 2024"
          color={k.yoyGrowth < 0 ? "text-emerald-400" : k.yoyGrowth > 0 ? "text-amber-400" : "text-white/65"}
          sparkData={yoySparkData}
          sparkColor={k.yoyGrowth < 0 ? "#10b981" : "#f59e0b"}
        />
      </div>
    </div>
  );
}
