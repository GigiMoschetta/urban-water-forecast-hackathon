"use client";

import React, { useMemo, useRef, useState, useCallback } from "react";
import DomainOverlay, { type DomainEvent } from "./DomainOverlay";

interface DataPoint { ym: number; v: number; }

export interface CategoryData {
  name: string;
  actuals: DataPoint[];
  ensemble: DataPoint[];
  naive: DataPoint[];
}

export interface ChartData {
  categories: Record<string, CategoryData>;
  metrics: Record<string, any>;
  kpis: Record<string, any>;
  domainEvents: DomainEvent[];
  cells?: { id: string; short: string; area: string; res: number; categories: number[]; totalVolume: number }[];
  forecastStart: number;
}

interface HeroChartProps {
  data: ChartData;
  activeCategory: string;
  activeCell?: string | null;
  showActuals: boolean;
  showEnsemble: boolean;
  showNaive: boolean;
  showSeasonality: boolean;
  showDomainEvents: boolean;
  dateRange: [number, number];
  hoveredSeries: string | null;
  onHoverSeries: (s: string | null) => void;
}

const W = 1000;
const H = 560;
const PAD = { top: 32, right: 40, bottom: 55, left: 72 };
const CW = W - PAD.left - PAD.right;
const CH = H - PAD.top - PAD.bottom;
const MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];

const COL = {
  actual: "#38bdf8",       // sky-400  — historical observed
  actualForecast: "#10b981", // emerald-500 — clearly distinct from sky blue
  ensemble: "#a78bfa",     // violet-400 — CV history
  ensembleForecast: "#c4b5fd", // violet-300 dashed — forecast
  naive: "#fb923c",        // orange-400
  grid: "rgba(56,189,248,0.05)",
  axis: "rgba(160,210,240,0.45)",
};

function fmtVol(v: number): string {
  if (v >= 1e6) return (v / 1e6).toFixed(2) + "M";
  if (v >= 1e3) return (v / 1e3).toFixed(0) + "K";
  return v.toFixed(0);
}

function ymIdx(ym: number) { return Math.floor(ym / 100) * 12 + (ym % 100) - 1; }
function ymLabel(ym: number) { const m = ym % 100; return m >= 1 && m <= 12 ? `${MONTHS[m-1]} ${Math.floor(ym/100)}` : `${Math.floor(ym/100)}`; }

export default function HeroChart({
  data, activeCategory, activeCell, showActuals, showEnsemble, showNaive,
  showSeasonality, showDomainEvents, dateRange,
  hoveredSeries, onHoverSeries,
}: HeroChartProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; items: { label: string; value: number; color: string }[]; ym: number } | null>(null);
  const [selectedEvent, setSelectedEvent] = useState<{ event: DomainEvent; x: number; y: number } | null>(null);
  const [hoveredEvent, setHoveredEvent] = useState<{ event: DomainEvent & { x: number; y: number; impactColor: string; title: string; category: string; impact: string }; x: number; y: number } | null>(null);

  const cat = data.categories[activeCategory] || data.categories["all"];
  const FSTART = data.forecastStart || 202601;

  const actuals = useMemo(() => (cat.actuals || []).filter(p => p.ym >= dateRange[0] && p.ym <= dateRange[1]), [cat, dateRange]);
  const ensemble = useMemo(() => (cat.ensemble || []).filter(p => p.ym >= dateRange[0] && p.ym <= dateRange[1]), [cat, dateRange]);
  const naive = useMemo(() => (cat.naive || []).filter(p => p.ym >= dateRange[0] && p.ym <= dateRange[1]), [cat, dateRange]);

  const { xScale, yScale, yTicks } = useMemo(() => {
    const pts = [
      ...(showActuals ? actuals : []),
      ...(showEnsemble ? ensemble : []),
      ...(showNaive ? naive : []),
    ];
    const allForX = [...actuals, ...ensemble, ...naive];
    if (pts.length === 0 || allForX.length === 0) return { xScale: () => 0, yScale: () => 0, yTicks: [] as number[] };

    const ymAll = allForX.map(p => p.ym);
    const xMin = ymIdx(Math.min(...ymAll));
    const xMax = ymIdx(Math.max(...ymAll));
    const vAll = pts.map(p => p.v);
    let vMin = Math.min(...vAll) * 0.88;
    let vMax = Math.max(...vAll) * 1.08;
    if (vMin === vMax) { vMin -= 10; vMax += 10; }

    const xs = (ym: number) => PAD.left + ((ymIdx(ym) - xMin) / Math.max(xMax - xMin, 1)) * CW;
    const ys = (v: number) => PAD.top + CH - ((v - vMin) / Math.max(vMax - vMin, 1)) * CH;

    const ticks: number[] = [];
    const step = (vMax - vMin) / 4;
    for (let i = 0; i <= 4; i++) ticks.push(vMin + step * i);

    return { xScale: xs, yScale: ys, yTicks: ticks };
  }, [actuals, ensemble, naive, showActuals, showEnsemble, showNaive]);

  const mkPath = useCallback((pts: DataPoint[]) =>
    pts.map((p, i) => `${i === 0 ? "M" : "L"}${xScale(p.ym).toFixed(1)},${yScale(p.v).toFixed(1)}`).join(" ")
  , [xScale, yScale]);

  const CV_START = 202201;
  // Split ensemble into CV history (2022-2025) + forecast (2026+); 2020-2021 = training window
  const ensembleHistPts   = useMemo(() => ensemble.filter(p => p.ym >= CV_START && p.ym < FSTART), [ensemble, FSTART]);
  const ensembleForecastPts = useMemo(() => ensemble.filter(p => p.ym >= FSTART), [ensemble, FSTART]);
  const ensembleForecastPath = useMemo(() => {
    if (ensembleForecastPts.length === 0) return "";
    const anchor = ensembleHistPts.at(-1);
    const pts = anchor ? [anchor, ...ensembleForecastPts] : ensembleForecastPts;
    return mkPath(pts);
  }, [ensembleHistPts, ensembleForecastPts, mkPath]);

  // Season configuration with rich gradients and icons
  const SEASONS: Record<string, { fill: string; glow: string; icon: string; label: string }> = {
    winter: { fill: "rgba(56,189,248,0.06)",  glow: "rgba(56,189,248,0.25)",   icon: "❄️", label: "Winter" },
    spring: { fill: "rgba(16,185,129,0.06)",  glow: "rgba(16,185,129,0.25)",   icon: "🌱", label: "Spring" },
    summer: { fill: "rgba(245,158,11,0.07)",  glow: "rgba(245,158,11,0.25)",   icon: "☀️", label: "Summer" },
    autumn: { fill: "rgba(234,88,12,0.06)",   glow: "rgba(234,88,12,0.25)",    icon: "🍂", label: "Autumn" },
  };
  const monthSeason = (m: number) => m >= 3 && m <= 5 ? "spring" : m >= 6 && m <= 8 ? "summer" : m >= 9 && m <= 11 ? "autumn" : "winter";

  const seasonBands = useMemo(() => {
    if (!showSeasonality) return [];
    const ymAll = [...actuals, ...ensemble, ...naive].map(p => p.ym);
    if (ymAll.length === 0) return [];
    const bands: { x: number; w: number; season: string; firstOfSeason: boolean; idx: number }[] = [];
    let ym = Math.min(...ymAll); const maxYm = Math.max(...ymAll);
    let prevSeason = "";
    let idx = 0;
    while (ym <= maxYm) {
      const m = ym % 100; const y = Math.floor(ym / 100);
      const season = monthSeason(m);
      const first = season !== prevSeason;
      prevSeason = season;
      const x1 = xScale(ym);
      const next = m === 12 ? (y + 1) * 100 + 1 : ym + 1;
      const x2 = xScale(next);
      if (!isNaN(x1) && !isNaN(x2)) bands.push({ x: x1, w: Math.max(x2 - x1, 0), season, firstOfSeason: first, idx: idx++ });
      ym = next;
    }
    return bands;
  }, [showSeasonality, actuals, ensemble, naive, xScale]);

  const eventPins = useMemo(() => {
    if (!showDomainEvents || !data.domainEvents) return [];
    let area: string | null = null;
    let cellCategories: number[] = [];
    if (activeCategory.startsWith("area_")) area = activeCategory.split("_")[1];
    else if (activeCategory.startsWith("cell_")) {
      const cid = activeCategory.split("_")[1];
      const cell = data.cells?.find(c => c.id === cid);
      if (cell) { area = cell.area; cellCategories = cell.categories.map(Number); }
    } else if (activeCell) {
      const cid = activeCell.replace("cell_", "");
      const cell = data.cells?.find(c => c.id === cid);
      if (cell) { area = cell.area; cellCategories = cell.categories.map(Number); }
    }
    // Determine active category number (if a specific category is selected)
    const activeCatNum = !activeCategory.startsWith("cell_") && !activeCategory.startsWith("area_") && activeCategory !== "all"
      ? parseInt(activeCategory)
      : NaN;
    const filtered = data.domainEvents
      .filter(ev => ev.ym >= dateRange[0] && ev.ym <= dateRange[1])
      .filter(ev => !ev.area || ev.area === area)
      .filter(ev => {
        // No relevantCategories field = universal event (show always)
        if (!ev.relevantCategories || ev.relevantCategories.length === 0) return true;
        // Viewing all categories
        if (activeCategory === "all" && cellCategories.length === 0) return true;
        // Viewing a specific category number
        if (!isNaN(activeCatNum)) return ev.relevantCategories.includes(activeCatNum);
        // Viewing a cell — show if any cell category matches
        if (cellCategories.length > 0) return ev.relevantCategories.some(c => cellCategories.includes(c));
        return true;
      })
      .sort((a, b) => a.ym - b.ym);
    const levels: Record<number, number> = {};
    return filtered.map(ev => {
      const x = xScale(ev.ym);
      const near = actuals.find(p => p.ym === ev.ym);
      const dataY = near ? yScale(near.v) : PAD.top + CH / 2;
      const bucket = Math.round(x / 60);
      const level = levels[bucket] || 0;
      levels[bucket] = level + 1;
      return { ...ev, x, y: PAD.top + 8 + level * 28, dataY };
    });
  }, [showDomainEvents, data.domainEvents, activeCategory, activeCell, data.cells, dateRange, actuals, xScale, yScale]);

  const handleMouseMove = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    const svg = svgRef.current;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    const mx = ((e.clientX - rect.left) / rect.width) * W;

    const allYms = new Set<number>();
    if (showActuals) actuals.forEach(p => allYms.add(p.ym));
    if (showEnsemble) ensemble.forEach(p => allYms.add(p.ym));
    if (showNaive) naive.forEach(p => allYms.add(p.ym));
    if (allYms.size === 0) { setTooltip(null); return; }

    let bestYm = 0, bestDist = Infinity;
    for (const ym of allYms) {
      const d = Math.abs(xScale(ym) - mx);
      if (d < bestDist) { bestDist = d; bestYm = ym; }
    }
    if (bestDist > 22) { setTooltip(null); return; }

    const items: { label: string; value: number; color: string }[] = [];
    const act = actuals.find(p => p.ym === bestYm);
    const ens = ensemble.find(p => p.ym === bestYm);
    const nav = naive.find(p => p.ym === bestYm);
    const isForecast = bestYm >= FSTART;
    const isPreCV = bestYm < CV_START;

    if (showActuals && act) items.push({ label: "Observed", value: act.v, color: COL.actual });
    if (showEnsemble && ens && !isPreCV) items.push({
      label: isForecast ? "Ensemble (forecast)" : "Ensemble (CV)",
      value: ens.v, color: COL.ensemble
    });
    if (showNaive && nav) items.push({
      label: isForecast ? "Naive (forecast)" : isPreCV ? "Naive (prior-year)" : "Naive (CV)",
      value: nav.v, color: COL.naive
    });

    if (items.length === 0) { setTooltip(null); return; }
    setTooltip({ x: xScale(bestYm), y: yScale(items[0].value), items, ym: bestYm });
  }, [actuals, ensemble, naive, showActuals, showEnsemble, showNaive, xScale, yScale, FSTART]);

  const actualPath = useMemo(() => actuals.length > 1 ? mkPath(actuals) : "", [actuals, mkPath]);

  // The last observed data point — this is where the forecast visually "starts"
  const lastActual = actuals.length > 0 ? actuals[actuals.length - 1] : null;

  const forecastPts = useMemo(() => {
    const fPts = ensemble.filter(p => p.ym >= FSTART);
    if (fPts.length === 0 || !lastActual) return [];
    // Anchor at the last observed point so the emerald line starts exactly there
    return [lastActual, ...fPts];
  }, [ensemble, lastActual, FSTART]);

  const yearLabels = useMemo(() =>
    Array.from(new Set([...actuals, ...ensemble, ...naive].map(p => Math.floor(p.ym / 100)))).sort()
  , [actuals, ensemble, naive]);

  // Series opacity based on hover
  const opFor = (series: string) =>
    hoveredSeries && hoveredSeries !== series ? 0.07 : 1;

  return (
    <div className="w-full h-full relative" onClick={() => setSelectedEvent(null)}>
      {showSeasonality && (
        <div className="absolute top-1 right-1 flex gap-3 bg-[#0a1628]/95 px-3 py-1.5 rounded-lg border border-cyan-500/12 z-20 backdrop-blur-sm">
          {(["spring", "summer", "autumn", "winter"] as const).map(k => {
            const s = SEASONS[k];
            return (
              <div key={k} className="flex items-center gap-1.5">
                <span className="text-xs leading-none">{s.icon}</span>
                <span className="text-[9px] text-white/50 font-semibold uppercase tracking-wide">{s.label}</span>
              </div>
            );
          })}
        </div>
      )}

      <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} className="w-full h-full" onMouseMove={handleMouseMove} onMouseLeave={() => setTooltip(null)} aria-label="Water demand chart">
        <defs>
          {/* Gradient definitions for each season */}
          <linearGradient id="grad-winter" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="rgba(56,189,248,0.12)" />
            <stop offset="100%" stopColor="rgba(56,189,248,0.02)" />
          </linearGradient>
          <linearGradient id="grad-spring" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="rgba(16,185,129,0.12)" />
            <stop offset="100%" stopColor="rgba(16,185,129,0.02)" />
          </linearGradient>
          <linearGradient id="grad-summer" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="rgba(245,158,11,0.14)" />
            <stop offset="100%" stopColor="rgba(245,158,11,0.02)" />
          </linearGradient>
          <linearGradient id="grad-autumn" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="rgba(234,88,12,0.12)" />
            <stop offset="100%" stopColor="rgba(234,88,12,0.02)" />
          </linearGradient>
          {/* Glow filter for domain event hover */}
          <filter id="domain-glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        </defs>

        {/* Season bands with gradient fills and animated entrance */}
        {seasonBands.map((b, i) => {
          const s = SEASONS[b.season];
          return (
            <g key={i} style={{ animation: `season-sweep 0.6s ease-out ${b.idx * 0.015}s both` }}>
              {/* Main gradient band */}
              <rect x={b.x} y={PAD.top} width={b.w} height={CH}
                fill={`url(#grad-${b.season})`} />
              {/* Top glow stripe */}
              <rect x={b.x} y={PAD.top} width={b.w} height={2}
                fill={s.glow} rx={1} />
              {/* Bottom season divider */}
              <line x1={b.x} y1={PAD.top} x2={b.x} y2={PAD.top + CH}
                stroke={s.glow} strokeWidth={b.firstOfSeason ? 1 : 0} strokeOpacity={0.5}
                strokeDasharray="4 6" />
              {/* Season emoji at the bottom on first month of each season */}
              {b.firstOfSeason && b.w > 5 && (
                <text x={b.x + 6} y={H - PAD.bottom - 4} fontSize={11}
                  fill="white" fillOpacity={0.3}>{s.icon}</text>
              )}
            </g>
          );
        })}

        {/* === TRAINING WINDOW BAND (2020-2021) === */}
        {(() => {
          const x0 = xScale(202001);
          const x1 = xScale(CV_START);
          if (isNaN(x0) || isNaN(x1) || x1 <= x0) return null;
          const bw = x1 - x0;
          const midX = x0 + bw / 2;
          return (
            <g>
              {/* Hatched fill */}
              <rect x={x0} y={PAD.top} width={bw} height={CH}
                fill="rgba(167,139,250,0.04)" />
              {/* Left edge */}
              <line x1={x0} y1={PAD.top} x2={x0} y2={PAD.top + CH}
                stroke="rgba(167,139,250,0.15)" strokeWidth={1} strokeDasharray="4 4" />
              {/* Right edge — transition to CV */}
              <line x1={x1} y1={PAD.top} x2={x1} y2={PAD.top + CH}
                stroke="rgba(167,139,250,0.35)" strokeWidth={1} strokeDasharray="4 4" />
              {/* Label */}
              <text x={midX} y={PAD.top + 14} textAnchor="middle"
                fontSize={8} fontWeight="700" fill="rgba(167,139,250,0.4)"
                fontFamily="var(--font-mono)" letterSpacing="1">TRAINING WINDOW</text>
            </g>
          );
        })()}

        {/* Y grid */}
        {yTicks.map((t, i) => (
          <g key={i}>
            <line x1={PAD.left} y1={yScale(t)} x2={W - PAD.right} y2={yScale(t)} stroke={COL.grid} />
            <text x={PAD.left - 8} y={yScale(t) + 4} fill={COL.axis} fontSize={10} textAnchor="end" fontFamily="var(--font-mono)">{fmtVol(t)}</text>
          </g>
        ))}

        {/* X year labels */}
        {yearLabels.map(y => {
          const x = xScale(y * 100 + 7);
          return x >= PAD.left && x <= W - PAD.right ? (
            <text key={y} x={x} y={H - PAD.bottom + 18} fill={COL.axis} fontSize={11} textAnchor="middle" fontFamily="var(--font-mono)" fontWeight="600">{y}</text>
          ) : null;
        })}

        {/* Y axis label */}
        <text x={14} y={PAD.top + CH / 2} fill="rgba(56,189,248,0.18)" fontSize={9} fontWeight="700" transform={`rotate(-90,14,${PAD.top + CH / 2})`} textAnchor="middle" fontFamily="var(--font-mono)">VOLUME (m3)</text>

        {/* === FORECAST START DEMARCATION LINE — at Dec 2025 (last observed point) === */}
        {(() => {
          // Place the line at the last actual data point, not at Jan 2026
          const fx = lastActual ? xScale(lastActual.ym) : xScale(FSTART);
          if (isNaN(fx) || fx < PAD.left || fx > W - PAD.right) return null;
          return (
            <g>
              {/* Glow */}
              <line x1={fx} y1={PAD.top - 4} x2={fx} y2={PAD.top + CH + 4}
                stroke="rgba(16,185,129,0.2)" strokeWidth={10}
                style={{ filter: "blur(5px)" }} />
              {/* Crisp dashed line */}
              <line x1={fx} y1={PAD.top - 4} x2={fx} y2={PAD.top + CH + 4}
                stroke="rgba(16,185,129,0.6)" strokeWidth={1.5} strokeDasharray="6 4" />
              {/* Badge background */}
              <rect x={fx - 50} y={PAD.top - 22} width={100} height={17} rx={4}
                fill="rgba(16,185,129,0.12)" stroke="rgba(16,185,129,0.35)" strokeWidth={1} />
              {/* Badge text */}
              <text x={fx} y={PAD.top - 10} textAnchor="middle"
                fontSize={9} fontWeight="700" fill="rgba(16,185,129,0.9)"
                fontFamily="var(--font-mono)" letterSpacing="1">
                FORECAST START
              </text>
              {/* Bottom labels */}
              <text x={fx - 8} y={H - PAD.bottom + 32} fill="rgba(56,189,248,0.35)" fontSize={9} fontWeight="700" fontFamily="var(--font-mono)" textAnchor="end">observed</text>
              <text x={fx + 8} y={H - PAD.bottom + 32} fill="rgba(16,185,129,0.4)" fontSize={9} fontWeight="700" fontFamily="var(--font-mono)">forecast</text>
            </g>
          );
        })()}

        {/* === NAIVE LINE (orange dashed) === */}
        {showNaive && naive.length > 1 && (
          <path d={mkPath(naive)} fill="none" stroke={COL.naive} strokeWidth={2.2}
            strokeDasharray="8 5" strokeLinecap="round"
            strokeOpacity={0.85 * opFor("naive")}
            style={{ transition: "stroke-opacity 200ms ease" }} />
        )}

        {/* === ENSEMBLE LINE: CV history segment (2022-2025, solid) === */}
        {showEnsemble && ensembleHistPts.length > 1 && (
          <path d={mkPath(ensembleHistPts)} fill="none" stroke={COL.ensemble} strokeWidth={2.2}
            strokeLinecap="round" strokeLinejoin="round"
            strokeOpacity={0.85 * opFor("ensemble")}
            style={{ transition: "stroke-opacity 200ms ease" }} />
        )}

        {/* === ENSEMBLE LINE: Forecast segment (dashed neon violet) === */}
        {showEnsemble && ensembleForecastPath && (
          <path d={ensembleForecastPath} fill="none" stroke={COL.ensembleForecast} strokeWidth={2.5}
            strokeDasharray="10 5" strokeLinecap="round"
            strokeOpacity={0.95 * opFor("ensemble")}
            style={{ transition: "stroke-opacity 200ms ease", filter: opFor("ensemble") > 0.5 ? "drop-shadow(0 0 3px #a78bfa66)" : "none" }} />
        )}

        {/* === ACTUALS LINE (sky blue solid thick) === */}
        {showActuals && actualPath && (
          <path d={actualPath} fill="none" stroke={COL.actual} strokeWidth={3}
            strokeLinecap="round" strokeLinejoin="round"
            strokeOpacity={opFor("actuals")}
            style={{ transition: "stroke-opacity 200ms ease" }} />
        )}

        {/* === FORECAST CONTINUATION (cyan, from last actual) === */}
        {showActuals && forecastPts.length > 1 && (
          <path d={mkPath(forecastPts)} fill="none" stroke={COL.actualForecast} strokeWidth={3}
            strokeLinecap="round" strokeLinejoin="round"
            strokeOpacity={opFor("actuals")}
            style={{ transition: "stroke-opacity 200ms ease" }} />
        )}

        {/* Junction dot — at Dec 2025, the exact boundary between observed and forecast */}
        {showActuals && lastActual && (
          <circle cx={xScale(lastActual.ym)} cy={yScale(lastActual.v)}
            r={5} fill={COL.actualForecast} stroke="#0a1628" strokeWidth={2.5}
            opacity={opFor("actuals")} style={{ transition: "opacity 200ms ease" }} />
        )}

        {/* === INVISIBLE HIT STRIPS for hover-to-highlight === */}
        {showActuals && actualPath && (
          <path d={actualPath} fill="none" stroke="transparent" strokeWidth={20}
            style={{ cursor: "pointer" }}
            onMouseEnter={() => onHoverSeries("actuals")}
            onMouseLeave={() => onHoverSeries(null)} />
        )}
        {showEnsemble && ensembleHistPts.length > 1 && (
          <path d={mkPath(ensembleHistPts)} fill="none" stroke="transparent" strokeWidth={20}
            style={{ cursor: "pointer" }}
            onMouseEnter={() => onHoverSeries("ensemble")}
            onMouseLeave={() => onHoverSeries(null)} />
        )}
        {showEnsemble && ensembleForecastPath && (
          <path d={ensembleForecastPath} fill="none" stroke="transparent" strokeWidth={20}
            style={{ cursor: "pointer" }}
            onMouseEnter={() => onHoverSeries("ensemble")}
            onMouseLeave={() => onHoverSeries(null)} />
        )}
        {showNaive && naive.length > 1 && (
          <path d={mkPath(naive)} fill="none" stroke="transparent" strokeWidth={20}
            style={{ cursor: "pointer" }}
            onMouseEnter={() => onHoverSeries("naive")}
            onMouseLeave={() => onHoverSeries(null)} />
        )}

        {/* Domain events with animated pins */}
        {showDomainEvents && eventPins.map((pin, pinIdx) => (
          <g key={pin.id} className="cursor-pointer"
            style={{ animation: `domain-enter 0.55s cubic-bezier(0.34,1.56,0.64,1) ${pinIdx * 0.07}s both, domain-float 4s ease-in-out ${0.55 + pinIdx * 0.3}s infinite` }}
            onClick={(e) => { e.stopPropagation(); setSelectedEvent({ event: pin, x: pin.x, y: pin.y }); }}
            onMouseEnter={() => setHoveredEvent({ event: pin as any, x: pin.x, y: pin.y })}
            onMouseLeave={() => setHoveredEvent(null)}
          >
            {/* Animated connector line with shimmer */}
            <line x1={pin.x} y1={pin.y + 12} x2={pin.x} y2={pin.dataY}
              stroke={pin.impactColor} strokeOpacity={0.15} strokeWidth={1}
              strokeDasharray="3 4"
              style={{ animation: `connector-shimmer 2s linear ${pinIdx * 0.2}s infinite` }} />
            {/* Pulse ring (animated glow) */}
            <circle cx={pin.x} cy={pin.y} r={12}
              fill="none" stroke={pin.impactColor} strokeWidth={1.5} strokeOpacity={0.4}
              style={{ animation: `domain-pulse 2.5s ease-out ${pinIdx * 0.4}s infinite` }} />
            {/* Outer glow circle */}
            <circle cx={pin.x} cy={pin.y} r={14}
              fill="none" stroke={pin.impactColor} strokeWidth={0.5} strokeOpacity={0.1} />
            {/* Main circle background */}
            <circle cx={pin.x} cy={pin.y} r={12}
              fill="rgba(6,13,27,0.92)" stroke={pin.impactColor} strokeWidth={1.5}
              style={{ filter: `drop-shadow(0 0 4px ${pin.impactColor}40)` }} />
            {/* Icon */}
            <text x={pin.x} y={pin.y} textAnchor="middle" dominantBaseline="central"
              fontSize={13} fill={pin.impactColor} fontWeight="700" role="img"
              style={{ filter: `drop-shadow(0 0 2px ${pin.impactColor}60)` }}>{pin.icon}</text>
          </g>
        ))}

        {/* === EVENT HOVER TOOLTIP === */}
        {hoveredEvent && (() => {
          const { event: ev, x, y } = hoveredEvent;
          const flip = x > W * 0.6;
          const cardW = 182;
          const cx = flip ? x - cardW - 14 : x + 14;
          const cy = Math.max(PAD.top + 4, y - 8);
          const mo = MONTHS[((ev as any).ym % 100) - 1];
          const yr = Math.floor((ev as any).ym / 100);
          const impactColor = (ev as any).impactColor || "#a78bfa";
          const evTitle = (ev as any).title || "";
          const evCategory = (ev as any).category || "";
          const evImpact = (ev as any).impact || "";
          return (
            <g style={{ pointerEvents: "none" }} filter="url(#domain-glow)">
              {/* Connector line */}
              <line x1={x} y1={y} x2={flip ? cx + cardW : cx} y2={cy + 16}
                stroke={impactColor} strokeOpacity={0.5} strokeWidth={1} />
              {/* Card shadow */}
              <rect x={cx + 2} y={cy + 2} width={cardW} height={58} rx={10}
                fill="rgba(0,0,0,0.3)" />
              {/* Card background */}
              <rect x={cx} y={cy} width={cardW} height={58} rx={10}
                fill="rgba(6,13,27,0.97)" stroke={impactColor} strokeWidth={1.5} strokeOpacity={0.7} />
              {/* Left accent bar */}
              <rect x={cx} y={cy + 4} width={3} height={50} rx={2}
                fill={impactColor} fillOpacity={0.9} />
              {/* Date + category header */}
              <text x={cx + 14} y={cy + 16} fontSize={9} fontWeight="700"
                fill="rgba(255,255,255,0.45)" fontFamily="var(--font-mono)" letterSpacing="0.5">
                {mo} {yr} · {evCategory}
              </text>
              {/* Event title */}
              <text x={cx + 14} y={cy + 34} fontSize={12} fontWeight="700"
                fill="white" fontFamily="var(--font-sans)">
                {evTitle}
              </text>
              {/* Impact label */}
              <text x={cx + 14} y={cy + 50} fontSize={9} fontWeight="600"
                fill={impactColor}>
                ⚡ {evImpact}
              </text>
            </g>
          );
        })()}

        {/* === MULTI-LINE TOOLTIP === */}
        {tooltip && (
          <g style={{ pointerEvents: "none" }}>
            <line x1={tooltip.x} y1={PAD.top} x2={tooltip.x} y2={PAD.top + CH} stroke="rgba(255,255,255,0.08)" />
            {tooltip.items.map((it, i) => (
              <circle key={i} cx={tooltip.x} cy={yScale(it.value)} r={5} fill={it.color} stroke="#0a1628" strokeWidth={2.5} />
            ))}
            {(() => {
              const cardH = 24 + tooltip.items.length * 20;
              const cardW = 195;
              const cx = tooltip.x > W / 2 ? tooltip.x - cardW - 14 : tooltip.x + 14;
              const cy = Math.max(PAD.top, Math.min(PAD.top + CH / 4, PAD.top + CH - cardH));
              return (
                <g>
                  <rect x={cx} y={cy} width={cardW} height={cardH} rx={8} fill="rgba(6,13,27,0.95)" stroke="rgba(56,189,248,0.1)" />
                  <text x={cx + cardW / 2} y={cy + 15} textAnchor="middle" fontSize={11} fill="rgba(160,210,240,0.6)" fontWeight="600" fontFamily="var(--font-mono)">{ymLabel(tooltip.ym)}</text>
                  {tooltip.items.map((it, i) => (
                    <g key={i}>
                      <rect x={cx + 10} y={cy + 24 + i * 20} width={8} height={8} rx={2} fill={it.color} />
                      <text x={cx + 24} y={cy + 32 + i * 20} fontSize={11} fill="rgba(255,255,255,0.6)">{it.label}</text>
                      <text x={cx + cardW - 10} y={cy + 32 + i * 20} fontSize={12} fill="white" fontWeight="700" textAnchor="end" fontFamily="var(--font-mono)">{fmtVol(it.value)}</text>
                    </g>
                  ))}
                </g>
              );
            })()}
          </g>
        )}
      </svg>

      <DomainOverlay event={selectedEvent?.event || null} x={selectedEvent?.x || 0} y={selectedEvent?.y || 0} onClose={() => setSelectedEvent(null)} />
    </div>
  );
}
