"use client";

import { useEffect, useState, useMemo } from "react";
import { motion } from "framer-motion";
import HeroChart, { type ChartData, type CategoryData } from "./components/HeroChart";
import LeftPanel from "./components/LeftPanel";
import RightPanel from "./components/RightPanel";
import BottomPanel from "./components/BottomPanel";

interface DashboardData extends ChartData {
  cells: { id: string; short: string; area: string; res: number; categories: number[]; totalVolume: number }[];
}

export default function Home() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [activeCategory, setActiveCategory] = useState("all");
  const [activeCell, setActiveCell] = useState<string | null>(null);

  const [showActuals, setShowActuals] = useState(true);
  const [showEnsemble, setShowEnsemble] = useState(true);
  const [showNaive, setShowNaive] = useState(false);
  const [showSeasonality, setShowSeasonality] = useState(false);
  const [showDomainEvents, setShowDomainEvents] = useState(false);
  const [hoveredSeries, setHoveredSeries] = useState<string | null>(null);

  const [dateRange, setDateRange] = useState<[number, number]>([202001, 202612]);

  useEffect(() => {
    fetch("/data.json")
      .then((r) => r.json())
      .then((d) => setData(d))
      .catch(console.error);
  }, []);

  const activeDataKey = useMemo(() => {
    if (activeCell) {
      if (activeCategory !== "all" && !activeCategory.startsWith("area_") && !activeCategory.startsWith("cell_")) {
        const cellId = activeCell.replace("cell_", "");
        const cell = data?.cells.find(c => c.id === cellId);
        if (cell?.categories.includes(parseInt(activeCategory))) {
          return `${activeCell}_${activeCategory}`;
        }
      }
      return activeCell;
    }
    return activeCategory;
  }, [activeCell, activeCategory, data]);

  if (!data) {
    return (
      <div className="w-screen h-screen flex items-center justify-center bg-[#060d1b]">
        <div className="flex flex-col items-center gap-4">
          <motion.div
            className="w-12 h-12 rounded-full border-2 border-t-cyan-500 border-r-transparent border-b-transparent border-l-transparent"
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
          />
          <p className="text-cyan-300/40 text-sm font-light tracking-wider uppercase">Loading Datasets...</p>
        </div>
      </div>
    );
  }

  const activeCatData = data.categories[activeDataKey] || data.categories["all"];
  const activeKpis = data.kpis[activeDataKey] || data.kpis.all;

  const categoriesMap: Record<string, { name: string }> = {};
  for (const [key, cat] of Object.entries(data.categories)) {
    if (!key.startsWith("cell_") && !key.startsWith("area_")) {
      categoriesMap[key] = { name: (cat as CategoryData).name };
    }
  }

  return (
    <div className="dashboard-root">
      {/* ── Header ── */}
      <header className="header-area" style={{ position: "relative" }}>
        {/* Left: Brand */}
        <div className="flex items-center gap-3">
          <div className="relative w-10 h-10 rounded-xl flex items-center justify-center bg-gradient-to-br from-cyan-600/25 to-teal-600/15 border border-cyan-500/25"
            style={{ boxShadow: "0 0 18px rgba(34,211,238,0.12)" }}>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#22d3ee" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2C12 2 6 8.5 6 13a6 6 0 0 0 12 0c0-4.5-6-11-6-11z"/>
            </svg>
            {/* Subtle pulse ring */}
            <div className="absolute inset-0 rounded-xl border border-cyan-400/15 animate-ping" style={{ animationDuration: "3s" }} />
          </div>
          <div>
            <h1 className="text-[17px] font-bold tracking-tight leading-none text-white">
              Flow<span className="text-cyan-400">_IT</span>
            </h1>
            <p className="text-[9px] text-cyan-400/45 tracking-[2.5px] uppercase mt-0.5 font-semibold">
              12-Month Water Demand Forecast · Padova &amp; Trieste
            </p>
          </div>
        </div>

        {/* Center: Key metric pills */}
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1.5 px-3 py-1 rounded-full border border-emerald-500/20 bg-emerald-500/[0.06]">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-[11px] font-bold text-emerald-400 tabular-nums">{data.metrics.ensemble.mape}%</span>
            <span className="text-[9px] text-white/30 font-semibold uppercase tracking-wide">MAPE</span>
          </div>
          <div className="flex items-center gap-1.5 px-3 py-1 rounded-full border border-sky-500/15 bg-sky-500/[0.04]">
            <span className="text-[11px] font-semibold text-sky-400/80 tabular-nums">{data.cells?.length ?? 163}</span>
            <span className="text-[9px] text-white/30 font-semibold uppercase tracking-wide">cells · 5 categories</span>
          </div>
          <div className="flex items-center gap-1.5 px-3 py-1 rounded-full border border-cyan-500/12 bg-white/[0.02]">
            <span className="text-[11px] font-semibold text-white/40 tabular-nums">72 mo.</span>
            <span className="text-[9px] text-white/25 font-semibold uppercase tracking-wide">observed</span>
          </div>
        </div>

        {/* Right: Active view + period */}
        <div className="flex items-center gap-5">
          <div className="text-right">
            <p className="text-[10px] text-white/30 uppercase tracking-widest font-semibold mb-0.5">Active View</p>
            <p className="text-[15px] font-bold text-cyan-300 leading-none">{activeCatData.name}</p>
          </div>
          <div className="h-7 w-px bg-cyan-500/10" />
          <div className="text-right">
            <p className="text-[10px] text-white/30 uppercase tracking-widest font-semibold mb-0.5">Period</p>
            <p className="text-[12px] text-white/45 font-mono tabular-nums font-semibold">
              Jan {Math.floor(dateRange[0] / 100)} — Dec {Math.floor(dateRange[1] / 100)}
            </p>
          </div>
        </div>

        {/* Bottom gradient rule */}
        <div className="absolute bottom-0 left-0 right-0 h-px"
          style={{ background: "linear-gradient(90deg, transparent, rgba(34,211,238,0.08) 30%, rgba(34,211,238,0.08) 70%, transparent)" }} />
      </header>

      {/* ── Left Panel ── */}
      <aside className="sidebar-left overflow-hidden">
        <LeftPanel
          categories={categoriesMap}
          activeCategory={activeCategory}
          onCategoryChange={setActiveCategory}
          cells={data.cells}
          activeCell={activeCell}
          onCellChange={setActiveCell}
          dateRange={dateRange}
          onDateRangeChange={setDateRange}
        />
      </aside>

      {/* ── Chart ── */}
      <main className="main-content">
        <div className="absolute inset-0 p-4">
          <HeroChart
            data={data}
            activeCategory={activeDataKey}
            activeCell={activeCell}
            showActuals={showActuals}
            showEnsemble={showEnsemble}
            showNaive={showNaive}
            showSeasonality={showSeasonality}
            showDomainEvents={showDomainEvents}
            dateRange={dateRange}
            hoveredSeries={hoveredSeries}
            onHoverSeries={setHoveredSeries}
          />
        </div>
      </main>

      {/* ── Bottom Panel ── */}
      <section className="bottom-panel">
        <BottomPanel
          kpis={activeKpis}
          metrics={data.metrics.ensemble}
          model={"ensemble"}
          categoryName={activeCatData.name}
          activeCategory={activeDataKey}
          cells={data.cells}
          activeCell={activeCell}
          onCellChange={setActiveCell}
          actuals={activeCatData.actuals || []}
        />
      </section>

      {/* ── Right Panel ── */}
      <aside className="sidebar-right">
        <RightPanel
          showActuals={showActuals}
          onToggleActuals={() => setShowActuals(!showActuals)}
          showEnsemble={showEnsemble}
          onToggleEnsemble={() => setShowEnsemble(!showEnsemble)}
          showNaive={showNaive}
          onToggleNaive={() => setShowNaive(!showNaive)}
          showSeasonality={showSeasonality}
          onToggleSeasonality={() => setShowSeasonality(!showSeasonality)}
          showDomainEvents={showDomainEvents}
          onToggleDomainEvents={() => setShowDomainEvents(!showDomainEvents)}
          hoveredSeries={hoveredSeries}
          onHoverSeries={setHoveredSeries}
          metrics={data.metrics as any}
        />
      </aside>
    </div>
  );
}
