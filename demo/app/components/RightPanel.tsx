"use client";
import { motion } from "framer-motion";

interface FoldMetric {
  fold: string;
  mape: number;
  mae: number;
  rmse: number;
}

interface ModelMetrics {
  mape: number;
  mae: number;
  rmse: number;
  folds?: FoldMetric[];
}

interface Metrics {
  ensemble: ModelMetrics;
  naive: ModelMetrics;
}

interface RightPanelProps {
  showActuals: boolean;
  onToggleActuals: () => void;
  showEnsemble: boolean;
  onToggleEnsemble: () => void;
  showNaive: boolean;
  onToggleNaive: () => void;
  showSeasonality: boolean;
  onToggleSeasonality: () => void;
  showDomainEvents: boolean;
  onToggleDomainEvents: () => void;
  hoveredSeries: string | null;
  onHoverSeries: (s: string | null) => void;
  metrics: Metrics;
}

function Toggle({ on, onToggle, label, sub, color = "bg-cyan-600" }: { on: boolean; onToggle: () => void; label: string; sub?: string; color?: string }) {
  return (
    <button
      onClick={onToggle}
      role="switch"
      aria-checked={on}
      aria-label={`${label}: ${on ? "enabled" : "disabled"}`}
      className={`group flex flex-col w-full p-3 rounded-xl border transition-all focus:outline-none focus:ring-1 focus:ring-cyan-500/30 ${
        on
          ? "bg-white/[0.04] border-cyan-500/20 shadow-[0_0_12px_rgba(8,145,178,0.06)]"
          : "bg-white/[0.01] border-white/5 hover:bg-white/[0.03] hover:border-white/8"
      }`}
    >
      <div className="flex items-center justify-between w-full">
        <span className={`text-[13px] font-semibold ${on ? "text-white/90" : "text-white/35"}`}>{label}</span>
        <div className={`w-8 h-[18px] rounded-full transition-all relative ${on ? color : "bg-white/10"}`}>
          <motion.div
            className="absolute top-[3px] w-3 h-3 rounded-full bg-white shadow-sm"
            animate={{ left: on ? 17 : 3 }}
            transition={{ type: "spring", stiffness: 500, damping: 30 }}
          />
        </div>
      </div>
      {sub && <span className="text-[10px] text-white/55 mt-0.5 text-left">{sub}</span>}
    </button>
  );
}

export default function RightPanel({
  showActuals, onToggleActuals,
  showEnsemble, onToggleEnsemble,
  showNaive, onToggleNaive,
  showSeasonality, onToggleSeasonality,
  showDomainEvents, onToggleDomainEvents,
  hoveredSeries, onHoverSeries,
  metrics
}: RightPanelProps) {
  const mapeImprovement = ((1 - metrics.ensemble.mape / metrics.naive.mape) * 100).toFixed(0);

  return (
    <div className="panel-container flex flex-col h-full p-4 gap-4 overflow-y-auto">
      {/* ── Series Toggles ── */}
      <div>
        <p className="section-label">Data Series</p>
        <div className="flex flex-col gap-1.5">
          <Toggle on={showActuals} onToggle={onToggleActuals} label="Observed Volume" sub="Historical metered data" color="bg-sky-500" />
          <Toggle on={showEnsemble} onToggle={onToggleEnsemble} label="Ensemble Model" sub="3-model champion (v1 + v9o + naive)" color="bg-violet-500" />
          <Toggle on={showNaive} onToggle={onToggleNaive} label="Naive Baseline" sub="Same-month prior year" color="bg-amber-500" />
        </div>
      </div>

      {/* ── Performance Metrics ── */}
      <div className="bg-white/[0.02] rounded-xl p-4 border border-cyan-500/8">
        <p className="section-label text-center">Model Performance</p>

        <div className="grid grid-cols-3 gap-2 text-center mb-3">
          <div className="bg-white/[0.02] p-2 rounded-lg">
            <p className="text-[9px] text-white/55 uppercase mb-0.5 font-semibold">MAPE</p>
            <p className="text-lg font-bold text-emerald-400 tabular-nums">{metrics.ensemble.mape}%</p>
            <p className="text-[9px] text-white/30 tabular-nums line-through">{metrics.naive.mape}%</p>
          </div>
          <div className="bg-white/[0.02] p-2 rounded-lg">
            <p className="text-[9px] text-white/55 uppercase mb-0.5 font-semibold">MAE</p>
            <p className="text-[15px] font-bold text-white/70 tabular-nums">{metrics.ensemble.mae}</p>
            <p className="text-[9px] text-white/30 tabular-nums line-through">{metrics.naive.mae}</p>
          </div>
          <div className="bg-white/[0.02] p-2 rounded-lg">
            <p className="text-[9px] text-white/55 uppercase mb-0.5 font-semibold">RMSE</p>
            <p className="text-[15px] font-bold text-white/70 tabular-nums">{metrics.ensemble.rmse}</p>
            <p className="text-[9px] text-white/30 tabular-nums line-through">{metrics.naive.rmse}</p>
          </div>
        </div>

        <div className="bg-emerald-500/8 border border-emerald-500/15 rounded-lg px-3 py-1.5 text-center mb-3">
          <p className="text-[10px] text-emerald-400 font-bold">
            {mapeImprovement}% MAPE improvement over naive baseline
          </p>
        </div>

        {metrics.ensemble.folds && (
          <div>
            <p className="text-[9px] text-white/40 uppercase tracking-[1.5px] font-bold mb-2">Per-Fold MAPE</p>
            <div className="flex flex-col gap-1.5">
              {metrics.ensemble.folds.map((f) => {
                const naiveFold = metrics.naive.folds?.find(nf => nf.fold === f.fold);
                const foldLabel = f.fold.replace("fold_", "F");
                const yearMap: Record<string, string> = { "F1": "2022", "F2": "2023", "F3": "2024", "F4": "2025" };
                return (
                  <div key={f.fold} className="flex items-center gap-2">
                    <span className="text-[10px] text-white/55 font-mono font-semibold w-5">{foldLabel}</span>
                    <span className="text-[9px] text-white/30 w-7 font-mono">{yearMap[foldLabel] || ""}</span>
                    <div className="flex-1 h-[10px] bg-white/[0.03] rounded-full overflow-hidden">
                      <div className="h-full bg-emerald-500/25 rounded-full transition-all" style={{ width: `${Math.min(f.mape / 35 * 100, 100)}%` }} />
                    </div>
                    <span className="text-[10px] text-emerald-400 font-bold tabular-nums w-12 text-right">{f.mape}%</span>
                    {naiveFold && <span className="text-[9px] text-white/25 tabular-nums w-12 text-right line-through">{naiveFold.mape}%</span>}
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* ── Overlays ── */}
      <div>
        <p className="section-label">Analytic Overlays</p>
        <div className="flex flex-col gap-1.5">
          <Toggle on={showSeasonality} onToggle={onToggleSeasonality} label="Seasonality" sub="Seasonal background bands" color="bg-cyan-600" />
          <Toggle on={showDomainEvents} onToggle={onToggleDomainEvents} label="Domain Events" sub="COVID, drought, policy changes" color="bg-cyan-600" />
        </div>
      </div>

      {/* ── Legend ── */}
      <div className="mt-auto pt-4 border-t border-cyan-500/8">
        <div className="flex flex-col gap-2.5">
          <div
            className="flex items-center gap-3 cursor-pointer group"
            onMouseEnter={() => onHoverSeries("actuals")}
            onMouseLeave={() => onHoverSeries(null)}
          >
            <div className={`w-7 h-[3px] bg-sky-400 rounded-full transition-opacity duration-200 ${hoveredSeries && hoveredSeries !== "actuals" ? "opacity-10" : "opacity-100"}`} />
            <span className={`text-[11px] font-semibold transition-opacity duration-200 ${hoveredSeries && hoveredSeries !== "actuals" ? "text-white/10" : "text-white/55"}`}>Observed Volume</span>
          </div>
          <div
            className="flex items-center gap-3 cursor-pointer group"
            onMouseEnter={() => onHoverSeries("ensemble")}
            onMouseLeave={() => onHoverSeries(null)}
          >
            <div className={`w-7 h-[3px] bg-violet-400 rounded-full transition-opacity duration-200 ${hoveredSeries && hoveredSeries !== "ensemble" ? "opacity-10" : "opacity-100"}`} />
            <span className={`text-[11px] font-semibold transition-opacity duration-200 ${hoveredSeries && hoveredSeries !== "ensemble" ? "text-white/10" : "text-white/55"}`}>Ensemble Forecast</span>
          </div>
          <div
            className="flex items-center gap-3 cursor-pointer group"
            onMouseEnter={() => onHoverSeries("naive")}
            onMouseLeave={() => onHoverSeries(null)}
          >
            <div className={`w-7 h-[3px] border-b-2 border-dashed border-amber-500/60 transition-opacity duration-200 ${hoveredSeries && hoveredSeries !== "naive" ? "opacity-10" : "opacity-100"}`} />
            <span className={`text-[11px] font-semibold transition-opacity duration-200 ${hoveredSeries && hoveredSeries !== "naive" ? "text-white/10" : "text-white/55"}`}>Naive Baseline</span>
          </div>
        </div>
      </div>
    </div>
  );
}
