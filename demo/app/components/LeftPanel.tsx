"use client";
import { useState, useMemo } from "react";

interface CellInfo {
  id: string;
  short: string;
  area: string;
  categories: number[];
  totalVolume: number;
}

interface LeftPanelProps {
  categories: Record<string, { name: string }>;
  activeCategory: string;
  onCategoryChange: (key: string) => void;
  cells: CellInfo[];
  activeCell: string | null;
  onCellChange: (cellKey: string | null) => void;
  dateRange: [number, number];
  onDateRangeChange: (range: [number, number]) => void;
}

const CAT_ICONS: Record<string, string> = {
  all: "ALL", "1": "DOM", "2": "COM", "3": "IND", "4": "AGR", "5": "OTH",
};

function fmt(n: number) {
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(0) + "K";
  return n.toFixed(0);
}

export default function LeftPanel({
  activeCategory, onCategoryChange,
  cells = [], activeCell, onCellChange,
  dateRange, onDateRangeChange
}: LeftPanelProps) {
  const [searchTerm, setSearchTerm] = useState("");

  const startYear = Math.floor(dateRange[0] / 100);
  const endYear = Math.floor(dateRange[1] / 100);

  const activeCellData = useMemo(() =>
    activeCell ? cells.find(c => `cell_${c.id}` === activeCell) : null
  , [activeCell, cells]);

  const availableCategories = useMemo(() => {
    const rootCats = [
      { id: "all", name: "All Categories", icon: CAT_ICONS["all"] },
      { id: "1", name: "Domestic", icon: CAT_ICONS["1"] },
      { id: "2", name: "Commercial", icon: CAT_ICONS["2"] },
      { id: "3", name: "Industrial", icon: CAT_ICONS["3"] },
      { id: "4", name: "Farming", icon: CAT_ICONS["4"] },
      { id: "5", name: "Other", icon: CAT_ICONS["5"] },
    ];
    if (!activeCellData) return rootCats;
    return rootCats.filter(c => c.id === "all" || activeCellData.categories.includes(parseInt(c.id)));
  }, [activeCellData]);

  const filteredCells = useMemo(() => {
    return cells.filter(c => {
      if (searchTerm && !c.id.toLowerCase().includes(searchTerm.toLowerCase()) && !c.short.toLowerCase().includes(searchTerm.toLowerCase())) return false;
      if (activeCategory.startsWith("area_")) {
        const areaCode = activeCategory.split("_")[1];
        if (c.area !== areaCode) return false;
      }
      if (!activeCell && !activeCategory.startsWith("area_") && activeCategory !== "all") {
        if (!c.categories.includes(parseInt(activeCategory))) return false;
      }
      return true;
    });
  }, [cells, searchTerm, activeCategory, activeCell]);

  const pdCells = filteredCells.filter(c => c.area === "PD");
  const tsCells = filteredCells.filter(c => c.area === "TS");

  return (
    <div className="panel-container flex flex-col h-full overflow-hidden">
      {/* ── Time Window ── */}
      <div className="p-4 border-b border-cyan-500/8">
        <p className="section-label">Time Window</p>
        <div className="flex justify-between text-[11px] font-mono mb-1.5">
          <span className="text-white/50 font-semibold">{startYear}</span>
          <span className="text-cyan-400 font-bold">{endYear}</span>
        </div>
        <input
          type="range" min="2020" max="2026" value={endYear}
          onChange={(e) => onDateRangeChange([202001, parseInt(e.target.value) * 100 + 12])}
          className="w-full"
          aria-label={`Temporal window end year: ${endYear}`}
        />
      </div>

      {/* ── Rate Categories ── */}
      <div className="p-4 border-b border-cyan-500/8">
        <div className="flex items-center justify-between mb-2">
          <p className="section-label mb-0">Rate Categories</p>
          {(activeCell || activeCategory !== 'all') && (
            <button
              onClick={() => { onCellChange(null); onCategoryChange("all"); }}
              className="text-[10px] text-cyan-400 font-bold uppercase hover:text-cyan-300 transition-colors"
            >
              Reset
            </button>
          )}
        </div>
        <div className="flex flex-col gap-1">
          {availableCategories.map(cat => (
            <button
              key={cat.id}
              onClick={() => onCategoryChange(cat.id)}
              aria-pressed={activeCategory === cat.id}
              className={`flex items-center gap-3 px-3 py-2 rounded-lg text-[12px] transition-all focus:outline-none focus:ring-1 focus:ring-cyan-500/30 ${
                activeCategory === cat.id
                  ? "bg-cyan-600/15 text-cyan-200 font-bold border border-cyan-500/25"
                  : "text-white/50 hover:bg-white/[0.03] hover:text-white/70 border border-transparent"
              }`}
            >
              <span className={`text-[9px] font-mono font-bold w-7 text-center rounded px-1 py-0.5 ${
                activeCategory === cat.id ? "bg-cyan-500/20 text-cyan-300" : "bg-white/5 text-white/30"
              }`}>{cat.icon}</span>
              <span>{cat.name}</span>
            </button>
          ))}
        </div>
      </div>

      {/* ── Territory ── */}
      <div className="p-4 border-b border-cyan-500/8">
        <p className="section-label">Territory</p>
        <div className="grid grid-cols-2 gap-2">
          {["PD", "TS"].map(a => (
            <button
              key={a}
              onClick={() => { onCellChange(null); onCategoryChange(`area_${a}`); }}
              className={`py-2 rounded-lg text-[11px] font-bold border transition-all focus:outline-none focus:ring-1 focus:ring-cyan-500/30 ${
                activeCategory === `area_${a}`
                  ? "bg-cyan-600/15 border-cyan-500/30 text-cyan-300"
                  : "bg-white/[0.02] border-white/8 text-white/40 hover:border-cyan-500/15 hover:text-white/60"
              }`}
            >
              {a === "PD" ? "Padova" : "Trieste"}
            </button>
          ))}
        </div>
      </div>

      {/* ── H3 Cell Index ── */}
      <div className="flex flex-col min-h-0 overflow-hidden">
        <div className="px-4 pt-4 pb-2">
          <p className="section-label">
            H3 Cell Index
            <span className="text-cyan-400/30 ml-1">({filteredCells.length})</span>
          </p>
          <input
            type="text" placeholder="Search by cell ID..." value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full bg-white/[0.03] border border-cyan-500/10 rounded-lg px-3 py-2 text-[12px] text-white/80 placeholder:text-white/25 outline-none focus:border-cyan-500/30 focus:ring-1 focus:ring-cyan-500/15 transition-colors"
            aria-label="Search H3 cells by ID"
          />
        </div>

        <div className="overflow-y-auto px-3 pb-4" style={{ maxHeight: "210px" }}>
          <div className="flex flex-col gap-3 py-1">
            {pdCells.length > 0 && (
              <div>
                <p className="text-[9px] font-bold text-sky-400/40 mb-1.5 ml-1 uppercase tracking-[1.5px]">Padova ({pdCells.length})</p>
                <div className="flex flex-col gap-0.5">
                  {pdCells.map(c => (
                    <CellItem key={c.id} cell={c} active={activeCell === `cell_${c.id}`} onClick={() => onCellChange(activeCell === `cell_${c.id}` ? null : `cell_${c.id}`)} />
                  ))}
                </div>
              </div>
            )}
            {tsCells.length > 0 && (
              <div>
                <p className="text-[9px] font-bold text-teal-400/40 mb-1.5 ml-1 uppercase tracking-[1.5px]">Trieste ({tsCells.length})</p>
                <div className="flex flex-col gap-0.5">
                  {tsCells.map(c => (
                    <CellItem key={c.id} cell={c} active={activeCell === `cell_${c.id}`} onClick={() => onCellChange(activeCell === `cell_${c.id}` ? null : `cell_${c.id}`)} />
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function CellItem({ cell, active, onClick }: { cell: CellInfo; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      aria-pressed={active}
      className={`flex items-center justify-between px-3 py-1.5 rounded-md text-[11px] transition-all border focus:outline-none focus:ring-1 focus:ring-cyan-500/25 ${
        active
          ? "bg-cyan-600/15 border-cyan-500/25 text-white font-semibold"
          : "bg-transparent border-transparent text-white/40 hover:bg-white/[0.03] hover:text-white/60"
      }`}
    >
      <span className="font-mono tracking-tight">{cell.short}</span>
      <span className={`font-mono text-[9px] ${active ? "text-cyan-400/60" : "text-white/20"}`}>{fmt(cell.totalVolume)}</span>
    </button>
  );
}
