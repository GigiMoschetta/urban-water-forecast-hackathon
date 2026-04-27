"use client";
import { motion, AnimatePresence } from "framer-motion";

export interface DomainEvent {
  id: string;
  ym: number;
  icon: string;
  title: string;
  category: string;
  impact: string;
  impactColor: string;
  insights: string[];
  area?: string;
  relevantCategories?: number[];
}

interface DomainOverlayProps {
  event: DomainEvent | null;
  x: number;
  y: number;
  onClose: () => void;
}

export default function DomainOverlay({ event, x, y, onClose }: DomainOverlayProps) {
  if (!event) return null;

  // Position card (flip sides if too close to edge)
  const flipX = x > 500;
  const cardX = flipX ? x - 310 : x + 20;
  const cardY = Math.max(10, Math.min(y - 60, 250));

  return (
    <AnimatePresence>
      <motion.div
        key={event.id}
        className="absolute z-50 pointer-events-auto"
        style={{ left: cardX, top: cardY }}
        initial={{ opacity: 0, scale: 0.9, y: 10 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.9, y: 10 }}
        transition={{ type: "spring", stiffness: 300, damping: 25 }}
      >
        <div
          className="w-[290px] p-4 cursor-default rounded-xl"
          style={{
            background: "rgba(6, 13, 27, 0.95)",
            backdropFilter: "blur(40px) saturate(1.5)",
            border: "1px solid rgba(56, 189, 248, 0.12)",
          }}
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="flex items-start gap-3 mb-3">
            <div
              className="w-10 h-10 rounded-xl flex items-center justify-center text-xl flex-shrink-0"
              style={{
                background: `${event.impactColor}15`,
                border: `1px solid ${event.impactColor}30`,
              }}
            >
              {event.icon}
            </div>
            <div className="flex-1 min-w-0">
              <h3 className="text-[13px] font-semibold text-white/90 leading-tight">
                {event.title}
              </h3>
              <div className="flex items-center gap-2 mt-1">
                <span
                  className="text-[8px] font-medium px-2 py-0.5 rounded-full uppercase tracking-wide"
                  style={{
                    background: `${event.impactColor}15`,
                    color: event.impactColor,
                    border: `1px solid ${event.impactColor}25`,
                  }}
                >
                  {event.category}
                </span>
                <span className="text-[9px] text-white/30">
                  {formatYM(event.ym)}
                </span>
              </div>
            </div>
            <button
              onClick={onClose}
              className="text-white/20 hover:text-white/60 text-[14px] leading-none p-1 -mt-1 -mr-1"
            >
              ✕
            </button>
          </div>

          {/* Impact badge */}
          <div
            className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[10px] font-medium mb-3"
            style={{
              background: `${event.impactColor}10`,
              border: `1px solid ${event.impactColor}20`,
              color: event.impactColor,
            }}
          >
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: event.impactColor }} />
            {event.impact}
          </div>

          {/* Insights */}
          <div className="flex flex-col gap-1.5">
            {event.insights.map((insight, i) => (
              <motion.div
                key={i}
                className="flex items-start gap-2"
                initial={{ opacity: 0, x: -5 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.06 }}
              >
                <span className="text-[8px] text-cyan-400/40 mt-0.5 flex-shrink-0">▸</span>
                <p className="text-[10px] text-white/55 leading-relaxed">{insight}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
}

function formatYM(ym: number): string {
  const y = Math.floor(ym / 100);
  const m = ym % 100;
  const mn = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  return `${mn[m-1]} ${y}`;
}
