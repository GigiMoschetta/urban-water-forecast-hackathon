"use client";
import { motion } from "framer-motion";

interface BackgroundProps {
  mouseX: number;
  mouseY: number;
}

export default function Background({ mouseX, mouseY }: BackgroundProps) {
  return (
    <div className="fixed inset-0 pointer-events-none overflow-hidden" aria-hidden>
      <div className="absolute inset-0 bg-[#060d1b]" />

      {/* Water-themed top glow */}
      <motion.div
        className="absolute inset-0 opacity-30"
        style={{
          background: "radial-gradient(ellipse 80% 60% at 50% 0%, rgba(8,145,178,0.12) 0%, transparent 70%)",
          x: mouseX * -20,
          y: mouseY * -20,
        }}
      />

      {/* Subtle teal accent */}
      <motion.div
        className="absolute inset-0 opacity-15"
        style={{
          background: "radial-gradient(ellipse 60% 50% at 80% 80%, rgba(20,184,166,0.08) 0%, transparent 60%)",
          x: mouseX * -40,
          y: mouseY * -40,
        }}
      />

      {/* Grid pattern */}
      <motion.div
        className="absolute inset-0 opacity-[0.02]"
        style={{
          backgroundImage: "linear-gradient(rgba(56,189,248,0.15) 1px, transparent 1px), linear-gradient(90deg, rgba(56,189,248,0.15) 1px, transparent 1px)",
          backgroundSize: "60px 60px",
          x: mouseX * -10,
          y: mouseY * -10,
        }}
      />
    </div>
  );
}
