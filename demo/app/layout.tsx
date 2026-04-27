import type { Metadata } from "next";
import React from "react";

export const metadata: Metadata = {
  title: "💧 Flow_IT 🇮🇹 — Predictive Water Intelligence",
  description:
    "Flow_IT: AI-powered water demand forecasting for AcegasApsAmga — Padova & Trieste. 3-model ensemble achieving 19.91% MAPE.",
};

import "./globals.css";

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700&family=Playfair+Display:ital,wght@0,400;0,700;1,400&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="antialiased">{children}</body>
    </html>
  );
}
