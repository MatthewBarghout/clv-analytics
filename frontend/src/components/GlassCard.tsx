import React from 'react';

/**
 * The app's two card surfaces.
 *
 * `GlassCard` is the outer container; `Panel` is the smaller inner card that was
 * previously hand-rolled as `bg-panel-raised rounded-lg p-4 border border-line` in
 * 20 places, each drifting slightly. Both read from the tokens in index.css so
 * the chrome is defined once.
 */

interface GlassCardProps {
  children: React.ReactNode;
  className?: string;
  gradient?: 'pos' | 'neg' | 'info' | 'warn';
}

const GRADIENTS: Record<NonNullable<GlassCardProps['gradient']>, string> = {
  pos: 'from-pos/10 to-pos/5',
  neg: 'from-neg/10 to-neg/5',
  info: 'from-info/10 to-info/5',
  warn: 'from-warn/10 to-warn/5',
};

export function GlassCard({ children, className = '', gradient }: GlassCardProps) {
  return (
    <div
      className={`relative rounded-xl p-6 bg-panel border border-line shadow-lg shadow-black/30 transition-colors duration-200 hover:border-line-strong ${
        gradient ? `bg-gradient-to-br ${GRADIENTS[gradient]}` : ''
      } ${className}`}
    >
      {children}
    </div>
  );
}

interface PanelProps {
  children: React.ReactNode;
  className?: string;
  /** Tighter padding for dense rows such as stat tiles. */
  dense?: boolean;
}

export function Panel({ children, className = '', dense = false }: PanelProps) {
  return (
    <div
      className={`rounded-lg bg-panel-raised border border-line ${
        dense ? 'p-3' : 'p-4'
      } ${className}`}
    >
      {children}
    </div>
  );
}
