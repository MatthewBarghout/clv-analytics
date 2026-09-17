/**
 * One Recharts theme for every chart in the app.
 *
 * Axis, grid and tooltip styling was previously copy-pasted into each of the four
 * charting components, with the hex values drifting between them. These constants
 * mirror the design tokens in index.css — keep the two in step.
 */

export const CHART_COLORS = {
  grid: '#1d212a',
  axis: '#6a7280',
  tick: '#98a0ac',
  pos: '#4ade80',
  neg: '#f87171',
  info: '#60a5fa',
  warn: '#fbbf24',
  neutral: '#98a0ac',
} as const;

/** Series colours, in order, for charts that plot several lines at once. */
export const SERIES_COLORS = [
  CHART_COLORS.info,
  CHART_COLORS.pos,
  CHART_COLORS.warn,
  CHART_COLORS.neg,
] as const;

export const gridProps = {
  strokeDasharray: '3 3',
  stroke: CHART_COLORS.grid,
} as const;

export const axisProps = {
  stroke: CHART_COLORS.axis,
  tick: { fill: CHART_COLORS.tick, fontSize: 11 },
  tickLine: false,
} as const;

export const tooltipStyle = {
  backgroundColor: '#101217',
  border: '1px solid #2b313d',
  borderRadius: '8px',
  color: '#e9ebef',
  fontSize: '12px',
  fontVariantNumeric: 'tabular-nums',
} as const;

export const legendProps = {
  wrapperStyle: { fontSize: '12px', color: CHART_COLORS.tick },
} as const;
