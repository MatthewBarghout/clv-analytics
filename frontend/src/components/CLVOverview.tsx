import React, { useMemo } from 'react';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { GlassCard } from './GlassCard';
import { TimeRangeSelector } from './TimeRangeSelector';

interface CLVStats {
  mean_clv: number | null;
  median_clv: number | null;
  total_analyzed: number;
  positive_clv_count: number;
  positive_clv_percentage: number;
  by_bookmaker: Record<string, { avg_clv: number; count: number }>;
  by_market_type: Record<string, { avg_clv: number; count: number }>;
}

interface CLVHistoryPoint {
  date: string;
  avg_clv: number | null;
  count: number;
}

interface Props {
  stats: CLVStats | null;
  history: CLVHistoryPoint[];
  timeRange: '7d' | '30d' | '90d' | 'all';
  onTimeRangeChange: (range: '7d' | '30d' | '90d' | 'all') => void;
}

const tooltipStyle = {
  backgroundColor: '#1F2937',
  border: '1px solid #374151',
  borderRadius: '8px',
  backdropFilter: 'blur(10px)',
};

export const CLVOverview: React.FC<Props> = React.memo(({ stats, history, timeRange, onTimeRangeChange }) => {
  const marketChartData = useMemo(
    () =>
      stats?.by_market_type
        ? Object.entries(stats.by_market_type).map(([market, data]) => ({
            market: market.toUpperCase(),
            avg_clv: data.avg_clv,
            count: data.count,
          }))
        : [],
    [stats?.by_market_type]
  );

  const marketEntries = useMemo(
    () => (stats?.by_market_type ? Object.entries(stats.by_market_type) : []),
    [stats?.by_market_type]
  );

  const hasMarketData = stats?.by_market_type && Object.keys(stats.by_market_type).length > 0;

  return (
    <>
      {hasMarketData && <GlassCard className="mb-8">
        <h2 className="text-2xl font-bold mb-6 bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">
          CLV by Market Type
        </h2>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={marketChartData}>
            <defs>
              <linearGradient id="colorPositive" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#22c55e" stopOpacity={0.8} />
                <stop offset="100%" stopColor="#22c55e" stopOpacity={0.3} />
              </linearGradient>
              <linearGradient id="colorNegative" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#ef4444" stopOpacity={0.8} />
                <stop offset="100%" stopColor="#ef4444" stopOpacity={0.3} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
            <XAxis dataKey="market" stroke="#9CA3AF" tick={{ fill: '#9CA3AF' }} />
            <YAxis
              stroke="#9CA3AF"
              tick={{ fill: '#9CA3AF' }}
              label={{ value: 'Avg CLV %', angle: -90, position: 'insideLeft', fill: '#9CA3AF' }}
            />
            <Tooltip
              contentStyle={tooltipStyle}
              labelStyle={{ color: '#F3F4F6' }}
              itemStyle={{ color: '#9CA3AF' }}
              formatter={(value: number | undefined, _name: string | undefined, props: any) => [
                `${value ? value.toFixed(2) : '0'}% (${props.payload.count} bets)`,
                'Avg CLV',
              ]}
            />
            <Bar dataKey="avg_clv" radius={[8, 8, 0, 0]} isAnimationActive animationDuration={800}>
              {marketEntries.map(([_market, data], index) => (
                <Cell key={`cell-${index}`} fill={data.avg_clv > 0 ? 'url(#colorPositive)' : 'url(#colorNegative)'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </GlassCard>}

      <GlassCard className="mb-8">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-6">
          <h2 className="text-2xl font-bold mb-4 md:mb-0 bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">
            CLV Trend Over Time
          </h2>
          <TimeRangeSelector selected={timeRange} onChange={onTimeRangeChange} />
        </div>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={history}>
            <defs>
              <linearGradient id="colorCLV" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#10B981" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#10B981" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
            <XAxis dataKey="date" stroke="#9CA3AF" tick={{ fill: '#9CA3AF', fontSize: 12 }} />
            <YAxis
              stroke="#9CA3AF"
              tick={{ fill: '#9CA3AF', fontSize: 12 }}
              label={{ value: 'Avg CLV %', angle: -90, position: 'insideLeft', fill: '#9CA3AF' }}
            />
            <Tooltip
              contentStyle={tooltipStyle}
              labelStyle={{ color: '#F3F4F6', fontWeight: 'bold' }}
              itemStyle={{ color: '#10B981' }}
            />
            <Legend wrapperStyle={{ color: '#9CA3AF' }} />
            <Line
              type="monotone"
              dataKey="avg_clv"
              stroke="#10B981"
              strokeWidth={3}
              dot={{ fill: '#10B981', strokeWidth: 2, r: 4 }}
              activeDot={{ r: 6, strokeWidth: 2 }}
              fill="url(#colorCLV)"
              name="Avg CLV %"
              isAnimationActive
              animationBegin={0}
              animationDuration={1000}
              animationEasing="ease-in-out"
            />
          </LineChart>
        </ResponsiveContainer>
      </GlassCard>
    </>
  );
});
