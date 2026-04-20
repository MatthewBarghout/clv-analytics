import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';

const API_BASE = 'http://localhost:8000/api';
const REFRESH_INTERVAL_MS = 60_000; // 60 seconds

interface ArbOpportunity {
  id: number;
  event_title: string;
  market_source: 'kalshi';
  sportsbook_name: string;
  sportsbook_odds: number;
  pm_implied_prob: number;   // already %, e.g. 52.3
  pm_implied_odds: number;
  arb_spread: number;        // percentage points
  market_url: string | null;
  timestamp: string;
  game_id: number | null;
}

interface ArbHistoryPoint {
  date: string;
  count: number;
  avg_spread: number;
  max_spread: number;
}

interface ArbResponse {
  opportunities: ArbOpportunity[];
  total: number;
  min_spread: number;
}

interface ArbHistoryResponse {
  history: ArbHistoryPoint[];
  days: number;
  total_records: number;
}

const tooltipStyle = {
  backgroundColor: '#1F2937',
  border: '1px solid #374151',
  borderRadius: '8px',
};

function spreadColor(spread: number): string {
  if (spread >= 2) return 'text-green-400 bg-green-500/20 border-green-500/30';
  if (spread >= 1) return 'text-yellow-400 bg-yellow-500/15 border-yellow-500/25';
  return 'text-gray-400 bg-gray-500/15 border-gray-500/25';
}

function sourceColor(_source: string): string {
  return 'text-blue-400 bg-blue-500/20 border-blue-500/30';
}

export const ArbOpportunities = React.memo(function ArbOpportunities() {
  const [data, setData] = useState<ArbResponse | null>(null);
  const [history, setHistory] = useState<ArbHistoryPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sourceFilter, setSourceFilter] = useState<string>('');
  const [minSpread, setMinSpread] = useState<number>(0);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchData = useCallback(async (isManual = false) => {
    if (isManual) setRefreshing(true);
    setError(null);
    try {
      const params = new URLSearchParams({ limit: '100', min_spread: minSpread.toString() });
      if (sourceFilter) params.append('source', sourceFilter);

      const [oppRes, histRes] = await Promise.all([
        fetch(`${API_BASE}/arb-opportunities?${params}`),
        fetch(`${API_BASE}/arb-history?days=7&min_spread=0.5`),
      ]);

      if (oppRes.ok) setData(await oppRes.json());
      if (histRes.ok) {
        const h: ArbHistoryResponse = await histRes.json();
        setHistory(h.history);
      }
      setLastRefresh(new Date());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch arb data');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [minSpread, sourceFilter]);

  const manualRefresh = async () => {
    setRefreshing(true);
    try {
      // Trigger backend re-poll first
      await fetch(`${API_BASE}/arb/refresh`, { method: 'POST' });
    } catch {
      // ignore — still fetch cached data below
    }
    fetchData(true);
  };

  // Initial load + auto-refresh every 60s
  useEffect(() => {
    fetchData();
    intervalRef.current = setInterval(() => fetchData(), REFRESH_INTERVAL_MS);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [fetchData]);

  const opportunities = data?.opportunities ?? [];
  const highSpread = opportunities.filter((o) => o.arb_spread >= 2).length;
  const medSpread = opportunities.filter((o) => o.arb_spread >= 1 && o.arb_spread < 2).length;

  if (loading) {
    return (
      <div className="flex items-center justify-center py-16">
        <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-500 mr-3"></div>
        <span className="text-gray-400">Scanning prediction markets...</span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Summary bar */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-white/5 rounded-lg p-4 border border-white/10">
          <div className="text-xs text-gray-400 uppercase mb-1">Total Opportunities</div>
          <div className="text-2xl font-bold text-white">{opportunities.length}</div>
        </div>
        <div className="bg-white/5 rounded-lg p-4 border border-green-500/20">
          <div className="text-xs text-gray-400 uppercase mb-1">Strong Arb ≥2%</div>
          <div className="text-2xl font-bold text-green-400">{highSpread}</div>
        </div>
        <div className="bg-white/5 rounded-lg p-4 border border-yellow-500/20">
          <div className="text-xs text-gray-400 uppercase mb-1">Marginal 1–2%</div>
          <div className="text-2xl font-bold text-yellow-400">{medSpread}</div>
        </div>
        <div className="bg-white/5 rounded-lg p-4 border border-white/10">
          <div className="text-xs text-gray-400 uppercase mb-1">Best Spread</div>
          <div className="text-2xl font-bold text-cyan-400">
            {opportunities.length > 0 ? `+${opportunities[0].arb_spread.toFixed(2)}%` : '—'}
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap gap-4 items-center justify-between">
        <div className="flex gap-3 items-center flex-wrap">
          <div className="flex items-center gap-2">
            <label className="text-sm text-gray-400">Source:</label>
            <select
              value={sourceFilter}
              onChange={(e) => setSourceFilter(e.target.value)}
              className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-white text-sm"
            >
              <option value="">All</option>
              <option value="kalshi">Kalshi</option>
            </select>
          </div>
          <div className="flex items-center gap-2">
            <label className="text-sm text-gray-400">Min Spread:</label>
            <select
              value={minSpread}
              onChange={(e) => setMinSpread(Number(e.target.value))}
              className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-white text-sm"
            >
              <option value={0}>Any</option>
              <option value={0.5}>0.5%+</option>
              <option value={1}>1%+</option>
              <option value={2}>2%+</option>
            </select>
          </div>
        </div>
        <div className="flex items-center gap-3 text-sm text-gray-400">
          {lastRefresh && (
            <span>
              Last refresh: {lastRefresh.toLocaleTimeString()} · auto-refreshes every 60s
            </span>
          )}
          <button
            onClick={manualRefresh}
            disabled={refreshing}
            className="px-3 py-1.5 bg-blue-500/20 hover:bg-blue-500/30 disabled:opacity-40 rounded border border-blue-500/30 text-blue-400 text-sm transition-all"
          >
            {refreshing ? 'Refreshing...' : 'Refresh Now'}
          </button>
        </div>
      </div>

      {error && (
        <div className="px-4 py-2 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 text-sm">
          {error}
        </div>
      )}

      {/* Arb opportunities table */}
      {opportunities.length === 0 ? (
        <div className="text-center py-16">
          <div className="text-5xl mb-4">🔄</div>
          <p className="text-lg text-gray-400 mb-2">No active arbitrage opportunities</p>
          <p className="text-sm text-gray-500">
            The system polls Kalshi every 5 minutes. Opportunities appear
            when prediction market odds diverge from sportsbook lines.
          </p>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-700/50">
                {['Event', 'Source', 'Sportsbook', 'SB Odds', 'PM Implied Prob', 'PM Odds', 'Arb Spread', 'Updated', ''].map((h, i) => (
                  <th
                    key={h || i}
                    className={`py-3 px-3 text-gray-400 font-semibold uppercase text-xs tracking-wider ${
                      i === 0 || i === 2 ? 'text-left' : i === 8 ? '' : 'text-center'
                    }`}
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {opportunities.map((opp) => (
                <tr key={opp.id} className="border-b border-gray-700/30 hover:bg-white/5 transition-all">
                  <td className="py-3 px-3 max-w-[200px]">
                    <span className="text-white font-medium text-xs leading-tight line-clamp-2">
                      {opp.event_title}
                    </span>
                  </td>
                  <td className="text-center py-3 px-3">
                    <span className={`px-1.5 py-0.5 rounded text-xs font-medium border ${sourceColor(opp.market_source)}`}>
                      {opp.market_source}
                    </span>
                  </td>
                  <td className="py-3 px-3 text-gray-300 text-xs">{opp.sportsbook_name}</td>
                  <td className="text-center py-3 px-3 text-blue-400 font-mono text-xs">
                    {opp.sportsbook_odds.toFixed(3)}
                  </td>
                  <td className="text-center py-3 px-3 text-gray-300 text-xs">
                    {opp.pm_implied_prob.toFixed(1)}%
                  </td>
                  <td className="text-center py-3 px-3 text-gray-300 font-mono text-xs">
                    {opp.pm_implied_odds.toFixed(3)}
                  </td>
                  <td className="text-center py-3 px-3">
                    <span className={`px-2 py-0.5 rounded text-xs font-bold border ${spreadColor(opp.arb_spread)}`}>
                      +{opp.arb_spread.toFixed(2)}%
                    </span>
                  </td>
                  <td className="text-center py-3 px-3 text-gray-500 text-xs">
                    {new Date(opp.timestamp).toLocaleTimeString()}
                  </td>
                  <td className="py-3 px-3">
                    {opp.market_url && (
                      <a
                        href={opp.market_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="px-2 py-1 bg-white/10 hover:bg-white/20 rounded border border-white/20 text-xs text-gray-300 transition-all whitespace-nowrap"
                      >
                        View Market
                      </a>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="text-xs text-gray-500 mt-3">
            Arb Spread = PM implied probability minus sportsbook implied probability.
            Positive spread indicates the PM assigns higher probability than the sportsbook.
          </p>
        </div>
      )}

      {/* Historical chart */}
      {history.length > 0 && (
        <div className="bg-white/5 rounded-lg p-6 border border-white/10">
          <h3 className="text-lg font-semibold text-white mb-4">7-Day Arb Spread Trend</h3>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={history}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
              <XAxis dataKey="date" stroke="#9CA3AF" tick={{ fill: '#9CA3AF', fontSize: 11 }} />
              <YAxis stroke="#9CA3AF" tick={{ fill: '#9CA3AF', fontSize: 11 }} tickFormatter={(v) => `${v}%`} />
              <Tooltip
                contentStyle={tooltipStyle}
                formatter={(value: string | number | undefined, name: string | undefined) => [
                  value != null ? `${typeof value === 'number' ? value.toFixed(2) : value}%` : '—',
                  name === 'avg_spread' ? 'Avg Spread' : 'Max Spread',
                ]}
              />
              <Legend />
              <Line type="monotone" dataKey="avg_spread" stroke="#60A5FA" strokeWidth={2} dot={false} name="avg_spread" />
              <Line type="monotone" dataKey="max_spread" stroke="#34D399" strokeWidth={2} dot={false} strokeDasharray="4 4" name="max_spread" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
});
