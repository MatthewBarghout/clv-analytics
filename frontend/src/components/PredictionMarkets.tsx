import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';

const API_BASE = 'http://localhost:8000/api';
const REFRESH_INTERVAL_MS = 60_000;

interface PaperTrade {
  id: number;
  market_ticker: string;
  event_description: string;
  side: 'YES' | 'NO';
  entry_price: number;
  size_usd: number;
  entry_timestamp: string;
  exit_price: number | null;
  exit_timestamp: string | null;
  resolution_result: 'WIN' | 'LOSS' | 'PUSH' | null;
  pnl: number | null;
  strategy_tag: string;
  is_open: boolean;
}

interface PaperTradeStats {
  total_trades: number;
  open_trades: number;
  win_rate: number;
  total_pnl: number;
  avg_pnl: number;
  pnl_by_strategy_tag: Record<string, { pnl: number; trades: number }>;
  trades_by_month: Array<{ month: string; pnl: number; trades: number; wins: number }>;
}

interface CrossPlatformSignal {
  id: number;
  event_description: string;
  kalshi_ticker: string;
  kalshi_price: number;
  polymarket_price: number | null;
  metaculus_forecast: number | null;
  divergence_score: number;
  timestamp: string;
}

function pnlColor(pnl: number | null): string {
  if (pnl === null) return 'text-gray-400';
  return pnl >= 0 ? 'text-green-400' : 'text-red-400';
}

function resultBadge(result: string | null): string {
  if (result === 'WIN') return 'text-green-400 bg-green-500/20 border-green-500/30';
  if (result === 'LOSS') return 'text-red-400 bg-red-500/20 border-red-500/30';
  if (result === 'PUSH') return 'text-yellow-400 bg-yellow-500/20 border-yellow-500/30';
  return 'text-gray-400 bg-gray-500/15 border-gray-500/25';
}

function sideBadge(side: string): string {
  return side === 'YES'
    ? 'text-green-400 bg-green-500/20 border-green-500/30'
    : 'text-red-400 bg-red-500/20 border-red-500/30';
}

function formatPct(p: number): string {
  return `${(p * 100).toFixed(1)}%`;
}

function formatUsd(v: number): string {
  return v >= 0 ? `+$${v.toFixed(2)}` : `-$${Math.abs(v).toFixed(2)}`;
}

export const PredictionMarkets = React.memo(function PredictionMarkets() {
  const [trades, setTrades] = useState<PaperTrade[]>([]);
  const [stats, setStats] = useState<PaperTradeStats | null>(null);
  const [signals, setSignals] = useState<CrossPlatformSignal[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null);
  const [error, setError] = useState<string | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchData = useCallback(async (isManual = false) => {
    if (isManual) setRefreshing(true);
    setError(null);
    try {
      const [tradesRes, statsRes, signalsRes] = await Promise.all([
        fetch(`${API_BASE}/paper-trades?limit=100`),
        fetch(`${API_BASE}/paper-trades/stats`),
        fetch(`${API_BASE}/cross-platform-signals?limit=20&min_divergence=0.05`),
      ]);

      if (tradesRes.ok) setTrades(await tradesRes.json());
      if (statsRes.ok) setStats(await statsRes.json());
      if (signalsRes.ok) setSignals(await signalsRes.json());
      setLastRefresh(new Date());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch prediction market data');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    intervalRef.current = setInterval(() => fetchData(), REFRESH_INTERVAL_MS);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [fetchData]);

  const openTrades = useMemo(() => trades.filter((t) => t.is_open), [trades]);
  const closedTrades = useMemo(
    () => trades.filter((t) => !t.is_open).sort((a, b) => {
      const at = a.exit_timestamp ?? a.entry_timestamp;
      const bt = b.exit_timestamp ?? b.entry_timestamp;
      return new Date(bt).getTime() - new Date(at).getTime();
    }),
    [trades],
  );

  if (loading) {
    return (
      <div className="flex items-center justify-center py-16">
        <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-500 mr-3"></div>
        <span className="text-gray-400">Loading prediction market data...</span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Stats bar */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-white/5 rounded-lg p-4 border border-white/10">
          <div className="text-xs text-gray-400 uppercase mb-1">Total Trades</div>
          <div className="text-2xl font-bold text-white">{stats?.total_trades ?? 0}</div>
        </div>
        <div className="bg-white/5 rounded-lg p-4 border border-blue-500/20">
          <div className="text-xs text-gray-400 uppercase mb-1">Open Positions</div>
          <div className="text-2xl font-bold text-blue-400">{stats?.open_trades ?? 0}</div>
        </div>
        <div className="bg-white/5 rounded-lg p-4 border border-white/10">
          <div className="text-xs text-gray-400 uppercase mb-1">Win Rate</div>
          <div className="text-2xl font-bold text-white">
            {stats ? `${stats.win_rate.toFixed(1)}%` : '—'}
          </div>
        </div>
        <div
          className={`bg-white/5 rounded-lg p-4 border ${
            (stats?.total_pnl ?? 0) >= 0 ? 'border-green-500/20' : 'border-red-500/20'
          }`}
        >
          <div className="text-xs text-gray-400 uppercase mb-1">Total P&amp;L</div>
          <div className={`text-2xl font-bold ${pnlColor(stats?.total_pnl ?? null)}`}>
            {stats ? formatUsd(stats.total_pnl) : '—'}
          </div>
        </div>
      </div>

      {/* Controls row */}
      <div className="flex items-center justify-between text-sm text-gray-400">
        {lastRefresh && (
          <span>Last refresh: {lastRefresh.toLocaleTimeString()} · auto-refreshes every 60s</span>
        )}
        <button
          onClick={() => fetchData(true)}
          disabled={refreshing}
          className="px-3 py-1.5 bg-blue-500/20 hover:bg-blue-500/30 disabled:opacity-40 rounded border border-blue-500/30 text-blue-400 text-sm transition-all"
        >
          {refreshing ? 'Refreshing...' : 'Refresh Now'}
        </button>
      </div>

      {error && (
        <div className="px-4 py-2 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 text-sm">
          {error}
        </div>
      )}

      {/* Open Positions */}
      <div>
        <h3 className="text-base font-semibold text-white mb-3">
          Open Positions
          <span className="ml-2 text-xs text-gray-400 font-normal">({openTrades.length})</span>
        </h3>
        {openTrades.length === 0 ? (
          <div className="text-center py-8 text-gray-500 text-sm">
            No open positions. The system generates signals every 10 minutes.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-700/50">
                  {['Ticker', 'Event', 'Side', 'Entry Price', 'Size', 'Edge', 'Strategy', 'Opened'].map((h) => (
                    <th
                      key={h}
                      className="py-3 px-3 text-gray-400 font-semibold uppercase text-xs tracking-wider text-left"
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {openTrades.map((t) => (
                  <tr key={t.id} className="border-b border-gray-700/30 hover:bg-white/5 transition-all">
                    <td className="py-3 px-3 font-mono text-xs text-gray-300">{t.market_ticker}</td>
                    <td className="py-3 px-3 max-w-[220px] text-xs text-white leading-tight">
                      <span className="line-clamp-2">{t.event_description}</span>
                    </td>
                    <td className="py-3 px-3">
                      <span className={`px-1.5 py-0.5 rounded text-xs font-medium border ${sideBadge(t.side)}`}>
                        {t.side}
                      </span>
                    </td>
                    <td className="py-3 px-3 font-mono text-xs text-blue-400">{formatPct(t.entry_price)}</td>
                    <td className="py-3 px-3 text-xs text-gray-300">${t.size_usd.toFixed(0)}</td>
                    <td className="py-3 px-3 text-xs text-gray-400">—</td>
                    <td className="py-3 px-3 text-xs text-gray-400">{t.strategy_tag}</td>
                    <td className="py-3 px-3 text-xs text-gray-500">
                      {new Date(t.entry_timestamp).toLocaleDateString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Trade History */}
      <div>
        <h3 className="text-base font-semibold text-white mb-3">
          Trade History
          <span className="ml-2 text-xs text-gray-400 font-normal">({closedTrades.length})</span>
        </h3>
        {closedTrades.length === 0 ? (
          <div className="text-center py-8 text-gray-500 text-sm">
            No settled trades yet.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-700/50">
                  {['Ticker', 'Event', 'Side', 'Entry', 'Exit', 'Result', 'P&L', 'Strategy', 'Entry Date', 'Exit Date'].map((h) => (
                    <th
                      key={h}
                      className="py-3 px-3 text-gray-400 font-semibold uppercase text-xs tracking-wider text-left"
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {closedTrades.map((t) => (
                  <tr key={t.id} className="border-b border-gray-700/30 hover:bg-white/5 transition-all">
                    <td className="py-3 px-3 font-mono text-xs text-gray-300">{t.market_ticker}</td>
                    <td className="py-3 px-3 max-w-[180px] text-xs text-white">
                      <span className="line-clamp-2">{t.event_description}</span>
                    </td>
                    <td className="py-3 px-3">
                      <span className={`px-1.5 py-0.5 rounded text-xs font-medium border ${sideBadge(t.side)}`}>
                        {t.side}
                      </span>
                    </td>
                    <td className="py-3 px-3 font-mono text-xs text-blue-400">{formatPct(t.entry_price)}</td>
                    <td className="py-3 px-3 font-mono text-xs text-gray-300">
                      {t.exit_price !== null ? formatPct(t.exit_price) : '—'}
                    </td>
                    <td className="py-3 px-3">
                      {t.resolution_result && (
                        <span className={`px-1.5 py-0.5 rounded text-xs font-medium border ${resultBadge(t.resolution_result)}`}>
                          {t.resolution_result}
                        </span>
                      )}
                    </td>
                    <td className={`py-3 px-3 text-xs font-mono font-semibold ${pnlColor(t.pnl)}`}>
                      {t.pnl !== null ? formatUsd(t.pnl) : '—'}
                    </td>
                    <td className="py-3 px-3 text-xs text-gray-400">{t.strategy_tag}</td>
                    <td className="py-3 px-3 text-xs text-gray-500">
                      {new Date(t.entry_timestamp).toLocaleDateString()}
                    </td>
                    <td className="py-3 px-3 text-xs text-gray-500">
                      {t.exit_timestamp ? new Date(t.exit_timestamp).toLocaleDateString() : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Recent Signals */}
      {signals.length > 0 && (
        <div>
          <h3 className="text-base font-semibold text-white mb-3">
            Recent Signals
            <span className="ml-2 text-xs text-gray-400 font-normal">divergence &gt; 5%</span>
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-700/50">
                  {['Ticker', 'Event', 'Kalshi', 'Polymarket', 'Metaculus', 'Divergence', 'Detected'].map((h) => (
                    <th
                      key={h}
                      className="py-3 px-3 text-gray-400 font-semibold uppercase text-xs tracking-wider text-left"
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {signals.map((s) => (
                  <tr key={s.id} className="border-b border-gray-700/30 hover:bg-white/5 transition-all">
                    <td className="py-3 px-3 font-mono text-xs text-gray-300">{s.kalshi_ticker}</td>
                    <td className="py-3 px-3 max-w-[200px] text-xs text-white">
                      <span className="line-clamp-2">{s.event_description}</span>
                    </td>
                    <td className="py-3 px-3 font-mono text-xs text-blue-400">{formatPct(s.kalshi_price)}</td>
                    <td className="py-3 px-3 font-mono text-xs text-gray-300">
                      {s.polymarket_price !== null ? formatPct(s.polymarket_price) : '—'}
                    </td>
                    <td className="py-3 px-3 font-mono text-xs text-gray-300">
                      {s.metaculus_forecast !== null ? formatPct(s.metaculus_forecast) : '—'}
                    </td>
                    <td className="py-3 px-3">
                      <span className={`px-2 py-0.5 rounded text-xs font-bold border ${
                        s.divergence_score >= 0.10
                          ? 'text-green-400 bg-green-500/20 border-green-500/30'
                          : 'text-yellow-400 bg-yellow-500/15 border-yellow-500/25'
                      }`}>
                        {(s.divergence_score * 100).toFixed(1)}%
                      </span>
                    </td>
                    <td className="py-3 px-3 text-xs text-gray-500">
                      {new Date(s.timestamp).toLocaleTimeString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
});
