import React, { useCallback, useEffect, useRef, useState } from 'react';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const API_BASE = 'http://localhost:8000/api';

interface BankrollDataPoint {
  date: string;
  game_date: string;
  bet_number: number;
  cumulative_pl: number;
  bankroll: number;
  drawdown: number;
  drawdown_pct: number;
  result: 'win' | 'loss' | 'push';
  profit: number;
  bet_size: number;
  game: string;
  bookmaker: string;
  outcome: string;
  market: string;
  odds: number;
  closing_odds: number;
  clv: number;
}

interface BankrollSimulation {
  has_data: boolean;
  source?: string;
  message?: string;
  data_points: BankrollDataPoint[];
  summary: {
    starting_bankroll: number;
    ending_bankroll: number;
    total_profit_loss: number;
    roi_pct: number;
    total_bets: number;
    win_count: number;
    loss_count: number;
    push_count: number;
    win_rate: number;
    max_drawdown: number;
    max_drawdown_pct: number;
    total_wagered: number;
    peak_bankroll: number;
    strategy: string;
    sharpe_ratio: number;
    sortino_ratio: number;
    avg_bet_size: number;
    bet_size_std: number;
  };
  by_bookmaker?: Record<string, { bets: number; wins: number; losses: number; pushes: number; profit: number; wagered: number; roi: number; avg_clv: number; win_rate: number }>;
  by_market?: Record<string, { bets: number; wins: number; losses: number; pushes: number; profit: number; wagered: number; roi: number; avg_clv: number; win_rate: number }>;
}

const tooltipStyle = {
  backgroundColor: '#1F2937',
  border: '1px solid #374151',
  borderRadius: '8px',
};

export const BankrollSimulator: React.FC = () => {
  const [bankrollSimulation, setBankrollSimulation] = useState<BankrollSimulation | null>(null);
  const [loadingSimulation, setLoadingSimulation] = useState(false);
  const [simBetSize, setSimBetSize] = useState(100);
  const [simStartingBankroll, setSimStartingBankroll] = useState(10000);
  const [simStrategy, setSimStrategy] = useState('fixed');
  const [simFractionPercent, setSimFractionPercent] = useState(2);
  const [simMaxBetPercent, setSimMaxBetPercent] = useState(25);
  const [simBookmakerFilter, setSimBookmakerFilter] = useState<string | null>(null);
  const [simMarketFilter, setSimMarketFilter] = useState<string | null>(null);
  const [simClvThreshold, setSimClvThreshold] = useState<number | null>(null);
  // Default to Best EV+ picks — aligned with the Best EV tab
  const [simSource, setSimSource] = useState<'best_ev' | 'all'>('best_ev');

  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const fetchBankrollSimulation = useCallback(async (params: {
    betSize: number;
    startingBankroll: number;
    strategy: string;
    fractionPercent: number;
    maxBetPercent: number;
    bookmakerFilter: string | null;
    marketFilter: string | null;
    clvThreshold: number | null;
    source: string;
  }) => {
    try {
      setLoadingSimulation(true);
      const urlParams = new URLSearchParams({
        bet_size: params.betSize.toString(),
        starting_bankroll: params.startingBankroll.toString(),
        strategy: params.strategy,
        fraction_percent: params.fractionPercent.toString(),
        max_bet_percent: params.maxBetPercent.toString(),
        source: params.source,
      });
      if (params.bookmakerFilter) urlParams.append('bookmaker_filter', params.bookmakerFilter);
      if (params.marketFilter) urlParams.append('market_filter', params.marketFilter);
      if (params.clvThreshold !== null) urlParams.append('clv_threshold', params.clvThreshold.toString());

      const res = await fetch(`${API_BASE}/bankroll-simulation?${urlParams}`);
      if (res.ok) {
        const data = await res.json();
        setBankrollSimulation(data);
      }
    } catch (err) {
      console.error('Error fetching bankroll simulation:', err);
    } finally {
      setLoadingSimulation(false);
    }
  }, []);

  // Debounced effect: wait 500ms after last change before fetching
  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      fetchBankrollSimulation({
        betSize: simBetSize,
        startingBankroll: simStartingBankroll,
        strategy: simStrategy,
        fractionPercent: simFractionPercent,
        maxBetPercent: simMaxBetPercent,
        bookmakerFilter: simBookmakerFilter,
        marketFilter: simMarketFilter,
        clvThreshold: simClvThreshold,
        source: simSource,
      });
    }, 500);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [simBetSize, simStartingBankroll, simStrategy, simFractionPercent, simMaxBetPercent, simBookmakerFilter, simMarketFilter, simClvThreshold, simSource, fetchBankrollSimulation]);

  return (
    <div className="space-y-6">
      {/* Data Source Toggle */}
      <div className="flex items-center gap-3">
        <span className="text-sm text-gray-400 font-medium">Data Source:</span>
        <div className="flex bg-white/5 rounded-lg p-1 border border-white/10">
          <button
            onClick={() => setSimSource('best_ev')}
            className={`px-4 py-1.5 rounded text-sm font-medium transition-all ${
              simSource === 'best_ev'
                ? 'bg-gradient-to-r from-green-500/30 to-emerald-500/30 text-white border border-green-500/50'
                : 'text-gray-400 hover:text-white'
            }`}
          >
            Best EV+ Picks Only
          </button>
          <button
            onClick={() => setSimSource('all')}
            className={`px-4 py-1.5 rounded text-sm font-medium transition-all ${
              simSource === 'all'
                ? 'bg-white/20 text-white border border-white/30'
                : 'text-gray-400 hover:text-white'
            }`}
          >
            All Tracked Bets
          </button>
        </div>
        {simSource === 'best_ev' && (
          <span className="text-xs text-green-400 bg-green-500/10 px-2 py-1 rounded border border-green-500/20">
            Aligned with Best EV+ tab
          </span>
        )}
      </div>

      {/* Controls */}
      <div className="bg-white/5 rounded-lg p-4 border border-white/10 space-y-4">
        <div className="flex flex-wrap gap-4 items-center">
          <div className="flex items-center gap-2">
            <label className="text-sm text-gray-400">Strategy:</label>
            <select
              value={simStrategy}
              onChange={(e) => setSimStrategy(e.target.value)}
              className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-white text-sm"
            >
              <option value="fixed">Fixed Unit</option>
              <option value="fractional">Fractional</option>
              <option value="kelly">Full Kelly</option>
              <option value="half_kelly">Half Kelly</option>
              <option value="confidence">Confidence-Weighted</option>
            </select>
          </div>
          {simStrategy === 'fixed' && (
            <div className="flex items-center gap-2">
              <label className="text-sm text-gray-400">Bet Size:</label>
              <select
                value={simBetSize}
                onChange={(e) => setSimBetSize(Number(e.target.value))}
                className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-white text-sm"
              >
                <option value={25}>$25</option>
                <option value={50}>$50</option>
                <option value={100}>$100</option>
                <option value={250}>$250</option>
                <option value={500}>$500</option>
              </select>
            </div>
          )}
          {(simStrategy === 'fractional' || simStrategy === 'confidence') && (
            <div className="flex items-center gap-2">
              <label className="text-sm text-gray-400">Fraction:</label>
              <select
                value={simFractionPercent}
                onChange={(e) => setSimFractionPercent(Number(e.target.value))}
                className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-white text-sm"
              >
                <option value={1}>1%</option>
                <option value={2}>2%</option>
                <option value={3}>3%</option>
                <option value={5}>5%</option>
              </select>
            </div>
          )}
          {(simStrategy === 'kelly' || simStrategy === 'half_kelly') && (
            <div className="flex items-center gap-2">
              <label className="text-sm text-gray-400">Max Bet:</label>
              <select
                value={simMaxBetPercent}
                onChange={(e) => setSimMaxBetPercent(Number(e.target.value))}
                className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-white text-sm"
              >
                <option value={10}>10%</option>
                <option value={15}>15%</option>
                <option value={25}>25%</option>
              </select>
            </div>
          )}
          <div className="flex items-center gap-2">
            <label className="text-sm text-gray-400">Bankroll:</label>
            <select
              value={simStartingBankroll}
              onChange={(e) => setSimStartingBankroll(Number(e.target.value))}
              className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-white text-sm"
            >
              <option value={1000}>$1,000</option>
              <option value={5000}>$5,000</option>
              <option value={10000}>$10,000</option>
              <option value={25000}>$25,000</option>
              <option value={50000}>$50,000</option>
            </select>
          </div>
        </div>
        <div className="flex flex-wrap gap-4 items-center">
          <div className="flex items-center gap-2">
            <label className="text-sm text-gray-400">Bookmaker:</label>
            <select
              value={simBookmakerFilter || ''}
              onChange={(e) => setSimBookmakerFilter(e.target.value || null)}
              className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-white text-sm"
            >
              <option value="">All</option>
              <option value="DraftKings">DraftKings</option>
              <option value="FanDuel">FanDuel</option>
              <option value="theScore Bet">theScore</option>
              <option value="BetMGM">BetMGM</option>
            </select>
          </div>
          <div className="flex items-center gap-2">
            <label className="text-sm text-gray-400">Market:</label>
            <select
              value={simMarketFilter || ''}
              onChange={(e) => setSimMarketFilter(e.target.value || null)}
              className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-white text-sm"
            >
              <option value="">All</option>
              <option value="h2h">Moneyline</option>
              <option value="spreads">Spreads</option>
              <option value="totals">Totals</option>
            </select>
          </div>
          <div className="flex items-center gap-2">
            <label className="text-sm text-gray-400">Min CLV:</label>
            <select
              value={simClvThreshold ?? ''}
              onChange={(e) => setSimClvThreshold(e.target.value ? Number(e.target.value) : null)}
              className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-white text-sm"
            >
              <option value="">Any</option>
              <option value={1}>1%+</option>
              <option value={2}>2%+</option>
              <option value={3}>3%+</option>
              <option value={5}>5%+</option>
            </select>
          </div>
        </div>
      </div>

      {loadingSimulation ? (
        <div className="text-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-purple-500 mx-auto"></div>
          <p className="text-gray-400 mt-4">Running simulation...</p>
        </div>
      ) : bankrollSimulation?.has_data ? (
        <>
          {/* Summary Cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-4">
            {[
              { label: 'Total P&L', value: `${bankrollSimulation.summary.total_profit_loss >= 0 ? '+' : ''}$${bankrollSimulation.summary.total_profit_loss.toFixed(2)}`, color: bankrollSimulation.summary.total_profit_loss >= 0 ? 'text-green-400' : 'text-red-400' },
              { label: 'ROI', value: `${bankrollSimulation.summary.roi_pct >= 0 ? '+' : ''}${bankrollSimulation.summary.roi_pct.toFixed(2)}%`, color: bankrollSimulation.summary.roi_pct >= 0 ? 'text-green-400' : 'text-red-400' },
              { label: 'Win Rate', value: `${bankrollSimulation.summary.win_rate.toFixed(1)}%`, sub: `${bankrollSimulation.summary.win_count}W-${bankrollSimulation.summary.loss_count}L`, color: 'text-white' },
              { label: 'Max Drawdown', value: `$${bankrollSimulation.summary.max_drawdown.toFixed(2)}`, sub: `${bankrollSimulation.summary.max_drawdown_pct.toFixed(1)}% of peak`, color: 'text-red-400' },
              { label: 'Total Bets', value: bankrollSimulation.summary.total_bets.toString(), color: 'text-blue-400' },
              { label: 'Peak Bankroll', value: `$${bankrollSimulation.summary.peak_bankroll.toLocaleString()}`, color: 'text-purple-400' },
              { label: 'Sharpe Ratio', value: bankrollSimulation.summary.sharpe_ratio?.toFixed(2) ?? 'N/A', sub: 'Risk-adjusted', color: bankrollSimulation.summary.sharpe_ratio >= 0 ? 'text-cyan-400' : 'text-orange-400' },
              { label: 'Avg Bet Size', value: `$${bankrollSimulation.summary.avg_bet_size?.toFixed(0) ?? 'N/A'}`, sub: bankrollSimulation.summary.strategy?.replace('_', ' '), color: 'text-yellow-400' },
            ].map((card) => (
              <div key={card.label} className="bg-white/5 rounded-lg p-4 border border-white/10">
                <div className="text-xs text-gray-400 mb-1 uppercase">{card.label}</div>
                <div className={`text-2xl font-bold ${card.color}`}>{card.value}</div>
                {card.sub && <div className="text-xs text-gray-500 mt-1">{card.sub}</div>}
              </div>
            ))}
          </div>

          {/* Performance Breakdown */}
          {(bankrollSimulation.by_bookmaker || bankrollSimulation.by_market) && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {bankrollSimulation.by_bookmaker && Object.keys(bankrollSimulation.by_bookmaker).length > 0 && (
                <div className="bg-white/5 rounded-lg p-6 border border-white/10">
                  <h3 className="text-lg font-semibold text-white mb-4">By Bookmaker</h3>
                  <div className="space-y-3">
                    {Object.entries(bankrollSimulation.by_bookmaker)
                      .sort(([, a], [, b]) => b.profit - a.profit)
                      .map(([name, stats]) => (
                        <div key={name} className="flex items-center justify-between py-2 border-b border-gray-700/50">
                          <div>
                            <div className="text-white font-medium">{name}</div>
                            <div className="text-xs text-gray-500">{stats.bets} bets | {stats.win_rate}% win</div>
                          </div>
                          <div className="text-right">
                            <div className={`font-bold ${stats.profit >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                              {stats.profit >= 0 ? '+' : ''}${stats.profit.toFixed(0)}
                            </div>
                            <div className={`text-xs ${stats.roi >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                              {stats.roi >= 0 ? '+' : ''}{stats.roi}% ROI
                            </div>
                          </div>
                        </div>
                      ))}
                  </div>
                </div>
              )}
              {bankrollSimulation.by_market && Object.keys(bankrollSimulation.by_market).length > 0 && (
                <div className="bg-white/5 rounded-lg p-6 border border-white/10">
                  <h3 className="text-lg font-semibold text-white mb-4">By Market Type</h3>
                  <div className="space-y-3">
                    {Object.entries(bankrollSimulation.by_market)
                      .sort(([, a], [, b]) => b.profit - a.profit)
                      .map(([name, stats]) => (
                        <div key={name} className="flex items-center justify-between py-2 border-b border-gray-700/50">
                          <div>
                            <div className="text-white font-medium">{name.toUpperCase()}</div>
                            <div className="text-xs text-gray-500">{stats.bets} bets | {stats.win_rate}% win | {stats.avg_clv}% avg CLV</div>
                          </div>
                          <div className="text-right">
                            <div className={`font-bold ${stats.profit >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                              {stats.profit >= 0 ? '+' : ''}${stats.profit.toFixed(0)}
                            </div>
                            <div className={`text-xs ${stats.roi >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                              {stats.roi >= 0 ? '+' : ''}{stats.roi}% ROI
                            </div>
                          </div>
                        </div>
                      ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* P&L Curve Chart */}
          <div className="bg-white/5 rounded-lg p-6 border border-white/10">
            <h3 className="text-lg font-semibold text-white mb-4">Cumulative P&L</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={bankrollSimulation.data_points}>
                <defs>
                  <linearGradient id="colorPL" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#10B981" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#10B981" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
                <XAxis dataKey="bet_number" stroke="#9CA3AF" tick={{ fill: '#9CA3AF', fontSize: 12 }} label={{ value: 'Bet #', position: 'insideBottom', offset: -5, fill: '#9CA3AF' }} />
                <YAxis stroke="#9CA3AF" tick={{ fill: '#9CA3AF', fontSize: 12 }} tickFormatter={(v) => `$${v}`} />
                <Tooltip
                  contentStyle={tooltipStyle}
                  formatter={(value: number | undefined, name: string | undefined) => {
                    if (value === undefined) return ['N/A', name ?? ''];
                    if (name === 'cumulative_pl') return [`$${value.toFixed(2)}`, 'Cumulative P&L'];
                    return [value, name ?? ''];
                  }}
                  labelFormatter={(label) => `Bet #${label}`}
                />
                <Line type="monotone" dataKey="cumulative_pl" stroke="#10B981" strokeWidth={2} dot={false} fill="url(#colorPL)" />
                <Line type="monotone" dataKey={() => 0} stroke="#6B7280" strokeWidth={1} strokeDasharray="5 5" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Drawdown Chart */}
          <div className="bg-white/5 rounded-lg p-6 border border-white/10">
            <h3 className="text-lg font-semibold text-white mb-4">Drawdown Analysis</h3>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={bankrollSimulation.data_points}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
                <XAxis dataKey="bet_number" stroke="#9CA3AF" tick={{ fill: '#9CA3AF', fontSize: 12 }} />
                <YAxis stroke="#9CA3AF" tick={{ fill: '#9CA3AF', fontSize: 12 }} tickFormatter={(v) => `${v}%`} />
                <Tooltip
                  contentStyle={tooltipStyle}
                  formatter={(value: number | undefined) => [`${value?.toFixed(2) ?? '0'}%`, 'Drawdown']}
                  labelFormatter={(label) => `Bet #${label}`}
                />
                <Bar dataKey="drawdown_pct" fill="#EF4444" opacity={0.7} radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Bet History Table */}
          <div className="bg-white/5 rounded-lg p-6 border border-white/10">
            <h3 className="text-lg font-semibold text-white mb-1">Bet History</h3>
            <p className="text-xs text-gray-500 mb-4">
              {bankrollSimulation.source === 'best_ev'
                ? 'Showing Best EV+ tracked picks only — same picks displayed in the Best EV+ tab'
                : 'Showing all CLV-tracked opportunities'}
            </p>
            <div className="overflow-x-auto max-h-[500px]">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-gray-900 z-10">
                  <tr className="border-b border-gray-700">
                    {['Date', 'Game', 'Book', 'Bet', 'Entry', 'EV/CLV', 'Result', 'P&L', 'Running'].map((h) => (
                      <th key={h} className={`py-2 px-2 text-gray-400 text-xs ${h === 'Date' || h === 'Game' || h === 'Book' || h === 'Bet' ? 'text-left' : h === 'P&L' || h === 'Running' ? 'text-right' : 'text-center'}`}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {bankrollSimulation.data_points.map((bet, idx) => (
                    <tr key={idx} className="border-b border-gray-800 hover:bg-white/5">
                      <td className="py-2 px-2 text-gray-400 text-xs whitespace-nowrap">{bet.game_date}</td>
                      <td className="py-2 px-2 text-white text-xs max-w-[150px] truncate" title={bet.game}>{bet.game}</td>
                      <td className="py-2 px-2 text-gray-300 text-xs max-w-[80px] truncate" title={bet.bookmaker}>{bet.bookmaker}</td>
                      <td className="py-2 px-2 text-xs">
                        <span className="text-purple-400">{bet.market}</span>
                        <span className="text-gray-500 ml-1">-</span>
                        <span className="text-white ml-1 max-w-[100px] truncate inline-block align-bottom" title={bet.outcome}>{bet.outcome}</span>
                      </td>
                      <td className="text-center py-2 px-2 text-blue-400 text-xs font-mono">{bet.odds.toFixed(2)}</td>
                      <td className="text-center py-2 px-2 text-green-400 text-xs font-medium">
                        {bet.clv >= 0 ? '+' : ''}{bet.clv}%
                      </td>
                      <td className="text-center py-2 px-2">
                        <span className={`px-1.5 py-0.5 rounded text-xs font-medium ${bet.result === 'win' ? 'bg-green-500/20 text-green-400' : bet.result === 'loss' ? 'bg-red-500/20 text-red-400' : 'bg-gray-500/20 text-gray-400'}`}>
                          {bet.result === 'win' ? 'W' : bet.result === 'loss' ? 'L' : 'P'}
                        </span>
                      </td>
                      <td className={`text-right py-2 px-2 font-medium text-xs ${bet.profit >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {bet.profit >= 0 ? '+' : ''}${bet.profit.toFixed(0)}
                      </td>
                      <td className={`text-right py-2 px-2 font-medium text-xs ${bet.cumulative_pl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {bet.cumulative_pl >= 0 ? '+' : ''}${bet.cumulative_pl.toFixed(0)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      ) : (
        <div className="text-center py-12 text-gray-400">
          <div className="text-6xl mb-4">📊</div>
          <p className="text-lg mb-2">No simulation data available</p>
          {bankrollSimulation?.message ? (
            <p className="text-sm max-w-md mx-auto">{bankrollSimulation.message}</p>
          ) : simSource === 'best_ev' ? (
            <p className="text-sm">
              No settled Best EV+ picks found yet. Picks are saved daily and settled after games complete.
              <br />
              <span className="text-gray-500">Switch to "All Tracked Bets" to see historical CLV-based data.</span>
            </p>
          ) : (
            <p className="text-sm">Settled bets are needed to run the bankroll simulation</p>
          )}
        </div>
      )}
    </div>
  );
};
