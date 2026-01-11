import React, { useEffect, useState } from 'react';
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
import { GlassCard } from './components/GlassCard';
import { AnimatedCounter } from './components/AnimatedCounter';
import { TimeRangeSelector } from './components/TimeRangeSelector';
import { GameDetailsModal } from './components/GameDetailsModal';
import { GameAnalysis } from './components/GameAnalysis';

interface CLVStats {
  mean_clv: number | null;
  median_clv: number | null;
  total_analyzed: number;
  positive_clv_count: number;
  positive_clv_percentage: number;
  by_bookmaker: Record<string, { avg_clv: number; count: number }>;
  by_market_type: Record<string, { avg_clv: number; count: number }>;
}

interface BookmakerStats {
  bookmaker_name: string;
  total_snapshots: number;
  avg_clv: number | null;
  positive_clv_percentage: number;
}

interface CLVHistoryPoint {
  date: string;
  avg_clv: number | null;
  count: number;
}

interface GameWithCLV {
  game_id: number;
  home_team: string;
  away_team: string;
  commence_time: string;
  completed: boolean;
  snapshots_count: number;
  closing_lines_count: number;
  avg_clv: number | null;
  home_score?: number | null;
  away_score?: number | null;
  winner?: string | null;
}

interface MLModelStats {
  is_trained: boolean;
  movement_mae: number | null;
  movement_rmse: number | null;
  movement_r2: number | null;
  directional_accuracy: number | null;
  directional_precision: number | null;
  directional_recall: number | null;
  training_records: number | null;
  last_trained: string | null;
  baseline_mae: number | null;
  improvement_vs_baseline: number | null;
}

interface FeatureImportance {
  feature_name: string;
  importance: number;
}

interface EVOpportunity {
  game_id: number;
  home_team: string;
  away_team: string;
  commence_time: string;
  bookmaker_name: string;
  market_type: string;
  outcome_name: string;
  current_line: string;
  predicted_movement: number;
  predicted_direction: string;
  confidence: number;
  ev_score: number;
  was_constrained: boolean;
}

interface DailyCLVReport {
  id: number;
  report_date: string;
  games_analyzed: number;
  total_opportunities: number;
  avg_clv: number;
  median_clv: number;
  positive_clv_count: number;
  positive_clv_percentage: number;
  // Performance tracking
  settled_count?: number;
  win_count?: number;
  loss_count?: number;
  push_count?: number;
  hypothetical_profit?: number;
  win_rate?: number;
  roi?: number;
  best_opportunities: Array<{
    game_id: number;
    bookmaker: string;
    market_type: string;
    outcome: string;
    clv: number;
    entry_odds: number;
    closing_odds: number;
  }>;
  by_bookmaker: Record<string, { avg_clv: number; count: number; positive_count: number }>;
  by_market: Record<string, { avg_clv: number; count: number; positive_count: number }>;
}

const API_BASE = 'http://localhost:8000/api';

export default function Dashboard() {
  const [stats, setStats] = useState<CLVStats | null>(null);
  const [bookmakers, setBookmakers] = useState<BookmakerStats[]>([]);
  const [history, setHistory] = useState<CLVHistoryPoint[]>([]);
  const [games, setGames] = useState<GameWithCLV[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [timeRange, setTimeRange] = useState<'7d' | '30d' | '90d' | 'all'>('30d');
  const [selectedGame, setSelectedGame] = useState<GameWithCLV | null>(null);
  const [gamesView, setGamesView] = useState<'recent' | 'history' | 'best-ev' | 'daily-reports'>('recent');
  const [historyGames, setHistoryGames] = useState<GameWithCLV[]>([]);
  const [expandedGameId, setExpandedGameId] = useState<number | null>(null);
  const [mlStats, setMlStats] = useState<MLModelStats | null>(null);
  const [featureImportance, setFeatureImportance] = useState<FeatureImportance[]>([]);
  const [bestOpportunities, setBestOpportunities] = useState<EVOpportunity[]>([]);
  const [dailyReports, setDailyReports] = useState<DailyCLVReport[]>([]);

  useEffect(() => {
    fetchData();
  }, [timeRange]);

  useEffect(() => {
    if (gamesView === 'history') {
      fetchHistoryGames();
    } else if (gamesView === 'daily-reports') {
      fetchDailyReports();
    }
  }, [gamesView]);

  const fetchDailyReports = async () => {
    try {
      const res = await fetch(`${API_BASE}/daily-reports?limit=14`);
      if (res.ok) {
        const data = await res.json();
        setDailyReports(data);
      }
    } catch (err) {
      console.error('Error fetching daily reports:', err);
    }
  };

  const fetchData = async () => {
    try {
      setLoading(true);
      const [statsRes, bookmakersRes, historyRes, gamesRes, mlStatsRes, featureImportanceRes, opportunitiesRes] = await Promise.all([
        fetch(`${API_BASE}/stats`),
        fetch(`${API_BASE}/bookmakers`),
        fetch(`${API_BASE}/clv-history?time_range=${timeRange}`),
        fetch(`${API_BASE}/games?limit=20`),
        fetch(`${API_BASE}/ml/stats`),
        fetch(`${API_BASE}/ml/feature-importance`),
        fetch(`${API_BASE}/ml/best-opportunities?limit=50&min_ev_score=0`),
      ]);

      if (!statsRes.ok || !bookmakersRes.ok || !historyRes.ok || !gamesRes.ok) {
        throw new Error('Failed to fetch data');
      }

      const [statsData, bookmakersData, historyData, gamesData, mlStatsData, featureImportanceData, opportunitiesData] = await Promise.all([
        statsRes.json(),
        bookmakersRes.json(),
        historyRes.json(),
        gamesRes.json(),
        mlStatsRes.ok ? mlStatsRes.json() : null,
        featureImportanceRes.ok ? featureImportanceRes.json() : [],
        opportunitiesRes.ok ? opportunitiesRes.json() : [],
      ]);

      setStats(statsData);
      setBookmakers(bookmakersData);
      setHistory(historyData);
      setGames(gamesData);
      setMlStats(mlStatsData);
      setFeatureImportance(featureImportanceData);
      setBestOpportunities(opportunitiesData);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const fetchHistoryGames = async () => {
    try {
      // Fetch more games for history view (limit 100)
      const res = await fetch(`${API_BASE}/games?limit=100`);
      if (res.ok) {
        const data = await res.json();
        // Filter only completed games with CLV data
        const completedWithCLV = data.filter((g: GameWithCLV) => g.completed && g.avg_clv !== null);
        setHistoryGames(completedWithCLV);
      }
    } catch (err) {
      console.error('Error fetching history:', err);
    }
  };

  const handleGameClick = (game: GameWithCLV) => {
    setSelectedGame(game);
  };

  const handleCloseModal = () => {
    setSelectedGame(null);
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white flex items-center justify-center">
        <GlassCard className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <div className="text-xl">Loading analytics...</div>
        </GlassCard>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white flex items-center justify-center">
        <GlassCard gradient="red" className="text-center max-w-md">
          <div className="text-6xl mb-4">⚠️</div>
          <div className="text-xl text-red-400 mb-4">Error: {error}</div>
          <button
            onClick={fetchData}
            className="px-6 py-2 bg-red-500/20 hover:bg-red-500/30 rounded-lg border border-red-500/30 transition-all duration-200"
          >
            Retry
          </button>
        </GlassCard>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white p-4 md:p-8">
      <div className="max-w-7xl mx-auto">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-8">
          <h1 className="text-4xl md:text-5xl font-bold mb-4 md:mb-0 bg-gradient-to-r from-white via-gray-200 to-gray-400 bg-clip-text text-transparent">
            CLV Analytics Dashboard
          </h1>
          <button
            onClick={fetchData}
            className="px-4 py-2 bg-white/10 hover:bg-white/20 rounded-lg backdrop-blur-sm border border-white/20 transition-all duration-200"
          >
            Refresh Data
          </button>
        </div>

        {/* Enhanced Summary Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <GlassCard gradient={(stats?.mean_clv || 0) > 0 ? 'green' : 'red'}>
            <h3 className="text-gray-400 text-sm font-medium mb-2 uppercase tracking-wider">
              Mean CLV
            </h3>
            <AnimatedCounter
              value={stats?.mean_clv || 0}
              decimals={2}
              suffix="%"
              className={`text-4xl font-bold ${
                (stats?.mean_clv || 0) > 0 ? 'text-clv-positive-400' : 'text-clv-negative-400'
              }`}
            />
          </GlassCard>

          <GlassCard gradient="blue">
            <h3 className="text-gray-400 text-sm font-medium mb-2 uppercase tracking-wider">
              Total Analyzed
            </h3>
            <AnimatedCounter
              value={stats?.total_analyzed || 0}
              className="text-4xl font-bold text-blue-400"
            />
            <p className="text-sm text-gray-400 mt-2">Betting opportunities</p>
          </GlassCard>

          <GlassCard gradient="green">
            <h3 className="text-gray-400 text-sm font-medium mb-2 uppercase tracking-wider">
              Positive CLV Rate
            </h3>
            <AnimatedCounter
              value={stats?.positive_clv_percentage || 0}
              decimals={1}
              suffix="%"
              className="text-4xl font-bold text-clv-positive-400"
            />
            <p className="text-sm text-gray-400 mt-2">
              {stats?.positive_clv_count || 0} of {stats?.total_analyzed || 0} bets
            </p>
          </GlassCard>

          <GlassCard gradient="purple">
            <h3 className="text-gray-400 text-sm font-medium mb-2 uppercase tracking-wider">
              Median CLV
            </h3>
            <AnimatedCounter
              value={stats?.median_clv || 0}
              decimals={2}
              suffix="%"
              className={`text-4xl font-bold ${
                (stats?.median_clv || 0) > 0 ? 'text-clv-positive-400' : 'text-clv-negative-400'
              }`}
            />
          </GlassCard>
        </div>

        {/* ML Model Performance */}
        {mlStats?.is_trained && (
          <GlassCard className="mb-8">
            <h2 className="text-2xl font-bold mb-6 bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">
              ML Model Performance
            </h2>

            {/* Model Metrics Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4 mb-6">
              <div className="bg-white/5 rounded-lg p-4 border border-white/10">
                <h3 className="text-gray-400 text-xs font-medium mb-1 uppercase tracking-wider">
                  Movement MAE
                </h3>
                <AnimatedCounter
                  value={mlStats.movement_mae || 0}
                  decimals={4}
                  className="text-2xl font-bold text-purple-400"
                />
                <p className="text-xs text-gray-500 mt-1">Avg prediction error</p>
              </div>

              <div className="bg-white/5 rounded-lg p-4 border border-white/10">
                <h3 className="text-gray-400 text-xs font-medium mb-1 uppercase tracking-wider">
                  Directional Accuracy
                </h3>
                <AnimatedCounter
                  value={(mlStats.directional_accuracy || 0) * 100}
                  decimals={1}
                  suffix="%"
                  className="text-2xl font-bold text-green-400"
                />
                <p className="text-xs text-gray-500 mt-1">Correct direction</p>
              </div>

              <div className="bg-white/5 rounded-lg p-4 border border-white/10">
                <h3 className="text-gray-400 text-xs font-medium mb-1 uppercase tracking-wider">
                  Improvement
                </h3>
                <AnimatedCounter
                  value={mlStats.improvement_vs_baseline || 0}
                  decimals={1}
                  suffix="%"
                  className="text-2xl font-bold text-blue-400"
                />
                <p className="text-xs text-gray-500 mt-1">vs baseline</p>
              </div>

              <div className="bg-white/5 rounded-lg p-4 border border-white/10">
                <h3 className="text-gray-400 text-xs font-medium mb-1 uppercase tracking-wider">
                  Precision
                </h3>
                <AnimatedCounter
                  value={(mlStats.directional_precision || 0) * 100}
                  decimals={1}
                  suffix="%"
                  className="text-2xl font-bold text-cyan-400"
                />
                <p className="text-xs text-gray-500 mt-1">Direction precision</p>
              </div>

              <div className="bg-white/5 rounded-lg p-4 border border-white/10">
                <h3 className="text-gray-400 text-xs font-medium mb-1 uppercase tracking-wider">
                  Training Records
                </h3>
                <AnimatedCounter
                  value={mlStats.training_records || 0}
                  className="text-2xl font-bold text-orange-400"
                />
                <p className="text-xs text-gray-500 mt-1">Data points</p>
              </div>
            </div>

            {/* Feature Importance Chart */}
            {featureImportance.length > 0 && (
              <div className="mt-6">
                <h3 className="text-lg font-semibold mb-4 text-gray-300">
                  Feature Importance
                </h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart
                    data={featureImportance.slice(0, 7)}
                    layout="vertical"
                    margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
                    <XAxis
                      type="number"
                      stroke="#9CA3AF"
                      tick={{ fill: '#9CA3AF' }}
                    />
                    <YAxis
                      type="category"
                      dataKey="feature_name"
                      stroke="#9CA3AF"
                      tick={{ fill: '#9CA3AF' }}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1F2937',
                        border: '1px solid #374151',
                        borderRadius: '8px',
                        backdropFilter: 'blur(10px)',
                      }}
                      labelStyle={{ color: '#F3F4F6' }}
                      itemStyle={{ color: '#9CA3AF' }}
                      formatter={(value: number | undefined) => [(value ? (value * 100).toFixed(2) : '0') + '%', 'Importance']}
                    />
                    <Bar
                      dataKey="importance"
                      fill="#3b82f6"
                      radius={[0, 4, 4, 0]}
                      isAnimationActive={true}
                      animationDuration={800}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Model Info */}
            {mlStats.last_trained && (
              <div className="mt-4 text-xs text-gray-500 text-center">
                Last trained: {new Date(mlStats.last_trained).toLocaleString()}
              </div>
            )}
          </GlassCard>
        )}

        {!mlStats?.is_trained && (
          <GlassCard className="mb-8 border-yellow-500/30 bg-yellow-500/5">
            <div className="flex items-center gap-4">
              <div className="text-4xl">⚠️</div>
              <div>
                <h3 className="text-lg font-semibold text-yellow-400 mb-1">
                  ML Model Not Trained
                </h3>
                <p className="text-sm text-gray-400">
                  Run <code className="px-2 py-1 bg-gray-800 rounded text-yellow-400">poetry run python scripts/train_model.py</code> to train the closing line prediction model.
                </p>
              </div>
            </div>
          </GlassCard>
        )}

        {/* Market Type Breakdown */}
        {stats?.by_market_type && Object.keys(stats.by_market_type).length > 0 && (
          <GlassCard className="mb-8">
            <h2 className="text-2xl font-bold mb-6 bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">
              CLV by Market Type
            </h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart
                data={Object.entries(stats.by_market_type).map(([market, data]) => ({
                  market: market.toUpperCase(),
                  avg_clv: data.avg_clv,
                  count: data.count,
                }))}
              >
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
                  contentStyle={{
                    backgroundColor: '#1F2937',
                    border: '1px solid #374151',
                    borderRadius: '8px',
                    backdropFilter: 'blur(10px)',
                  }}
                  labelStyle={{ color: '#F3F4F6' }}
                  itemStyle={{ color: '#9CA3AF' }}
                  formatter={(value: number | undefined, _name: string | undefined, props: any) => [
                    `${value ? value.toFixed(2) : '0'}% (${props.payload.count} bets)`,
                    'Avg CLV',
                  ]}
                />
                <Bar dataKey="avg_clv" radius={[8, 8, 0, 0]} isAnimationActive={true} animationDuration={800}>
                  {Object.entries(stats.by_market_type).map(([_market, data], index) => (
                    <Cell key={`cell-${index}`} fill={data.avg_clv > 0 ? 'url(#colorPositive)' : 'url(#colorNegative)'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </GlassCard>
        )}

        {/* CLV Trend Chart with Time Range Selector */}
        <GlassCard className="mb-8">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-6">
            <h2 className="text-2xl font-bold mb-4 md:mb-0 bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">
              CLV Trend Over Time
            </h2>
            <TimeRangeSelector selected={timeRange} onChange={setTimeRange} />
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
                contentStyle={{
                  backgroundColor: '#1F2937',
                  border: '1px solid #374151',
                  borderRadius: '8px',
                  backdropFilter: 'blur(10px)',
                }}
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
                isAnimationActive={true}
                animationBegin={0}
                animationDuration={1000}
                animationEasing="ease-in-out"
              />
            </LineChart>
          </ResponsiveContainer>
        </GlassCard>

        {/* Enhanced Bookmaker Stats */}
        <GlassCard className="mb-8">
          <h2 className="text-2xl font-bold mb-6 bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">
            Bookmaker Performance
          </h2>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-700/50">
                  <th className="text-left py-4 px-4 text-gray-400 font-semibold uppercase text-xs tracking-wider">
                    Bookmaker
                  </th>
                  <th className="text-center py-4 px-4 text-gray-400 font-semibold uppercase text-xs tracking-wider">
                    Avg CLV
                  </th>
                  <th className="text-center py-4 px-4 text-gray-400 font-semibold uppercase text-xs tracking-wider">
                    Positive CLV %
                  </th>
                  <th className="text-center py-4 px-4 text-gray-400 font-semibold uppercase text-xs tracking-wider">
                    Total Snapshots
                  </th>
                  <th className="text-right py-4 px-4 text-gray-400 font-semibold uppercase text-xs tracking-wider">
                    Performance
                  </th>
                </tr>
              </thead>
              <tbody>
                {bookmakers.map((bookmaker, index) => (
                  <tr
                    key={bookmaker.bookmaker_name}
                    className="border-b border-gray-700/30 hover:bg-white/5 transition-colors duration-200"
                    style={{ animationDelay: `${index * 50}ms` }}
                  >
                    <td className="py-4 px-4">
                      <span className="font-medium text-white">{bookmaker.bookmaker_name}</span>
                    </td>
                    <td className="text-center py-4 px-4">
                      <span
                        className={`font-bold text-lg ${
                          (bookmaker.avg_clv || 0) > 0 ? 'text-clv-positive-400' : 'text-clv-negative-400'
                        }`}
                      >
                        {bookmaker.avg_clv?.toFixed(2) || 'N/A'}%
                      </span>
                    </td>
                    <td className="text-center py-4 px-4">
                      <div className="flex items-center justify-center gap-2">
                        <div className="w-24 h-2 bg-gray-700 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-clv-positive-600 to-clv-positive-400 rounded-full transition-all duration-500"
                            style={{ width: `${bookmaker.positive_clv_percentage}%` }}
                          />
                        </div>
                        <span className="text-clv-positive-400 font-medium text-sm">
                          {bookmaker.positive_clv_percentage.toFixed(1)}%
                        </span>
                      </div>
                    </td>
                    <td className="text-center py-4 px-4">
                      <span className="text-blue-400 font-medium">{bookmaker.total_snapshots.toLocaleString()}</span>
                    </td>
                    <td className="text-right py-4 px-4">
                      {(bookmaker.avg_clv || 0) > 0 ? (
                        <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-clv-positive-500/20 text-clv-positive-400 border border-clv-positive-500/30">
                          Favorable
                        </span>
                      ) : (
                        <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-clv-negative-500/20 text-clv-negative-400 border border-clv-negative-500/30">
                          Unfavorable
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </GlassCard>

        {/* Games Table with History */}
        <GlassCard>
          <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-6">
            <h2 className="text-2xl font-bold bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent mb-4 md:mb-0">
              Games
            </h2>
            <div className="flex items-center gap-4">
              <div className="flex gap-2">
                <button
                  onClick={() => setGamesView('recent')}
                  className={`px-4 py-2 rounded-lg transition-all duration-200 ${
                    gamesView === 'recent'
                      ? 'bg-white/20 text-white border border-white/30'
                      : 'bg-white/5 text-gray-400 hover:text-white hover:bg-white/10 border border-white/10'
                  }`}
                >
                  Recent
                </button>
                <button
                  onClick={() => setGamesView('history')}
                  className={`px-4 py-2 rounded-lg transition-all duration-200 ${
                    gamesView === 'history'
                      ? 'bg-white/20 text-white border border-white/30'
                      : 'bg-white/5 text-gray-400 hover:text-white hover:bg-white/10 border border-white/10'
                  }`}
                >
                  History ({historyGames.length})
                </button>
                {mlStats?.is_trained && bestOpportunities.length > 0 && (
                  <button
                    onClick={() => setGamesView('best-ev')}
                    className={`px-4 py-2 rounded-lg transition-all duration-200 ${
                      gamesView === 'best-ev'
                        ? 'bg-gradient-to-r from-green-500/30 to-emerald-500/30 text-white border border-green-500/50'
                        : 'bg-white/5 text-gray-400 hover:text-white hover:bg-white/10 border border-white/10'
                    }`}
                  >
                    Best +EV ({bestOpportunities.length})
                  </button>
                )}
                <button
                  onClick={() => setGamesView('daily-reports')}
                  className={`px-4 py-2 rounded-lg transition-all duration-200 ${
                    gamesView === 'daily-reports'
                      ? 'bg-gradient-to-r from-blue-500/30 to-cyan-500/30 text-white border border-blue-500/50'
                      : 'bg-white/5 text-gray-400 hover:text-white hover:bg-white/10 border border-white/10'
                  }`}
                >
                  Daily Reports
                </button>
              </div>
              {gamesView !== 'best-ev' && (
                <div className="text-sm text-gray-400">
                  💡 Click any game to view detailed betting lines
                </div>
              )}
            </div>
          </div>
          <div className="overflow-x-auto">
            {gamesView === 'daily-reports' ? (
              // Daily Reports View
              <div className="space-y-6">
                {dailyReports.map((report, index) => (
                  <div
                    key={report.id}
                    className="bg-white/5 rounded-lg p-6 border border-white/10 hover:bg-white/8 transition-all duration-200"
                    style={{ animationDelay: `${index * 50}ms` }}
                  >
                    <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-4">
                      <div>
                        <h3 className="text-xl font-bold text-white mb-1">
                          {new Date(report.report_date).toLocaleDateString('en-US', {
                            weekday: 'long',
                            year: 'numeric',
                            month: 'long',
                            day: 'numeric',
                          })}
                        </h3>
                        <p className="text-sm text-gray-400">
                          {report.games_analyzed} games • {report.total_opportunities} betting opportunities
                        </p>
                      </div>
                      <div className="flex gap-4 mt-4 md:mt-0">
                        <div className="text-center">
                          <div className="text-sm text-gray-400 mb-1">Avg CLV</div>
                          <div className={`text-2xl font-bold ${report.avg_clv > 0 ? 'text-green-400' : 'text-red-400'}`}>
                            {report.avg_clv > 0 ? '+' : ''}{report.avg_clv.toFixed(2)}%
                          </div>
                        </div>
                        <div className="text-center">
                          <div className="text-sm text-gray-400 mb-1">Positive CLV</div>
                          <div className="text-2xl font-bold text-green-400">
                            {report.positive_clv_percentage.toFixed(0)}%
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Profit Tracking Stats - Only show if we have settled bets */}
                    {report.settled_count !== undefined && report.settled_count > 0 && (
                      <div className="mt-4 p-4 bg-gradient-to-r from-purple-500/10 to-blue-500/10 rounded-lg border border-purple-500/20">
                        <h4 className="text-sm font-semibold text-purple-300 mb-3 uppercase tracking-wider flex items-center gap-2">
                          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                            <path d="M8.433 7.418c.155-.103.346-.196.567-.267v1.698a2.305 2.305 0 01-.567-.267C8.07 8.34 8 8.114 8 8c0-.114.07-.34.433-.582zM11 12.849v-1.698c.22.071.412.164.567.267.364.243.433.468.433.582 0 .114-.07.34-.433.582a2.305 2.305 0 01-.567.267z" />
                            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-13a1 1 0 10-2 0v.092a4.535 4.535 0 00-1.676.662C6.602 6.234 6 7.009 6 8c0 .99.602 1.765 1.324 2.246.48.32 1.054.545 1.676.662v1.941c-.391-.127-.68-.317-.843-.504a1 1 0 10-1.51 1.31c.562.649 1.413 1.076 2.353 1.253V15a1 1 0 102 0v-.092a4.535 4.535 0 001.676-.662C13.398 13.766 14 12.991 14 12c0-.99-.602-1.765-1.324-2.246A4.535 4.535 0 0011 9.092V7.151c.391.127.68.317.843.504a1 1 0 101.511-1.31c-.563-.649-1.413-1.076-2.354-1.253V5z" clipRule="evenodd" />
                          </svg>
                          Hypothetical Performance ($100 per bet)
                        </h4>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                          <div className="text-center">
                            <div className="text-xs text-gray-400 mb-1">Win Rate</div>
                            <div className="text-xl font-bold text-white">
                              {report.win_rate?.toFixed(1)}%
                            </div>
                            <div className="text-xs text-gray-500 mt-1">
                              {report.win_count}W-{report.loss_count}L
                              {report.push_count! > 0 && `-${report.push_count}P`}
                            </div>
                          </div>
                          <div className="text-center">
                            <div className="text-xs text-gray-400 mb-1">Settled</div>
                            <div className="text-xl font-bold text-white">
                              {report.settled_count}/{report.total_opportunities}
                            </div>
                            <div className="text-xs text-gray-500 mt-1">
                              {((report.settled_count / report.total_opportunities) * 100).toFixed(0)}% complete
                            </div>
                          </div>
                          <div className="text-center">
                            <div className="text-xs text-gray-400 mb-1">Profit/Loss</div>
                            <div className={`text-xl font-bold ${(report.hypothetical_profit || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                              {(report.hypothetical_profit || 0) >= 0 ? '+' : ''}${report.hypothetical_profit?.toFixed(2)}
                            </div>
                            <div className="text-xs text-gray-500 mt-1">
                              ${(report.settled_count * 100).toLocaleString()} wagered
                            </div>
                          </div>
                          <div className="text-center">
                            <div className="text-xs text-gray-400 mb-1">ROI</div>
                            <div className={`text-xl font-bold ${(report.roi || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                              {(report.roi || 0) >= 0 ? '+' : ''}{report.roi?.toFixed(2)}%
                            </div>
                            <div className="text-xs text-gray-500 mt-1">
                              return on investment
                            </div>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Top 3 Best Opportunities */}
                    <div className="mt-4">
                      <h4 className="text-sm font-semibold text-gray-300 mb-3 uppercase tracking-wider">
                        Top 3 Best Opportunities
                      </h4>
                      <div className="space-y-2">
                        {report.best_opportunities.slice(0, 3).map((opp, oppIndex) => (
                          <div
                            key={oppIndex}
                            className="flex items-center justify-between bg-white/5 rounded-lg p-3 border border-white/5"
                          >
                            <div className="flex items-center gap-3">
                              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-yellow-500/30 to-orange-500/30 border border-yellow-500/50 flex items-center justify-center">
                                <span className="text-yellow-400 font-bold text-sm">#{oppIndex + 1}</span>
                              </div>
                              <div>
                                <div className="text-sm font-medium text-white">
                                  {opp.bookmaker} • {opp.market_type}
                                </div>
                                <div className="text-xs text-gray-400">{opp.outcome}</div>
                              </div>
                            </div>
                            <div className="text-right">
                              <div className="text-lg font-bold text-green-400">
                                +{opp.clv.toFixed(2)}%
                              </div>
                              <div className="text-xs text-gray-500">
                                {opp.entry_odds > 0 ? '+' : ''}{opp.entry_odds} → {opp.closing_odds > 0 ? '+' : ''}{opp.closing_odds}
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Quick Stats */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-4 pt-4 border-t border-white/10">
                      <div className="text-center">
                        <div className="text-xs text-gray-400 mb-1">Best Book</div>
                        <div className="text-sm font-medium text-blue-400">
                          {Object.entries(report.by_bookmaker)
                            .sort(([, a], [, b]) => b.avg_clv - a.avg_clv)[0]?.[0] || 'N/A'}
                        </div>
                      </div>
                      <div className="text-center">
                        <div className="text-xs text-gray-400 mb-1">Best Market</div>
                        <div className="text-sm font-medium text-purple-400">
                          {Object.entries(report.by_market)
                            .sort(([, a], [, b]) => b.avg_clv - a.avg_clv)[0]?.[0] || 'N/A'}
                        </div>
                      </div>
                      <div className="text-center">
                        <div className="text-xs text-gray-400 mb-1">Median CLV</div>
                        <div className={`text-sm font-medium ${report.median_clv > 0 ? 'text-green-400' : 'text-gray-400'}`}>
                          {report.median_clv > 0 ? '+' : ''}{report.median_clv.toFixed(2)}%
                        </div>
                      </div>
                      <div className="text-center">
                        <div className="text-xs text-gray-400 mb-1">Hit Rate</div>
                        <div className="text-sm font-medium text-cyan-400">
                          {report.positive_clv_count}/{report.total_opportunities}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
                {dailyReports.length === 0 && (
                  <div className="text-center py-12 text-gray-400">
                    <p className="text-lg mb-2">No daily reports yet</p>
                    <p className="text-sm">Reports are generated daily at 9:00 AM for completed games</p>
                  </div>
                )}
              </div>
            ) : gamesView === 'best-ev' ? (
              // Best +EV Opportunities Table
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-700/50">
                    <th className="text-left py-4 px-4 text-gray-400 font-semibold uppercase text-xs tracking-wider">
                      Game
                    </th>
                    <th className="text-left py-4 px-4 text-gray-400 font-semibold uppercase text-xs tracking-wider">
                      Market
                    </th>
                    <th className="text-left py-4 px-4 text-gray-400 font-semibold uppercase text-xs tracking-wider">
                      Outcome
                    </th>
                    <th className="text-center py-4 px-4 text-gray-400 font-semibold uppercase text-xs tracking-wider">
                      Current Line
                    </th>
                    <th className="text-center py-4 px-4 text-gray-400 font-semibold uppercase text-xs tracking-wider">
                      Predicted Movement
                    </th>
                    <th className="text-center py-4 px-4 text-gray-400 font-semibold uppercase text-xs tracking-wider">
                      Direction
                    </th>
                    <th className="text-center py-4 px-4 text-gray-400 font-semibold uppercase text-xs tracking-wider">
                      Confidence
                    </th>
                    <th className="text-left py-4 px-4 text-gray-400 font-semibold uppercase text-xs tracking-wider">
                      Bookmaker
                    </th>
                    <th className="text-center py-4 px-4 text-gray-400 font-semibold uppercase text-xs tracking-wider">
                      EV Score
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {bestOpportunities.map((opp, index) => (
                    <tr
                      key={index}
                      className="border-b border-gray-700/30 hover:bg-white/5 transition-all duration-200"
                      style={{ animationDelay: `${index * 30}ms` }}
                    >
                      <td className="py-4 px-4">
                        <span className="font-medium text-white text-sm">
                          {opp.away_team} @ {opp.home_team}
                        </span>
                        <div className="text-xs text-gray-500 mt-1">
                          {new Date(opp.commence_time).toLocaleString('en-US', {
                            month: 'short',
                            day: 'numeric',
                            hour: 'numeric',
                            minute: '2-digit',
                          })}
                        </div>
                      </td>
                      <td className="py-4 px-4">
                        <span className="px-2 py-1 rounded text-xs font-medium bg-blue-500/20 text-blue-400 border border-blue-500/30">
                          {opp.market_type}
                        </span>
                      </td>
                      <td className="py-4 px-4">
                        <span className="text-sm font-medium text-white">{opp.outcome_name}</span>
                      </td>
                      <td className="text-center py-4 px-4">
                        <span className="text-sm font-mono text-blue-400">{opp.current_line}</span>
                      </td>
                      <td className="text-center py-4 px-4">
                        <div className="flex items-center justify-center gap-1">
                          <span className={`text-sm font-mono font-bold ${opp.predicted_movement > 0 ? 'text-green-400' : 'text-red-400'}`}>
                            {opp.predicted_movement > 0 ? '+' : ''}{opp.predicted_movement.toFixed(3)}
                          </span>
                          {opp.was_constrained && (
                            <span
                              className="text-yellow-400 text-xs"
                              title="Prediction capped to realistic range based on historical data"
                            >
                              ⚠️
                            </span>
                          )}
                        </div>
                      </td>
                      <td className="text-center py-4 px-4">
                        <span className={`px-2 py-1 rounded text-xs font-medium border ${
                          opp.predicted_direction === 'UP'
                            ? 'bg-green-500/20 text-green-400 border-green-500/30'
                            : opp.predicted_direction === 'DOWN'
                            ? 'bg-red-500/20 text-red-400 border-red-500/30'
                            : 'bg-gray-500/20 text-gray-400 border-gray-500/30'
                        }`}>
                          {opp.predicted_direction}
                        </span>
                      </td>
                      <td className="text-center py-4 px-4">
                        <span className="text-sm font-medium text-cyan-400">
                          {(opp.confidence * 100).toFixed(0)}%
                        </span>
                      </td>
                      <td className="py-4 px-4">
                        <span className="text-sm text-gray-300">{opp.bookmaker_name}</span>
                      </td>
                      <td className="text-center py-4 px-4">
                        <span className="px-2 py-1 rounded text-sm font-bold bg-green-500/20 text-green-400 border border-green-500/30">
                          {opp.ev_score.toFixed(2)}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              // Regular Games Table
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-700/50">
                    <th className="text-left py-4 px-4 text-gray-400 font-semibold uppercase text-xs tracking-wider">
                      Game
                    </th>
                    <th className="text-left py-4 px-4 text-gray-400 font-semibold uppercase text-xs tracking-wider">
                      Time
                    </th>
                    <th className="text-center py-4 px-4 text-gray-400 font-semibold uppercase text-xs tracking-wider">
                      Avg CLV
                    </th>
                    <th className="text-center py-4 px-4 text-gray-400 font-semibold uppercase text-xs tracking-wider">
                      Snapshots
                    </th>
                    <th className="text-center py-4 px-4 text-gray-400 font-semibold uppercase text-xs tracking-wider">
                      Closing Lines
                    </th>
                    <th className="text-center py-4 px-4 text-gray-400 font-semibold uppercase text-xs tracking-wider">
                      Status
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {(gamesView === 'recent' ? games : historyGames).map((game, index) => (
                  <React.Fragment key={game.game_id}>
                    <tr
                      onClick={() => {
                        if (gamesView === 'history') {
                          setExpandedGameId(expandedGameId === game.game_id ? null : game.game_id);
                        } else {
                          handleGameClick(game);
                        }
                      }}
                      className="border-b border-gray-700/30 hover:bg-white/5 transition-all duration-200 cursor-pointer"
                      style={{ animationDelay: `${index * 30}ms` }}
                      title={gamesView === 'history' ? 'Click to view analysis' : 'Click to view detailed betting lines'}
                    >
                    <td className="py-4 px-4">
                      <div className="font-medium text-white">
                        <div>{game.away_team} @ {game.home_team}</div>
                        {game.home_score !== null && game.away_score !== null && (
                          <div className="text-sm mt-1">
                            <span className="text-gray-400">Final: </span>
                            <span className={`font-bold ${game.winner === 'away' ? 'text-green-400' : 'text-gray-300'}`}>
                              {game.away_score}
                            </span>
                            <span className="text-gray-500"> - </span>
                            <span className={`font-bold ${game.winner === 'home' ? 'text-green-400' : 'text-gray-300'}`}>
                              {game.home_score}
                            </span>
                          </div>
                        )}
                      </div>
                    </td>
                    <td className="py-4 px-4 text-gray-300 text-sm">
                      {new Date(game.commence_time).toLocaleString('en-US', {
                        month: 'short',
                        day: 'numeric',
                        hour: '2-digit',
                        minute: '2-digit',
                      })}
                    </td>
                    <td className="text-center py-4 px-4">
                      {game.avg_clv !== null ? (
                        <span
                          className={`font-bold text-lg ${
                            game.avg_clv > 0 ? 'text-clv-positive-400' : 'text-clv-negative-400'
                          }`}
                        >
                          {game.avg_clv > 0 ? '+' : ''}
                          {game.avg_clv.toFixed(2)}%
                        </span>
                      ) : (
                        <span className="text-gray-500 text-sm">N/A</span>
                      )}
                    </td>
                    <td className="text-center py-4 px-4">
                      <span className="inline-flex items-center px-2 py-1 rounded-md text-sm font-medium bg-blue-500/20 text-blue-400 border border-blue-500/30">
                        {game.snapshots_count}
                      </span>
                    </td>
                    <td className="text-center py-4 px-4">
                      <span className="inline-flex items-center px-2 py-1 rounded-md text-sm font-medium bg-purple-500/20 text-purple-400 border border-purple-500/30">
                        {game.closing_lines_count}
                      </span>
                    </td>
                    <td className="text-center py-4 px-4">
                      <span
                        className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-medium ${
                          game.completed
                            ? 'bg-gray-700/50 text-gray-300 border border-gray-600/50'
                            : 'bg-green-500/20 text-green-400 border border-green-500/30'
                        }`}
                      >
                        {game.completed ? 'Completed' : 'Upcoming'}
                      </span>
                    </td>
                  </tr>
                  {gamesView === 'history' && expandedGameId === game.game_id && (
                    <tr className="bg-gray-800/50">
                      <td colSpan={6} className="p-6">
                        <GameAnalysis gameId={game.game_id} />
                      </td>
                    </tr>
                  )}
                </React.Fragment>
                  ))}
                </tbody>
              </table>
            )}
            {gamesView === 'history' && historyGames.length === 0 && (
              <div className="text-center py-12 text-gray-400">
                <p className="text-lg mb-2">No completed games with CLV data yet</p>
                <p className="text-sm">Games will appear here once they're completed and have closing line data</p>
              </div>
            )}
            {gamesView === 'best-ev' && bestOpportunities.length === 0 && (
              <div className="text-center py-12 text-gray-400">
                <p className="text-lg mb-2">No +EV opportunities available</p>
                <p className="text-sm">Opportunities will appear here when the ML model predicts favorable betting conditions</p>
              </div>
            )}
          </div>
        </GlassCard>

        {/* Game Details Modal */}
        {selectedGame && (
          <GameDetailsModal
            gameId={selectedGame.game_id}
            homeTeam={selectedGame.home_team}
            awayTeam={selectedGame.away_team}
            onClose={handleCloseModal}
          />
        )}
      </div>
    </div>
  );
}
