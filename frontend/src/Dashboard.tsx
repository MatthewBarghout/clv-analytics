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
  const [gamesView, setGamesView] = useState<'recent' | 'history'>('recent');
  const [historyGames, setHistoryGames] = useState<GameWithCLV[]>([]);
  const [expandedGameId, setExpandedGameId] = useState<number | null>(null);

  useEffect(() => {
    fetchData();
  }, [timeRange]);

  useEffect(() => {
    if (gamesView === 'history') {
      fetchHistoryGames();
    }
  }, [gamesView]);

  const fetchData = async () => {
    try {
      setLoading(true);
      const [statsRes, bookmakersRes, historyRes, gamesRes] = await Promise.all([
        fetch(`${API_BASE}/stats`),
        fetch(`${API_BASE}/bookmakers`),
        fetch(`${API_BASE}/clv-history?time_range=${timeRange}`),
        fetch(`${API_BASE}/games?limit=20`),
      ]);

      if (!statsRes.ok || !bookmakersRes.ok || !historyRes.ok || !gamesRes.ok) {
        throw new Error('Failed to fetch data');
      }

      const [statsData, bookmakersData, historyData, gamesData] = await Promise.all([
        statsRes.json(),
        bookmakersRes.json(),
        historyRes.json(),
        gamesRes.json(),
      ]);

      setStats(statsData);
      setBookmakers(bookmakersData);
      setHistory(historyData);
      setGames(gamesData);
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
                  formatter={(value: number, _name: string, props: any) => [
                    `${value.toFixed(2)}% (${props.payload.count} bets)`,
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
              </div>
              <div className="text-sm text-gray-400">
                💡 Click any game to view detailed betting lines
              </div>
            </div>
          </div>
          <div className="overflow-x-auto">
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
                      <span className="font-medium text-white">
                        {game.away_team} @ {game.home_team}
                      </span>
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
            {gamesView === 'history' && historyGames.length === 0 && (
              <div className="text-center py-12 text-gray-400">
                <p className="text-lg mb-2">No completed games with CLV data yet</p>
                <p className="text-sm">Games will appear here once they're completed and have closing line data</p>
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
