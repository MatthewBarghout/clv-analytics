import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { GlassCard } from './components/GlassCard';
import { AnimatedCounter } from './components/AnimatedCounter';
import { GameDetailsModal } from './components/GameDetailsModal';
import { GameAnalysis } from './components/GameAnalysis';
import { MLStats } from './components/MLStats';
import { CLVOverview } from './components/CLVOverview';
import { BookmakerPerformance } from './components/BookmakerPerformance';
import { DailyReports } from './components/DailyReports';
import { BankrollSimulator } from './components/BankrollSimulator';
import { MyBets } from './components/MyBets';

// ── Types ──────────────────────────────────────────────────────────────────

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

const API_BASE = 'http://localhost:8000/api';

type GamesView = 'recent' | 'history' | 'best-ev' | 'daily-reports' | 'bankroll-sim' | 'my-bets';

// ── Dashboard ──────────────────────────────────────────────────────────────

export default function Dashboard() {
  const [stats, setStats] = useState<CLVStats | null>(null);
  const [bookmakers, setBookmakers] = useState<BookmakerStats[]>([]);
  const [history, setHistory] = useState<CLVHistoryPoint[]>([]);
  const [games, setGames] = useState<GameWithCLV[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [timeRange, setTimeRange] = useState<'7d' | '30d' | '90d' | 'all'>('30d');
  const [selectedGame, setSelectedGame] = useState<GameWithCLV | null>(null);
  const [gamesView, setGamesView] = useState<GamesView>('recent');
  const [historyGames, setHistoryGames] = useState<GameWithCLV[]>([]);
  const [expandedGameId, setExpandedGameId] = useState<number | null>(null);
  const [mlStats, setMlStats] = useState<MLModelStats | null>(null);
  const [featureImportance, setFeatureImportance] = useState<FeatureImportance[]>([]);
  const [bestOpportunities, setBestOpportunities] = useState<EVOpportunity[]>([]);

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      const [statsRes, bookmakersRes, historyRes, gamesRes, mlStatsRes, featureImportanceRes, opportunitiesRes] =
        await Promise.all([
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

      const [statsData, bookmakersData, historyData, gamesData, mlStatsData, featureImportanceData, opportunitiesData] =
        await Promise.all([
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
  }, [timeRange]);

  const fetchHistoryGames = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/games?limit=100`);
      if (res.ok) {
        const data = await res.json();
        setHistoryGames(data.filter((g: GameWithCLV) => g.completed && g.avg_clv !== null));
      }
    } catch (err) {
      console.error('Error fetching history:', err);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  useEffect(() => {
    if (gamesView === 'history') fetchHistoryGames();
  }, [gamesView, fetchHistoryGames]);

  const handleGameClick = useCallback((game: GameWithCLV) => setSelectedGame(game), []);
  const handleCloseModal = useCallback(() => setSelectedGame(null), []);

  // Tab data arrays memoized
  const displayedGames = useMemo(
    () => (gamesView === 'recent' ? games : historyGames),
    [gamesView, games, historyGames]
  );

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
        {/* Header */}
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

        {/* Summary Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <GlassCard gradient={(stats?.mean_clv || 0) > 0 ? 'green' : 'red'}>
            <h3 className="text-gray-400 text-sm font-medium mb-2 uppercase tracking-wider">Mean CLV</h3>
            <AnimatedCounter
              value={stats?.mean_clv || 0}
              decimals={2}
              suffix="%"
              className={`text-4xl font-bold ${(stats?.mean_clv || 0) > 0 ? 'text-clv-positive-400' : 'text-clv-negative-400'}`}
            />
          </GlassCard>
          <GlassCard gradient="blue">
            <h3 className="text-gray-400 text-sm font-medium mb-2 uppercase tracking-wider">Total Analyzed</h3>
            <AnimatedCounter value={stats?.total_analyzed || 0} className="text-4xl font-bold text-blue-400" />
            <p className="text-sm text-gray-400 mt-2">Betting opportunities</p>
          </GlassCard>
          <GlassCard gradient="green">
            <h3 className="text-gray-400 text-sm font-medium mb-2 uppercase tracking-wider">Positive CLV Rate</h3>
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
            <h3 className="text-gray-400 text-sm font-medium mb-2 uppercase tracking-wider">Median CLV</h3>
            <AnimatedCounter
              value={stats?.median_clv || 0}
              decimals={2}
              suffix="%"
              className={`text-4xl font-bold ${(stats?.median_clv || 0) > 0 ? 'text-clv-positive-400' : 'text-clv-negative-400'}`}
            />
          </GlassCard>
        </div>

        {/* ML Model Performance */}
        <MLStats mlStats={mlStats} featureImportance={featureImportance} />

        {/* CLV Overview (market breakdown + trend chart) */}
        <CLVOverview stats={stats} history={history} timeRange={timeRange} onTimeRangeChange={setTimeRange} />

        {/* Bookmaker Performance */}
        <BookmakerPerformance bookmakers={bookmakers} />

        {/* Games / Views section */}
        <GlassCard>
          <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-6">
            <h2 className="text-2xl font-bold bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent mb-4 md:mb-0">
              Games
            </h2>
            <div className="flex items-center gap-4">
              <div className="flex gap-2 flex-wrap">
                {(['recent', 'history', 'best-ev', 'daily-reports', 'bankroll-sim', 'my-bets'] as const).map((view) => {
                  const labels: Record<GamesView, string> = {
                    recent: 'Recent',
                    history: `History (${historyGames.length})`,
                    'best-ev': `Best +EV (${bestOpportunities.length})`,
                    'daily-reports': 'Daily Reports',
                    'bankroll-sim': 'Bankroll Sim',
                    'my-bets': 'My Bets',
                  };
                  const activeClass: Record<GamesView, string> = {
                    recent: 'bg-white/20 text-white border border-white/30',
                    history: 'bg-white/20 text-white border border-white/30',
                    'best-ev': 'bg-gradient-to-r from-green-500/30 to-emerald-500/30 text-white border border-green-500/50',
                    'daily-reports': 'bg-gradient-to-r from-blue-500/30 to-cyan-500/30 text-white border border-blue-500/50',
                    'bankroll-sim': 'bg-gradient-to-r from-purple-500/30 to-pink-500/30 text-white border border-purple-500/50',
                    'my-bets': 'bg-gradient-to-r from-yellow-500/30 to-orange-500/30 text-white border border-yellow-500/50',
                  };

                  // Hide best-ev tab when no data
                  if (view === 'best-ev' && (!mlStats?.is_trained || bestOpportunities.length === 0)) return null;

                  return (
                    <button
                      key={view}
                      onClick={() => setGamesView(view)}
                      className={`px-4 py-2 rounded-lg transition-all duration-200 ${
                        gamesView === view
                          ? activeClass[view]
                          : 'bg-white/5 text-gray-400 hover:text-white hover:bg-white/10 border border-white/10'
                      }`}
                    >
                      {labels[view]}
                    </button>
                  );
                })}
              </div>
              {gamesView !== 'best-ev' && gamesView !== 'bankroll-sim' && gamesView !== 'my-bets' && gamesView !== 'daily-reports' && (
                <div className="text-sm text-gray-400">💡 Click any game to view detailed betting lines</div>
              )}
            </div>
          </div>

          <div className="overflow-x-auto">
            {gamesView === 'daily-reports' && <DailyReports />}

            {gamesView === 'bankroll-sim' && <BankrollSimulator />}

            {gamesView === 'my-bets' && <MyBets />}

            {gamesView === 'best-ev' && (
              bestOpportunities.length > 0 ? (
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-gray-700/50">
                      {['Game', 'Market', 'Outcome', 'Current Line', 'Predicted Movement', 'Direction', 'Confidence', 'Bookmaker', 'EV Score'].map((h, i) => (
                        <th key={h} className={`py-4 px-4 text-gray-400 font-semibold uppercase text-xs tracking-wider ${i <= 2 || i === 7 ? 'text-left' : 'text-center'}`}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {bestOpportunities.map((opp, index) => (
                      <tr key={index} className="border-b border-gray-700/30 hover:bg-white/5 transition-all duration-200" style={{ animationDelay: `${index * 30}ms` }}>
                        <td className="py-4 px-4">
                          <span className="font-medium text-white text-sm">{opp.away_team} @ {opp.home_team}</span>
                          <div className="text-xs text-gray-500 mt-1">
                            {new Date(opp.commence_time).toLocaleString('en-US', { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' })}
                          </div>
                        </td>
                        <td className="py-4 px-4">
                          <span className="px-2 py-1 rounded text-xs font-medium bg-blue-500/20 text-blue-400 border border-blue-500/30">{opp.market_type}</span>
                        </td>
                        <td className="py-4 px-4"><span className="text-sm font-medium text-white">{opp.outcome_name}</span></td>
                        <td className="text-center py-4 px-4"><span className="text-sm font-mono text-blue-400">{opp.current_line}</span></td>
                        <td className="text-center py-4 px-4">
                          <div className="flex items-center justify-center gap-1">
                            <span className={`text-sm font-mono font-bold ${opp.predicted_movement > 0 ? 'text-green-400' : 'text-red-400'}`}>
                              {opp.predicted_movement > 0 ? '+' : ''}{opp.predicted_movement.toFixed(3)}
                            </span>
                            {opp.was_constrained && <span className="text-yellow-400 text-xs" title="Prediction capped to realistic range">⚠️</span>}
                          </div>
                        </td>
                        <td className="text-center py-4 px-4">
                          <span className={`px-2 py-1 rounded text-xs font-medium border ${opp.predicted_direction === 'UP' ? 'bg-green-500/20 text-green-400 border-green-500/30' : opp.predicted_direction === 'DOWN' ? 'bg-red-500/20 text-red-400 border-red-500/30' : 'bg-gray-500/20 text-gray-400 border-gray-500/30'}`}>
                            {opp.predicted_direction}
                          </span>
                        </td>
                        <td className="text-center py-4 px-4"><span className="text-sm font-medium text-cyan-400">{(opp.confidence * 100).toFixed(0)}%</span></td>
                        <td className="py-4 px-4"><span className="text-sm text-gray-300">{opp.bookmaker_name}</span></td>
                        <td className="text-center py-4 px-4">
                          <span className="px-2 py-1 rounded text-sm font-bold bg-green-500/20 text-green-400 border border-green-500/30">{opp.ev_score.toFixed(2)}</span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <div className="text-center py-12 text-gray-400">
                  <p className="text-lg mb-2">No +EV opportunities available</p>
                  <p className="text-sm">Opportunities appear when the ML model predicts favorable betting conditions</p>
                </div>
              )
            )}

            {(gamesView === 'recent' || gamesView === 'history') && (
              <>
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-gray-700/50">
                      {['Game', 'Time', 'Avg CLV', 'Snapshots', 'Closing Lines', 'Status'].map((h, i) => (
                        <th key={h} className={`py-4 px-4 text-gray-400 font-semibold uppercase text-xs tracking-wider ${i === 0 || i === 1 ? 'text-left' : 'text-center'}`}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {displayedGames.map((game, index) => (
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
                                  <span className={`font-bold ${game.winner === 'away' ? 'text-green-400' : 'text-gray-300'}`}>{game.away_score}</span>
                                  <span className="text-gray-500"> - </span>
                                  <span className={`font-bold ${game.winner === 'home' ? 'text-green-400' : 'text-gray-300'}`}>{game.home_score}</span>
                                </div>
                              )}
                            </div>
                          </td>
                          <td className="py-4 px-4 text-gray-300 text-sm">
                            {new Date(game.commence_time).toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}
                          </td>
                          <td className="text-center py-4 px-4">
                            {game.avg_clv !== null ? (
                              <span className={`font-bold text-lg ${game.avg_clv > 0 ? 'text-clv-positive-400' : 'text-clv-negative-400'}`}>
                                {game.avg_clv > 0 ? '+' : ''}{game.avg_clv.toFixed(2)}%
                              </span>
                            ) : (
                              <span className="text-gray-500 text-sm">N/A</span>
                            )}
                          </td>
                          <td className="text-center py-4 px-4">
                            <span className="inline-flex items-center px-2 py-1 rounded-md text-sm font-medium bg-blue-500/20 text-blue-400 border border-blue-500/30">{game.snapshots_count}</span>
                          </td>
                          <td className="text-center py-4 px-4">
                            <span className="inline-flex items-center px-2 py-1 rounded-md text-sm font-medium bg-purple-500/20 text-purple-400 border border-purple-500/30">{game.closing_lines_count}</span>
                          </td>
                          <td className="text-center py-4 px-4">
                            <span className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-medium ${game.completed ? 'bg-gray-700/50 text-gray-300 border border-gray-600/50' : 'bg-green-500/20 text-green-400 border border-green-500/30'}`}>
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
              </>
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
