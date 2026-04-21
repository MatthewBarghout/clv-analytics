import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { AnimatedCounter } from './components/AnimatedCounter';
import { GameDetailsModal } from './components/GameDetailsModal';
import { GameAnalysis } from './components/GameAnalysis';
import { MLStats } from './components/MLStats';
import { CLVOverview } from './components/CLVOverview';
import { BookmakerPerformance } from './components/BookmakerPerformance';
import { DailyReports } from './components/DailyReports';
import { BankrollSimulator } from './components/BankrollSimulator';
import { MyBets } from './components/MyBets';
import { BestEVOpportunities } from './components/BestEVOpportunities';
import { ArbOpportunities } from './components/ArbOpportunities';

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
  sport_key?: string | null;
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

const API_BASE = 'http://localhost:8000/api';

type View = 'overview' | 'best-ev' | 'games' | 'reports' | 'bankroll' | 'my-bets' | 'markets';
type SportFilter = 'all' | 'nba' | 'mlb';
type GamesTab = 'recent' | 'history';

// ── Sport helpers ──────────────────────────────────────────────────────────

const SPORT_LABELS: Record<string, { label: string; color: string; bg: string; border: string }> = {
  basketball_nba: { label: 'NBA', color: 'text-orange-400', bg: 'bg-orange-500/20', border: 'border-orange-500/40' },
  baseball_mlb:   { label: 'MLB', color: 'text-blue-400',   bg: 'bg-blue-500/20',   border: 'border-blue-500/40'   },
};

function SportBadge({ sportKey }: { sportKey?: string | null }) {
  if (!sportKey) return null;
  const s = SPORT_LABELS[sportKey];
  if (!s) return null;
  return (
    <span className={`inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-bold ${s.bg} ${s.color} border ${s.border}`}>
      {s.label}
    </span>
  );
}

function sportKeyFromFilter(filter: SportFilter): string | null {
  if (filter === 'nba') return 'basketball_nba';
  if (filter === 'mlb') return 'baseball_mlb';
  return null;
}

// ── Stat card ──────────────────────────────────────────────────────────────

interface StatCardProps {
  label: string;
  value: number;
  suffix?: string;
  decimals?: number;
  color: string;
  sub?: React.ReactNode;
}

function StatCard({ label, value, suffix, decimals, color, sub }: StatCardProps) {
  return (
    <div className="bg-white/5 rounded-lg p-4 border border-white/8">
      <div className="text-gray-500 text-xs font-medium uppercase tracking-wider mb-1.5">{label}</div>
      <AnimatedCounter value={value} suffix={suffix} decimals={decimals} className={`text-2xl font-bold ${color}`} />
      {sub && <div className="text-xs text-gray-600 mt-1">{sub}</div>}
    </div>
  );
}

// ── Nav tab ────────────────────────────────────────────────────────────────

interface NavTabProps {
  label: React.ReactNode;
  active: boolean;
  onClick: () => void;
}

function NavTab({ label, active, onClick }: NavTabProps) {
  return (
    <button
      onClick={onClick}
      className={`px-3 py-2 text-sm font-medium border-b-2 transition-colors duration-150 whitespace-nowrap ${
        active
          ? 'border-white text-white'
          : 'border-transparent text-gray-500 hover:text-gray-300'
      }`}
    >
      {label}
    </button>
  );
}

// ── Dashboard ──────────────────────────────────────────────────────────────

export default function Dashboard() {
  const [stats, setStats] = useState<CLVStats | null>(null);
  const [bookmakers, setBookmakers] = useState<BookmakerStats[]>([]);
  const [history, setHistory] = useState<CLVHistoryPoint[]>([]);
  const [games, setGames] = useState<GameWithCLV[]>([]);
  const [historyGames, setHistoryGames] = useState<GameWithCLV[]>([]);
  const [mlStats, setMlStats] = useState<MLModelStats | null>(null);
  const [featureImportance, setFeatureImportance] = useState<FeatureImportance[]>([]);
  const [arbCount, setArbCount] = useState<number>(0);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [timeRange, setTimeRange] = useState<'7d' | '30d' | '90d' | 'all'>('30d');
  const [view, setView] = useState<View>('overview');
  const [sport, setSport] = useState<SportFilter>('all');
  const [gamesTab, setGamesTab] = useState<GamesTab>('recent');
  const [selectedGame, setSelectedGame] = useState<GameWithCLV | null>(null);
  const [expandedGameId, setExpandedGameId] = useState<number | null>(null);

  // Track which tabs have been loaded to avoid redundant fetches
  const loadedTabs = React.useRef<Set<string>>(new Set());

  const fetchOverviewData = useCallback(async () => {
    try {
      setLoading(true);
      const [statsRes, bookmakersRes, historyRes, mlStatsRes, featureImportanceRes, arbRes] =
        await Promise.all([
          fetch(`${API_BASE}/stats`),
          fetch(`${API_BASE}/bookmakers`),
          fetch(`${API_BASE}/clv-history?time_range=${timeRange}`),
          fetch(`${API_BASE}/ml/stats`),
          fetch(`${API_BASE}/ml/feature-importance`),
          fetch(`${API_BASE}/arb-opportunities?min_spread=1.0&limit=5`),
        ]);

      if (!statsRes.ok || !bookmakersRes.ok || !historyRes.ok) {
        throw new Error('Failed to fetch data');
      }

      const [statsData, bookmakersData, historyData, mlStatsData, featureImportanceData, arbData] =
        await Promise.all([
          statsRes.json(),
          bookmakersRes.json(),
          historyRes.json(),
          mlStatsRes.ok ? mlStatsRes.json() : null,
          featureImportanceRes.ok ? featureImportanceRes.json() : [],
          arbRes.ok ? arbRes.json() : { total: 0 },
        ]);

      setStats(statsData);
      setBookmakers(bookmakersData);
      setHistory(historyData);
      setMlStats(mlStatsData);
      setFeatureImportance(featureImportanceData);
      setArbCount(arbData?.total ?? 0);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  }, [timeRange]);

  const fetchGamesData = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/games?limit=100`);
      if (!res.ok) throw new Error('Failed to fetch games');
      setGames(await res.json());
    } catch {
      // Non-fatal — games tab will show empty state
    }
  }, []);

  // Initial load: overview data only
  useEffect(() => {
    loadedTabs.current.add('overview');
    fetchOverviewData();
  }, [fetchOverviewData]);

  // Lazy-load when switching to games tab for the first time
  useEffect(() => {
    if (view === 'games' && !loadedTabs.current.has('games')) {
      loadedTabs.current.add('games');
      fetchGamesData();
    }
  }, [view, fetchGamesData]);

  // Global refresh: re-fetch overview + already-loaded tab data
  const fetchData = useCallback(async () => {
    fetchOverviewData();
    if (loadedTabs.current.has('games')) fetchGamesData();
  }, [fetchOverviewData, fetchGamesData]);

  useEffect(() => {
    if (gamesTab === 'history') {
      setHistoryGames(games.filter((g) => g.completed && g.avg_clv !== null));
    }
  }, [gamesTab, games]);

  // Sport-filtered game lists
  const sportKey = sportKeyFromFilter(sport);
  const recentGames = useMemo(
    () => games.filter((g) => !g.completed && (sportKey ? g.sport_key === sportKey : true)),
    [games, sportKey]
  );
  const completedGames = useMemo(
    () => historyGames.filter((g) => sportKey ? g.sport_key === sportKey : true),
    [historyGames, sportKey]
  );
  const displayedGames = gamesTab === 'recent' ? recentGames : completedGames;

  const handleGameClick = useCallback((game: GameWithCLV) => setSelectedGame(game), []);
  const handleCloseModal = useCallback(() => setSelectedGame(null), []);

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-950 text-white flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-blue-500 mx-auto mb-4"></div>
          <div className="text-gray-400">Loading analytics...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-950 text-white flex items-center justify-center">
        <div className="text-center max-w-sm">
          <div className="text-red-400 text-lg mb-3">Failed to load</div>
          <div className="text-gray-500 text-sm mb-4">{error}</div>
          <button onClick={fetchData} className="px-4 py-2 bg-red-500/20 hover:bg-red-500/30 rounded-lg border border-red-500/30 text-sm transition-all">
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      {/* ── Header ─────────────────────────────────────────────────────── */}
      <header className="border-b border-white/8 bg-gray-950 sticky top-0 z-50">
        <div className="max-w-screen-2xl mx-auto px-6 flex items-center justify-between">
          <div className="flex items-center gap-6 overflow-x-auto">
            <h1 className="text-sm font-semibold text-white tracking-tight py-3 shrink-0">CLV Analytics</h1>
            <div className="flex items-center">
              <NavTab label="Overview" active={view === 'overview'} onClick={() => setView('overview')} />
              {mlStats?.is_trained && (
                <NavTab label="Best EV+" active={view === 'best-ev'} onClick={() => setView('best-ev')} />
              )}
              <NavTab label="Games" active={view === 'games'} onClick={() => setView('games')} />
              <NavTab label="Reports" active={view === 'reports'} onClick={() => setView('reports')} />
              <NavTab label="Bankroll" active={view === 'bankroll'} onClick={() => setView('bankroll')} />
              <NavTab label="My Bets" active={view === 'my-bets'} onClick={() => setView('my-bets')} />
              <NavTab
                label={
                  <span className="flex items-center gap-1.5">
                    Markets
                    {arbCount > 0 && (
                      <span className="relative flex h-2 w-2">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75" />
                        <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500" />
                      </span>
                    )}
                  </span>
                }
                active={view === 'markets'}
                onClick={() => setView('markets')}
              />
            </div>
          </div>

          <button
            onClick={fetchData}
            className="text-xs text-gray-500 hover:text-gray-300 transition-colors py-3 shrink-0 ml-4"
          >
            Refresh
          </button>
        </div>
      </header>

      {/* ── Main content ────────────────────────────────────────────────── */}
      <main className="max-w-screen-2xl mx-auto px-6 py-4">

        {/* OVERVIEW ──────────────────────────────────────────────────────── */}
        {view === 'overview' && (
          <div className="space-y-4">
            {/* Stat cards */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
              <StatCard
                label="Mean CLV"
                value={stats?.mean_clv || 0}
                suffix="%"
                decimals={2}
                color={(stats?.mean_clv || 0) > 0 ? 'text-green-400' : 'text-red-400'}
              />
              <StatCard
                label="Median CLV"
                value={stats?.median_clv || 0}
                suffix="%"
                decimals={2}
                color={(stats?.median_clv || 0) > 0 ? 'text-green-400' : 'text-red-400'}
              />
              <StatCard
                label="Positive CLV Rate"
                value={stats?.positive_clv_percentage || 0}
                suffix="%"
                decimals={1}
                color="text-green-400"
                sub={`${stats?.positive_clv_count || 0} of ${stats?.total_analyzed || 0} bets`}
              />
              <StatCard
                label="Total Analyzed"
                value={stats?.total_analyzed || 0}
                color="text-blue-400"
                sub="Betting opportunities"
              />
            </div>

            <MLStats mlStats={mlStats} featureImportance={featureImportance} />
            <CLVOverview stats={stats} history={history} timeRange={timeRange} onTimeRangeChange={setTimeRange} />
            <BookmakerPerformance bookmakers={bookmakers} />
          </div>
        )}

        {/* BEST EV+ ──────────────────────────────────────────────────────── */}
        {view === 'best-ev' && (
          <div className="bg-white/4 rounded-xl border border-white/8 p-5">
            <BestEVOpportunities startingBankroll={10000} />
          </div>
        )}

        {/* GAMES ─────────────────────────────────────────────────────────── */}
        {view === 'games' && (
          <div className="space-y-4">
            {/* Controls row */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-1 border-b border-white/10">
                {(['recent', 'history'] as GamesTab[]).map((t) => (
                  <button
                    key={t}
                    onClick={() => setGamesTab(t)}
                    className={`px-3 py-2 text-sm font-medium border-b-2 -mb-px transition-colors ${
                      gamesTab === t
                        ? 'border-white text-white'
                        : 'border-transparent text-gray-500 hover:text-gray-300'
                    }`}
                  >
                    {t === 'recent' ? `Upcoming (${recentGames.length})` : `Completed (${completedGames.length})`}
                  </button>
                ))}
              </div>

              {/* Sport filter */}
              <div className="flex items-center gap-1 rounded-md bg-white/5 border border-white/10 p-0.5 text-xs font-medium">
                {(['all', 'nba', 'mlb'] as SportFilter[]).map((s) => (
                  <button
                    key={s}
                    onClick={() => setSport(s)}
                    className={`px-2.5 py-1 rounded transition-all ${
                      sport === s
                        ? s === 'nba'
                          ? 'bg-orange-500/25 text-orange-300 shadow-sm'
                          : s === 'mlb'
                          ? 'bg-blue-500/25 text-blue-300 shadow-sm'
                          : 'bg-white/10 text-white shadow-sm'
                        : 'text-gray-500 hover:text-gray-300'
                    }`}
                  >
                    {s === 'all' ? 'All' : s.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>

            <GamesTable
              games={displayedGames}
              mode={gamesTab}
              expandedGameId={expandedGameId}
              onGameClick={handleGameClick}
              onToggleExpand={(id) => setExpandedGameId(expandedGameId === id ? null : id)}
            />
          </div>
        )}

        {/* REPORTS ───────────────────────────────────────────────────────── */}
        {view === 'reports' && (
          <div className="bg-white/4 rounded-xl border border-white/8 p-5">
            <DailyReports />
          </div>
        )}

        {/* BANKROLL ──────────────────────────────────────────────────────── */}
        {view === 'bankroll' && (
          <div className="bg-white/4 rounded-xl border border-white/8 p-5">
            <BankrollSimulator />
          </div>
        )}

        {/* MY BETS ───────────────────────────────────────────────────────── */}
        {view === 'my-bets' && (
          <div className="bg-white/4 rounded-xl border border-white/8 p-5">
            <MyBets />
          </div>
        )}

        {/* MARKETS ───────────────────────────────────────────────────────── */}
        {view === 'markets' && (
          <div className="bg-white/4 rounded-xl border border-white/8 p-5">
            <ArbOpportunities />
          </div>
        )}
      </main>

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
  );
}

// ── GamesTable ─────────────────────────────────────────────────────────────

interface GamesTableProps {
  games: GameWithCLV[];
  mode: GamesTab;
  expandedGameId: number | null;
  onGameClick: (game: GameWithCLV) => void;
  onToggleExpand: (id: number) => void;
}

const GamesTable = React.memo(function GamesTable({
  games,
  mode,
  expandedGameId,
  onGameClick,
  onToggleExpand,
}: GamesTableProps) {
  if (games.length === 0) {
    return (
      <div className="bg-white/4 rounded-2xl border border-white/10 p-12 text-center">
        <p className="text-gray-400">No games to show.</p>
      </div>
    );
  }

  return (
    <div className="bg-white/4 rounded-xl border border-white/8 overflow-hidden">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-white/8">
            <th className="text-left py-2.5 px-4 text-gray-500 font-medium uppercase text-xs tracking-wider">Sport</th>
            <th className="text-left py-2.5 px-4 text-gray-500 font-medium uppercase text-xs tracking-wider">Game</th>
            <th className="text-left py-2.5 px-4 text-gray-500 font-medium uppercase text-xs tracking-wider">Time</th>
            <th className="text-center py-2.5 px-4 text-gray-500 font-medium uppercase text-xs tracking-wider">Avg CLV</th>
            <th className="text-center py-2.5 px-4 text-gray-500 font-medium uppercase text-xs tracking-wider">Snapshots</th>
            <th className="text-center py-2.5 px-4 text-gray-500 font-medium uppercase text-xs tracking-wider">Closing Lines</th>
            <th className="text-center py-2.5 px-4 text-gray-500 font-medium uppercase text-xs tracking-wider">Status</th>
          </tr>
        </thead>
        <tbody>
          {games.map((game) => (
            <React.Fragment key={game.game_id}>
              <tr
                onClick={() => mode === 'history' ? onToggleExpand(game.game_id) : onGameClick(game)}
                className="border-b border-white/5 hover:bg-white/3 transition-colors cursor-pointer"
              >
                <td className="py-2.5 px-4">
                  <SportBadge sportKey={game.sport_key} />
                </td>
                <td className="py-2.5 px-4">
                  <div className="font-medium text-white text-sm">{game.away_team} @ {game.home_team}</div>
                  {game.home_score != null && game.away_score != null && (
                    <div className="text-xs mt-0.5 text-gray-500">
                      Final:{' '}
                      <span className={game.winner === 'away' ? 'text-green-400 font-semibold' : 'text-gray-400'}>{game.away_score}</span>
                      {' – '}
                      <span className={game.winner === 'home' ? 'text-green-400 font-semibold' : 'text-gray-400'}>{game.home_score}</span>
                    </div>
                  )}
                </td>
                <td className="py-2.5 px-4 text-gray-400 text-xs whitespace-nowrap">
                  {new Date(game.commence_time).toLocaleString('en-US', {
                    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
                  })}
                </td>
                <td className="text-center py-2.5 px-4">
                  {game.avg_clv != null ? (
                    <span className={`text-sm font-semibold ${game.avg_clv > 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {game.avg_clv > 0 ? '+' : ''}{game.avg_clv.toFixed(2)}%
                    </span>
                  ) : (
                    <span className="text-gray-700 text-xs">—</span>
                  )}
                </td>
                <td className="text-center py-2.5 px-4 text-gray-400 text-xs">
                  {game.snapshots_count}
                </td>
                <td className="text-center py-2.5 px-4 text-gray-400 text-xs">
                  {game.closing_lines_count}
                </td>
                <td className="text-center py-2.5 px-4">
                  <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                    game.completed
                      ? 'text-gray-500'
                      : 'text-green-400'
                  }`}>
                    {game.completed ? 'Final' : 'Upcoming'}
                  </span>
                </td>
              </tr>
              {mode === 'history' && expandedGameId === game.game_id && (
                <tr className="bg-gray-900/50">
                  <td colSpan={7} className="p-6">
                    <GameAnalysis gameId={game.game_id} />
                  </td>
                </tr>
              )}
            </React.Fragment>
          ))}
        </tbody>
      </table>
    </div>
  );
});
