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
} from 'recharts';

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

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      setLoading(true);
      const [statsRes, bookmakersRes, historyRes, gamesRes] = await Promise.all([
        fetch(`${API_BASE}/stats`),
        fetch(`${API_BASE}/bookmakers`),
        fetch(`${API_BASE}/clv-history`),
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

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-900 text-white flex items-center justify-center">
        <div className="text-xl">Loading...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-900 text-white flex items-center justify-center">
        <div className="text-xl text-red-500">Error: {error}</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white p-8">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-4xl font-bold mb-8">CLV Analytics Dashboard</h1>

        {/* Summary Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="bg-gray-800 rounded-lg p-6">
            <h3 className="text-gray-400 text-sm font-medium mb-2">Mean CLV</h3>
            <p className={`text-3xl font-bold ${(stats?.mean_clv || 0) > 0 ? 'text-green-400' : 'text-red-400'}`}>
              {stats?.mean_clv?.toFixed(2) || 'N/A'}%
            </p>
          </div>

          <div className="bg-gray-800 rounded-lg p-6">
            <h3 className="text-gray-400 text-sm font-medium mb-2">Total Analyzed</h3>
            <p className="text-3xl font-bold text-blue-400">{stats?.total_analyzed || 0}</p>
          </div>

          <div className="bg-gray-800 rounded-lg p-6">
            <h3 className="text-gray-400 text-sm font-medium mb-2">Positive CLV</h3>
            <p className="text-3xl font-bold text-green-400">
              {stats?.positive_clv_percentage?.toFixed(1) || 0}%
            </p>
            <p className="text-sm text-gray-400 mt-1">
              ({stats?.positive_clv_count || 0} of {stats?.total_analyzed || 0})
            </p>
          </div>
        </div>

        {/* CLV Trend Chart */}
        <div className="bg-gray-800 rounded-lg p-6 mb-8">
          <h2 className="text-2xl font-bold mb-4">CLV Trend Over Time</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={history}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="date" stroke="#9CA3AF" />
              <YAxis stroke="#9CA3AF" />
              <Tooltip
                contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
                labelStyle={{ color: '#9CA3AF' }}
              />
              <Legend />
              <Line type="monotone" dataKey="avg_clv" stroke="#10B981" name="Avg CLV %" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Bookmaker Comparison */}
        <div className="bg-gray-800 rounded-lg p-6 mb-8">
          <h2 className="text-2xl font-bold mb-4">Bookmaker Comparison</h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={bookmakers}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="bookmaker_name" stroke="#9CA3AF" />
              <YAxis stroke="#9CA3AF" />
              <Tooltip
                contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
                labelStyle={{ color: '#9CA3AF' }}
              />
              <Legend />
              <Bar dataKey="avg_clv" fill="#3B82F6" name="Avg CLV %" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Recent Games Table */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-2xl font-bold mb-4">Recent Games</h2>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-700">
                  <th className="text-left py-3 px-4">Game</th>
                  <th className="text-left py-3 px-4">Time</th>
                  <th className="text-center py-3 px-4">Snapshots</th>
                  <th className="text-center py-3 px-4">Closing Lines</th>
                  <th className="text-center py-3 px-4">Status</th>
                </tr>
              </thead>
              <tbody>
                {games.map((game) => (
                  <tr key={game.game_id} className="border-b border-gray-700 hover:bg-gray-750">
                    <td className="py-3 px-4">
                      {game.away_team} @ {game.home_team}
                    </td>
                    <td className="py-3 px-4">
                      {new Date(game.commence_time).toLocaleString()}
                    </td>
                    <td className="text-center py-3 px-4">{game.snapshots_count}</td>
                    <td className="text-center py-3 px-4">{game.closing_lines_count}</td>
                    <td className="text-center py-3 px-4">
                      <span
                        className={`px-2 py-1 rounded text-xs ${
                          game.completed
                            ? 'bg-gray-700 text-gray-300'
                            : 'bg-green-900 text-green-300'
                        }`}
                      >
                        {game.completed ? 'Completed' : 'Upcoming'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
