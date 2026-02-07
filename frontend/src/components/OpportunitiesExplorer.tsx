import React, { useEffect, useState, useCallback } from 'react';

interface Opportunity {
  id: number;
  game_id: number;
  home_team: string;
  away_team: string;
  commence_time: string;
  bookmaker: string;
  market_type: string;
  outcome_name: string;
  point_line: number | null;
  entry_odds: number;
  closing_odds: number;
  clv_percentage: number;
  result: string | null;
  profit_loss: number | null;
  settled_at: string | null;
  status: string;
}

interface OpportunitiesResponse {
  opportunities: Opportunity[];
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
}

interface OpportunitiesExplorerProps {
  apiBase: string;
}

export function OpportunitiesExplorer({ apiBase }: OpportunitiesExplorerProps) {
  const [opportunities, setOpportunities] = useState<Opportunity[]>([]);
  const [loading, setLoading] = useState(true);
  const [total, setTotal] = useState(0);
  const [hasMore, setHasMore] = useState(false);

  // Filter state
  const [status, setStatus] = useState('all');
  const [minClv, setMinClv] = useState<number | null>(null);
  const [bookmaker, setBookmaker] = useState<string | null>(null);
  const [marketType, setMarketType] = useState<string | null>(null);
  const [sortBy, setSortBy] = useState('clv');
  const [limit] = useState(50);
  const [offset, setOffset] = useState(0);

  const fetchOpportunities = useCallback(async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams({
        status,
        sort_by: sortBy,
        limit: limit.toString(),
        offset: offset.toString(),
      });
      if (minClv !== null) params.append('min_clv', minClv.toString());
      if (bookmaker) params.append('bookmaker', bookmaker);
      if (marketType) params.append('market_type', marketType);

      const res = await fetch(`${apiBase}/ml/opportunities?${params}`);
      if (res.ok) {
        const data: OpportunitiesResponse = await res.json();
        setOpportunities(data.opportunities);
        setTotal(data.total);
        setHasMore(data.has_more);
      }
    } catch (err) {
      console.error('Error fetching opportunities:', err);
    } finally {
      setLoading(false);
    }
  }, [apiBase, status, minClv, bookmaker, marketType, sortBy, limit, offset]);

  useEffect(() => {
    fetchOpportunities();
  }, [fetchOpportunities]);

  // Reset offset when filters change
  useEffect(() => {
    setOffset(0);
  }, [status, minClv, bookmaker, marketType, sortBy]);

  const exportToCSV = () => {
    const headers = [
      'Game',
      'Date',
      'Bookmaker',
      'Market',
      'Outcome',
      'Entry Odds',
      'Closing Odds',
      'CLV %',
      'Result',
      'P&L',
    ];
    const rows = opportunities.map(opp => [
      `${opp.away_team} @ ${opp.home_team}`,
      new Date(opp.commence_time).toLocaleDateString(),
      opp.bookmaker,
      opp.market_type,
      opp.outcome_name,
      opp.entry_odds.toFixed(2),
      opp.closing_odds.toFixed(2),
      opp.clv_percentage.toFixed(2),
      opp.result || 'pending',
      opp.profit_loss?.toFixed(2) || '',
    ]);

    const csvContent = [
      headers.join(','),
      ...rows.map(row => row.join(','))
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `opportunities-${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
  };

  const formatOdds = (decimal: number) => {
    if (decimal >= 2) {
      return `+${Math.round((decimal - 1) * 100)}`;
    } else {
      return Math.round(-100 / (decimal - 1)).toString();
    }
  };

  return (
    <div className="space-y-6">
      {/* Filters */}
      <div className="bg-white/5 rounded-lg p-4 border border-white/10">
        <div className="flex flex-wrap gap-4 items-center">
          <div className="flex items-center gap-2">
            <label className="text-sm text-gray-400">Status:</label>
            <select
              value={status}
              onChange={(e) => setStatus(e.target.value)}
              className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-white text-sm"
            >
              <option value="all">All</option>
              <option value="pending">Pending</option>
              <option value="settled">Settled</option>
            </select>
          </div>

          <div className="flex items-center gap-2">
            <label className="text-sm text-gray-400">Min CLV:</label>
            <select
              value={minClv ?? ''}
              onChange={(e) => setMinClv(e.target.value ? Number(e.target.value) : null)}
              className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-white text-sm"
            >
              <option value="">Any</option>
              <option value={1}>1%+</option>
              <option value={2}>2%+</option>
              <option value={3}>3%+</option>
              <option value={5}>5%+</option>
            </select>
          </div>

          <div className="flex items-center gap-2">
            <label className="text-sm text-gray-400">Bookmaker:</label>
            <select
              value={bookmaker || ''}
              onChange={(e) => setBookmaker(e.target.value || null)}
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
              value={marketType || ''}
              onChange={(e) => setMarketType(e.target.value || null)}
              className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-white text-sm"
            >
              <option value="">All</option>
              <option value="h2h">Moneyline</option>
              <option value="spreads">Spreads</option>
              <option value="totals">Totals</option>
            </select>
          </div>

          <div className="flex items-center gap-2">
            <label className="text-sm text-gray-400">Sort:</label>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-white text-sm"
            >
              <option value="clv">CLV (High to Low)</option>
              <option value="date">Date (Recent)</option>
            </select>
          </div>

          <div className="flex-1" />

          <button
            onClick={exportToCSV}
            disabled={opportunities.length === 0}
            className="px-4 py-1.5 bg-blue-500/20 hover:bg-blue-500/30 text-blue-400 rounded border border-blue-500/50 text-sm disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Export CSV
          </button>
        </div>

        <div className="mt-3 text-sm text-gray-500">
          Showing {opportunities.length} of {total} opportunities
        </div>
      </div>

      {/* Results Table */}
      {loading ? (
        <div className="text-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-purple-500 mx-auto"></div>
          <p className="text-gray-400 mt-4">Loading opportunities...</p>
        </div>
      ) : opportunities.length === 0 ? (
        <div className="text-center py-12 text-gray-400">
          <p className="text-lg mb-2">No opportunities found</p>
          <p className="text-sm">Try adjusting your filters</p>
        </div>
      ) : (
        <div className="bg-white/5 rounded-lg border border-white/10 overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-gray-900">
                <tr className="border-b border-gray-700">
                  <th className="text-left py-3 px-4 text-gray-400 font-medium">Game</th>
                  <th className="text-left py-3 px-4 text-gray-400 font-medium">Date</th>
                  <th className="text-left py-3 px-4 text-gray-400 font-medium">Bookmaker</th>
                  <th className="text-left py-3 px-4 text-gray-400 font-medium">Market</th>
                  <th className="text-left py-3 px-4 text-gray-400 font-medium">Outcome</th>
                  <th className="text-center py-3 px-4 text-gray-400 font-medium">Entry</th>
                  <th className="text-center py-3 px-4 text-gray-400 font-medium">Close</th>
                  <th className="text-center py-3 px-4 text-gray-400 font-medium">CLV</th>
                  <th className="text-center py-3 px-4 text-gray-400 font-medium">Result</th>
                  <th className="text-right py-3 px-4 text-gray-400 font-medium">P&L</th>
                </tr>
              </thead>
              <tbody>
                {opportunities.map((opp) => (
                  <tr key={opp.id} className="border-b border-gray-800 hover:bg-white/5">
                    <td className="py-3 px-4 text-white">
                      <div className="max-w-[180px] truncate" title={`${opp.away_team} @ ${opp.home_team}`}>
                        {opp.away_team} @ {opp.home_team}
                      </div>
                    </td>
                    <td className="py-3 px-4 text-gray-400">
                      {new Date(opp.commence_time).toLocaleDateString()}
                    </td>
                    <td className="py-3 px-4 text-gray-300">
                      {opp.bookmaker}
                    </td>
                    <td className="py-3 px-4">
                      <span className="px-2 py-0.5 bg-purple-500/20 text-purple-400 rounded text-xs">
                        {opp.market_type.toUpperCase()}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-white">
                      <div className="max-w-[120px] truncate" title={opp.outcome_name}>
                        {opp.outcome_name}
                        {opp.point_line && (
                          <span className="text-gray-500 ml-1">
                            ({opp.point_line > 0 ? '+' : ''}{opp.point_line})
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="py-3 px-4 text-center font-mono text-blue-400">
                      {formatOdds(opp.entry_odds)}
                    </td>
                    <td className="py-3 px-4 text-center font-mono text-gray-400">
                      {formatOdds(opp.closing_odds)}
                    </td>
                    <td className="py-3 px-4 text-center">
                      <span className={`font-medium ${opp.clv_percentage >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {opp.clv_percentage >= 0 ? '+' : ''}{opp.clv_percentage.toFixed(1)}%
                      </span>
                    </td>
                    <td className="py-3 px-4 text-center">
                      {opp.result ? (
                        <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                          opp.result === 'win'
                            ? 'bg-green-500/20 text-green-400'
                            : opp.result === 'loss'
                            ? 'bg-red-500/20 text-red-400'
                            : 'bg-gray-500/20 text-gray-400'
                        }`}>
                          {opp.result.toUpperCase()}
                        </span>
                      ) : (
                        <span className="px-2 py-0.5 bg-yellow-500/20 text-yellow-400 rounded text-xs font-medium">
                          PENDING
                        </span>
                      )}
                    </td>
                    <td className="py-3 px-4 text-right font-medium">
                      {opp.profit_loss !== null ? (
                        <span className={opp.profit_loss >= 0 ? 'text-green-400' : 'text-red-400'}>
                          {opp.profit_loss >= 0 ? '+' : ''}${opp.profit_loss.toFixed(0)}
                        </span>
                      ) : (
                        <span className="text-gray-500">-</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          {(hasMore || offset > 0) && (
            <div className="flex justify-between items-center p-4 bg-gray-900/50">
              <button
                onClick={() => setOffset(Math.max(0, offset - limit))}
                disabled={offset === 0}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Previous
              </button>
              <span className="text-gray-400 text-sm">
                Page {Math.floor(offset / limit) + 1} of {Math.ceil(total / limit)}
              </span>
              <button
                onClick={() => setOffset(offset + limit)}
                disabled={!hasMore}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Next
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
