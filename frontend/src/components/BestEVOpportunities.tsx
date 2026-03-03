import React, { useCallback, useEffect, useState } from 'react';

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

interface BestEVOpportunitiesProps {
  startingBankroll?: number;
  onAddToBets?: (opp: EVOpportunity) => void;
}

const API_BASE = 'http://localhost:8000/api';

// Kelly criterion: fraction of bankroll to bet given edge and odds
function kellyStake(bankroll: number, decimalOdds: number, winProb: number): number {
  const b = decimalOdds - 1;
  const q = 1 - winProb;
  const f = (b * winProb - q) / b;
  // Quarter-Kelly for conservatism
  const quarterKelly = Math.max(0, f * 0.25);
  return Math.round(bankroll * quarterKelly * 100) / 100;
}

export const BestEVOpportunities = React.memo(function BestEVOpportunities({
  startingBankroll = 10000,
  onAddToBets,
}: BestEVOpportunitiesProps) {
  const [opportunities, setOpportunities] = useState<EVOpportunity[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null);
  const [savingPicks, setSavingPicks] = useState(false);
  const [savedMsg, setSavedMsg] = useState<string | null>(null);

  const today = new Date().toLocaleDateString('en-US', {
    weekday: 'long', month: 'long', day: 'numeric', year: 'numeric',
  });

  const fetchOpportunities = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(
        `${API_BASE}/ml/best-opportunities?today_only=true&limit=20&min_ev_score=2.0&min_confidence=0.62&min_hours_to_game=1.0`
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setOpportunities(Array.isArray(data) ? data : []);
      setLastRefresh(new Date());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch opportunities');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchOpportunities();
  }, [fetchOpportunities]);

  const handleSaveDailyPicks = async () => {
    setSavingPicks(true);
    setSavedMsg(null);
    try {
      const res = await fetch(`${API_BASE}/ml/save-daily-picks`, { method: 'POST' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setSavedMsg("Today's picks have been saved for tracking. They'll be settled after games complete.");
    } catch (err) {
      setSavedMsg('Error saving picks. Please try again.');
    } finally {
      setSavingPicks(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-green-500 mr-3"></div>
        <span className="text-gray-400">Scanning for today's best opportunities...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-12">
        <p className="text-red-400 mb-2">Error: {error}</p>
        <button
          onClick={fetchOpportunities}
          className="px-4 py-2 bg-red-500/20 hover:bg-red-500/30 rounded-lg border border-red-500/30 text-sm"
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div>
      {/* Header bar */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-6 gap-3">
        <div>
          <h3 className="text-lg font-bold text-white">Today's Best Bets</h3>
          <p className="text-sm text-gray-400">{today}</p>
          {lastRefresh && (
            <p className="text-xs text-gray-500 mt-1">
              Last updated: {lastRefresh.toLocaleTimeString()}
            </p>
          )}
        </div>
        <div className="flex items-center gap-3">
          {/* Filter badges */}
          <div className="flex gap-2 flex-wrap text-xs">
            <span className="px-2 py-1 rounded bg-blue-500/20 text-blue-300 border border-blue-500/30">
              Min EV: 2.0
            </span>
            <span className="px-2 py-1 rounded bg-purple-500/20 text-purple-300 border border-purple-500/30">
              Min Conf: 62%
            </span>
            <span className="px-2 py-1 rounded bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">
              Min 1h to game
            </span>
          </div>
          <button
            onClick={fetchOpportunities}
            className="px-3 py-1.5 bg-white/10 hover:bg-white/20 rounded-lg border border-white/20 text-sm transition-all"
          >
            Refresh
          </button>
          <button
            onClick={handleSaveDailyPicks}
            disabled={savingPicks || opportunities.length === 0}
            className="px-3 py-1.5 bg-green-500/20 hover:bg-green-500/30 disabled:opacity-40 rounded-lg border border-green-500/30 text-sm text-green-400 transition-all"
          >
            {savingPicks ? 'Saving...' : 'Save for Tracking'}
          </button>
        </div>
      </div>

      {savedMsg && (
        <div className="mb-4 px-4 py-2 rounded-lg bg-green-500/10 border border-green-500/30 text-green-300 text-sm">
          {savedMsg}
        </div>
      )}

      {opportunities.length === 0 ? (
        <div className="text-center py-12">
          <div className="text-4xl mb-4">🔍</div>
          <p className="text-lg text-gray-400 mb-2">No high-quality opportunities found for today</p>
          <p className="text-sm text-gray-500">
            The model requires ≥62% confidence and ≥2.0 EV score. Check back as more odds are collected.
          </p>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-700/50">
                {['Game', 'Time', 'Market', 'Outcome', 'Current Line', 'Pred. Move', 'Confidence', 'EV Score', 'Kelly Stake', 'Book', ''].map((h, i) => (
                  <th
                    key={h || i}
                    className={`py-3 px-3 text-gray-400 font-semibold uppercase text-xs tracking-wider ${
                      i <= 1 || i === 3 || i === 9 ? 'text-left' : 'text-center'
                    }`}
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {opportunities.map((opp, index) => {
                // Estimate win probability from confidence (rough proxy)
                const winProb = opp.confidence * (opp.predicted_direction === 'DOWN' ? 1 : 0.9);
                // Parse decimal odds from current_line or default to 2.0
                const decimalOdds = 2.0;
                const stake = kellyStake(startingBankroll, decimalOdds, winProb);

                const evColor =
                  opp.ev_score >= 5 ? 'text-green-400 bg-green-500/20 border-green-500/30' :
                  opp.ev_score >= 3 ? 'text-emerald-400 bg-emerald-500/15 border-emerald-500/25' :
                  'text-yellow-400 bg-yellow-500/15 border-yellow-500/25';

                return (
                  <tr
                    key={index}
                    className="border-b border-gray-700/30 hover:bg-white/5 transition-all duration-150"
                    style={{ animationDelay: `${index * 25}ms` }}
                  >
                    <td className="py-3 px-3">
                      <span className="font-medium text-white">{opp.away_team} @ {opp.home_team}</span>
                    </td>
                    <td className="py-3 px-3 text-gray-400 text-xs whitespace-nowrap">
                      {new Date(opp.commence_time).toLocaleString('en-US', {
                        month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit',
                      })}
                    </td>
                    <td className="text-center py-3 px-3">
                      <span className="px-1.5 py-0.5 rounded text-xs font-medium bg-blue-500/20 text-blue-400 border border-blue-500/30">
                        {opp.market_type}
                      </span>
                    </td>
                    <td className="py-3 px-3 text-white font-medium">{opp.outcome_name}</td>
                    <td className="text-center py-3 px-3 font-mono text-blue-300">{opp.current_line}</td>
                    <td className="text-center py-3 px-3">
                      <div className="flex items-center justify-center gap-1">
                        <span className={`font-mono font-bold text-xs ${opp.predicted_movement > 0 ? 'text-green-400' : 'text-red-400'}`}>
                          {opp.predicted_movement > 0 ? '+' : ''}{opp.predicted_movement.toFixed(3)}
                        </span>
                        {opp.was_constrained && (
                          <span className="text-yellow-400 text-xs" title="Prediction was capped">⚠</span>
                        )}
                      </div>
                    </td>
                    <td className="text-center py-3 px-3">
                      <div className="flex items-center justify-center">
                        <div className="w-16 bg-gray-700 rounded-full h-1.5 mr-2">
                          <div
                            className="bg-cyan-400 h-1.5 rounded-full"
                            style={{ width: `${opp.confidence * 100}%` }}
                          />
                        </div>
                        <span className="text-cyan-400 text-xs">{(opp.confidence * 100).toFixed(0)}%</span>
                      </div>
                    </td>
                    <td className="text-center py-3 px-3">
                      <span className={`px-2 py-0.5 rounded text-xs font-bold border ${evColor}`}>
                        {opp.ev_score.toFixed(2)}
                      </span>
                    </td>
                    <td className="text-center py-3 px-3">
                      <span className="text-xs text-amber-400 font-mono">
                        {stake > 0 ? `$${stake.toFixed(0)}` : '—'}
                      </span>
                    </td>
                    <td className="py-3 px-3 text-gray-300 text-xs">{opp.bookmaker_name}</td>
                    <td className="py-3 px-3">
                      {onAddToBets && (
                        <button
                          onClick={() => onAddToBets(opp)}
                          className="px-2 py-1 bg-green-500/20 hover:bg-green-500/30 rounded border border-green-500/30 text-green-400 text-xs transition-all whitespace-nowrap"
                        >
                          + Add Bet
                        </button>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p className="mt-3 text-xs text-gray-500 text-right">
            Showing {opportunities.length} opportunity{opportunities.length !== 1 ? 'ies' : ''} · Quarter-Kelly sizing shown
          </p>
        </div>
      )}
    </div>
  );
});
