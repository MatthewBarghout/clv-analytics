import React, { useCallback, useEffect, useMemo, useState } from 'react';

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
  sport_key?: string | null;
}

interface BestEVOpportunitiesProps {
  startingBankroll?: number;
  onAddToBets?: (opp: EVOpportunity) => void;
}

type SportFilter = 'all' | 'nba' | 'mlb';

const API_BASE = 'http://localhost:8000/api';

const SPORT_META: Record<string, { label: string; color: string; bg: string; border: string; accent: string }> = {
  basketball_nba: { label: 'NBA', color: 'text-orange-400', bg: 'bg-orange-500/15', border: 'border-orange-500/30', accent: 'bg-orange-500/25' },
  baseball_mlb:   { label: 'MLB', color: 'text-blue-400',   bg: 'bg-blue-500/15',   border: 'border-blue-500/30',   accent: 'bg-blue-500/25'   },
};

function kellyStake(bankroll: number, decimalOdds: number, winProb: number): number {
  const b = decimalOdds - 1;
  const q = 1 - winProb;
  const f = (b * winProb - q) / b;
  return Math.round(bankroll * Math.max(0, f * 0.25) * 100) / 100;
}

function formatGameTime(isoString: string): string {
  return new Date(isoString).toLocaleString('en-US', {
    weekday: 'short', month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit',
  });
}

// ── Deduplication ──────────────────────────────────────────────────────────
// For each unique (game_id, market_type, outcome_name), keep the row with the
// highest ev_score (multiple bookmakers / snapshots produce duplicates).
function deduplicateOpportunities(opps: EVOpportunity[]): EVOpportunity[] {
  const best = new Map<string, EVOpportunity>();
  for (const opp of opps) {
    const key = `${opp.game_id}|${opp.market_type}|${opp.outcome_name}`;
    const existing = best.get(key);
    if (!existing || opp.ev_score > existing.ev_score) {
      best.set(key, opp);
    }
  }
  return Array.from(best.values()).sort((a, b) => b.ev_score - a.ev_score);
}

// ── Pick card ──────────────────────────────────────────────────────────────

interface PickCardProps {
  opp: EVOpportunity;
  bankroll: number;
  onAddToBets?: (opp: EVOpportunity) => void;
}

const PickCard = React.memo(function PickCard({ opp, bankroll, onAddToBets }: PickCardProps) {
  const winProb = opp.confidence * (opp.predicted_direction === 'DOWN' ? 1 : 0.9);
  const stake = kellyStake(bankroll, 2.0, winProb);

  const evTier =
    opp.ev_score >= 5 ? { label: 'Elite', cls: 'text-green-400 bg-green-500/20 border-green-500/40' } :
    opp.ev_score >= 3 ? { label: 'Strong', cls: 'text-emerald-400 bg-emerald-500/15 border-emerald-500/30' } :
                        { label: 'Good',   cls: 'text-yellow-400 bg-yellow-500/15 border-yellow-500/30' };

  const marketColor =
    opp.market_type === 'h2h'     ? 'text-purple-400 bg-purple-500/15 border-purple-500/30' :
    opp.market_type === 'spreads' ? 'text-cyan-400 bg-cyan-500/15 border-cyan-500/30' :
                                    'text-sky-400 bg-sky-500/15 border-sky-500/30';

  return (
    <div className="bg-white/5 rounded-xl border border-white/10 p-4 hover:bg-white/8 transition-colors">
      {/* Top row: game + EV badge */}
      <div className="flex items-start justify-between gap-3 mb-3">
        <div>
          <div className="font-semibold text-white text-sm leading-tight">
            {opp.away_team} @ {opp.home_team}
          </div>
          <div className="text-xs text-gray-500 mt-0.5">{formatGameTime(opp.commence_time)}</div>
        </div>
        <span className={`shrink-0 px-2 py-0.5 rounded text-xs font-bold border ${evTier.cls}`}>
          EV {opp.ev_score.toFixed(2)} · {evTier.label}
        </span>
      </div>

      {/* Mid row: market / outcome / line */}
      <div className="flex flex-wrap gap-2 items-center mb-3">
        <span className={`px-2 py-0.5 rounded text-xs font-medium border ${marketColor}`}>
          {opp.market_type.toUpperCase()}
        </span>
        <span className="text-white font-medium text-sm">{opp.outcome_name}</span>
        <span className="text-gray-400 font-mono text-xs">{opp.current_line}</span>
        {opp.was_constrained && (
          <span className="text-yellow-400 text-xs" title="Prediction capped">⚠ capped</span>
        )}
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-3 gap-3 text-center mb-3">
        <div>
          <div className="text-[10px] text-gray-600 uppercase tracking-wider mb-0.5">Confidence</div>
          <div className="flex items-center justify-center gap-1.5">
            <div className="w-14 bg-gray-800 rounded-full h-1">
              <div className="bg-cyan-400 h-1 rounded-full" style={{ width: `${opp.confidence * 100}%` }} />
            </div>
            <span className="text-cyan-400 text-xs font-mono">{(opp.confidence * 100).toFixed(0)}%</span>
          </div>
        </div>
        <div>
          <div className="text-[10px] text-gray-600 uppercase tracking-wider mb-0.5">Pred. Move</div>
          <span className={`font-mono text-xs font-bold ${opp.predicted_movement > 0 ? 'text-green-400' : 'text-red-400'}`}>
            {opp.predicted_movement > 0 ? '+' : ''}{opp.predicted_movement.toFixed(3)}
          </span>
        </div>
        <div>
          <div className="text-[10px] text-gray-600 uppercase tracking-wider mb-0.5">Kelly Stake</div>
          <span className="text-amber-400 font-mono text-xs font-semibold">
            {stake > 0 ? `$${stake.toFixed(0)}` : '—'}
          </span>
        </div>
      </div>

      {/* Footer: book + action */}
      <div className="flex items-center justify-between">
        <span className="text-gray-500 text-xs">{opp.bookmaker_name}</span>
        {onAddToBets && (
          <button
            onClick={() => onAddToBets(opp)}
            className="px-2.5 py-1 bg-green-500/20 hover:bg-green-500/30 rounded border border-green-500/30 text-green-400 text-xs transition-all"
          >
            + Add Bet
          </button>
        )}
      </div>
    </div>
  );
});

// ── Sport section ──────────────────────────────────────────────────────────

interface SportSectionProps {
  sportKey: string;
  picks: EVOpportunity[];
  bankroll: number;
  onAddToBets?: (opp: EVOpportunity) => void;
}

const SportSection = React.memo(function SportSection({ sportKey, picks, bankroll, onAddToBets }: SportSectionProps) {
  const meta = SPORT_META[sportKey];
  const label = meta?.label ?? sportKey.toUpperCase();
  const headerCls = meta ? `${meta.color} ${meta.bg} ${meta.border}` : 'text-gray-400 bg-white/5 border-white/10';

  return (
    <div>
      <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full border text-xs font-bold mb-3 ${headerCls}`}>
        {label}
        <span className="opacity-60">{picks.length} pick{picks.length !== 1 ? 's' : ''}</span>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
        {picks.map((opp, i) => (
          <PickCard key={`${opp.game_id}-${opp.market_type}-${opp.outcome_name}-${i}`} opp={opp} bankroll={bankroll} onAddToBets={onAddToBets} />
        ))}
      </div>
    </div>
  );
});

// ── Main component ─────────────────────────────────────────────────────────

export const BestEVOpportunities = React.memo(function BestEVOpportunities({
  startingBankroll = 10000,
  onAddToBets,
}: BestEVOpportunitiesProps) {
  const [raw, setRaw] = useState<EVOpportunity[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null);
  const [savingPicks, setSavingPicks] = useState(false);
  const [savedMsg, setSavedMsg] = useState<string | null>(null);
  const [sportFilter, setSportFilter] = useState<SportFilter>('all');

  const today = new Date().toLocaleDateString('en-US', {
    weekday: 'long', month: 'long', day: 'numeric', year: 'numeric',
  });

  const fetchOpportunities = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(
        `${API_BASE}/ml/best-opportunities?today_only=true&limit=50&min_ev_score=2.0&min_confidence=0.62&min_hours_to_game=1.0`
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setRaw(Array.isArray(data) ? data : []);
      setLastRefresh(new Date());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch opportunities');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchOpportunities(); }, [fetchOpportunities]);

  const handleSaveDailyPicks = async () => {
    setSavingPicks(true);
    setSavedMsg(null);
    try {
      const res = await fetch(`${API_BASE}/ml/save-daily-picks`, { method: 'POST' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setSavedMsg("Today's picks saved for tracking. They'll settle after games complete.");
    } catch {
      setSavedMsg('Error saving picks. Please try again.');
    } finally {
      setSavingPicks(false);
    }
  };

  // Deduplicate then apply sport filter
  const deduped = useMemo(() => deduplicateOpportunities(raw), [raw]);

  const filtered = useMemo(() => {
    if (sportFilter === 'all') return deduped;
    const key = sportFilter === 'nba' ? 'basketball_nba' : 'baseball_mlb';
    return deduped.filter((o) => o.sport_key === key);
  }, [deduped, sportFilter]);

  // Group by sport_key for display
  const grouped = useMemo(() => {
    const map = new Map<string, EVOpportunity[]>();
    for (const opp of filtered) {
      const k = opp.sport_key ?? 'unknown';
      if (!map.has(k)) map.set(k, []);
      map.get(k)!.push(opp);
    }
    // Preferred order: NBA first, then MLB, then others
    const order = ['basketball_nba', 'baseball_mlb'];
    const keys = [...map.keys()].sort((a, b) => {
      const ai = order.indexOf(a);
      const bi = order.indexOf(b);
      if (ai === -1 && bi === -1) return a.localeCompare(b);
      if (ai === -1) return 1;
      if (bi === -1) return -1;
      return ai - bi;
    });
    return keys.map((k) => ({ sportKey: k, picks: map.get(k)! }));
  }, [filtered]);

  const nbaCnt = useMemo(() => deduped.filter((o) => o.sport_key === 'basketball_nba').length, [deduped]);
  const mlbCnt = useMemo(() => deduped.filter((o) => o.sport_key === 'baseball_mlb').length, [deduped]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-16">
        <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-green-500 mr-3" />
        <span className="text-gray-400">Scanning for today's best opportunities...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-16">
        <p className="text-red-400 mb-3">Error: {error}</p>
        <button onClick={fetchOpportunities} className="px-4 py-2 bg-red-500/20 hover:bg-red-500/30 rounded-lg border border-red-500/30 text-sm">
          Retry
        </button>
      </div>
    );
  }

  return (
    <div>
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-5">
        <div>
          <h3 className="text-lg font-bold text-white">Today's Best Bets</h3>
          <p className="text-sm text-gray-500">{today}</p>
          {lastRefresh && (
            <p className="text-xs text-gray-600 mt-0.5">Updated {lastRefresh.toLocaleTimeString()}</p>
          )}
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {/* Sport filter */}
          <div className="flex items-center gap-0.5 rounded-lg bg-white/5 border border-white/10 p-0.5 text-xs font-medium">
            <button
              onClick={() => setSportFilter('all')}
              className={`px-2.5 py-1 rounded transition-all ${sportFilter === 'all' ? 'bg-white/10 text-white' : 'text-gray-500 hover:text-gray-300'}`}
            >
              All ({deduped.length})
            </button>
            <button
              onClick={() => setSportFilter('nba')}
              className={`px-2.5 py-1 rounded transition-all ${sportFilter === 'nba' ? 'bg-orange-500/25 text-orange-300' : 'text-gray-500 hover:text-gray-300'}`}
            >
              NBA ({nbaCnt})
            </button>
            <button
              onClick={() => setSportFilter('mlb')}
              className={`px-2.5 py-1 rounded transition-all ${sportFilter === 'mlb' ? 'bg-blue-500/25 text-blue-300' : 'text-gray-500 hover:text-gray-300'}`}
            >
              MLB ({mlbCnt})
            </button>
          </div>

          <button
            onClick={fetchOpportunities}
            className="px-3 py-1.5 bg-white/10 hover:bg-white/15 rounded-lg border border-white/15 text-xs transition-all"
          >
            Refresh
          </button>
          <button
            onClick={handleSaveDailyPicks}
            disabled={savingPicks || deduped.length === 0}
            className="px-3 py-1.5 bg-green-500/20 hover:bg-green-500/30 disabled:opacity-40 rounded-lg border border-green-500/30 text-xs text-green-400 transition-all"
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

      {/* Threshold badges */}
      <div className="flex gap-2 flex-wrap text-xs mb-5">
        <span className="px-2 py-1 rounded bg-blue-500/15 text-blue-400 border border-blue-500/25">Min EV 2.0</span>
        <span className="px-2 py-1 rounded bg-purple-500/15 text-purple-400 border border-purple-500/25">Min Conf 62%</span>
        <span className="px-2 py-1 rounded bg-cyan-500/15 text-cyan-400 border border-cyan-500/25">Min 1h to game</span>
        {raw.length !== deduped.length && (
          <span className="px-2 py-1 rounded bg-gray-500/15 text-gray-400 border border-gray-500/25">
            {raw.length - deduped.length} duplicate{raw.length - deduped.length !== 1 ? 's' : ''} removed
          </span>
        )}
      </div>

      {filtered.length === 0 ? (
        <div className="text-center py-16">
          <p className="text-lg text-gray-400 mb-2">No opportunities found{sportFilter !== 'all' ? ` for ${sportFilter.toUpperCase()}` : ' for today'}</p>
          <p className="text-sm text-gray-600">
            {sportFilter !== 'all' ? 'Try switching to All sports.' : 'The model requires ≥62% confidence and ≥2.0 EV score. Check back as more odds are collected.'}
          </p>
        </div>
      ) : (
        <div className="space-y-6">
          {grouped.map(({ sportKey, picks }) => (
            <SportSection
              key={sportKey}
              sportKey={sportKey}
              picks={picks}
              bankroll={startingBankroll}
              onAddToBets={onAddToBets}
            />
          ))}
        </div>
      )}
    </div>
  );
});
