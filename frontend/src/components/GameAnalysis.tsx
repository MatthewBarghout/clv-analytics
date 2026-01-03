import React, { useEffect, useState } from 'react';
import { GlassCard } from './GlassCard';

interface GameAnalysisProps {
  gameId: number;
}

interface BestOpportunity {
  bookmaker: string;
  market_type: string;
  outcome: string;
  entry_odds: number;
  closing_odds: number;
  clv: number;
  timestamp: string;
}

interface MarketStats {
  avg_clv: number;
  count: number;
}

interface BookmakerStats {
  avg_clv: number;
  count: number;
}

interface AnalysisData {
  game_id: number;
  total_opportunities: number;
  avg_clv: number;
  by_market: Record<string, MarketStats>;
  by_bookmaker: Record<string, BookmakerStats>;
  best_opportunities: BestOpportunity[];
  positive_clv_count: number;
  positive_clv_percentage: number;
}

const API_BASE = 'http://localhost:8000/api';

export function GameAnalysis({ gameId }: GameAnalysisProps) {
  const [analysis, setAnalysis] = useState<AnalysisData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchAnalysis();
  }, [gameId]);

  const fetchAnalysis = async () => {
    try {
      setLoading(true);
      const res = await fetch(`${API_BASE}/games/${gameId}/analysis`);
      if (res.ok) {
        const data = await res.json();
        setAnalysis(data);
      }
    } catch (err) {
      console.error('Error fetching analysis:', err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-8">
        <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  if (!analysis || analysis.total_opportunities === 0) {
    return (
      <div className="text-center py-8 text-gray-400">
        <p>No analysis data available for this game</p>
      </div>
    );
  }

  const getMarketTypeName = (market: string) => {
    const names: Record<string, string> = {
      h2h: 'Moneyline',
      spreads: 'Point Spread',
      totals: 'Over/Under',
    };
    return names[market] || market;
  };

  return (
    <div className="space-y-6">
      {/* Summary */}
      <div>
        <h3 className="text-xl font-bold text-white mb-4">📊 Game Analysis Summary</h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <GlassCard className="p-4">
            <div className="text-sm text-gray-400 mb-1">Total Opportunities</div>
            <div className="text-2xl font-bold text-blue-400">{analysis.total_opportunities}</div>
          </GlassCard>
          <GlassCard className="p-4" gradient={analysis.avg_clv > 0 ? 'green' : 'red'}>
            <div className="text-sm text-gray-400 mb-1">Average CLV</div>
            <div className={`text-2xl font-bold ${analysis.avg_clv > 0 ? 'text-green-400' : 'text-red-400'}`}>
              {analysis.avg_clv > 0 ? '+' : ''}{analysis.avg_clv.toFixed(2)}%
            </div>
          </GlassCard>
          <GlassCard className="p-4" gradient="green">
            <div className="text-sm text-gray-400 mb-1">Positive CLV</div>
            <div className="text-2xl font-bold text-green-400">{analysis.positive_clv_count}</div>
          </GlassCard>
          <GlassCard className="p-4">
            <div className="text-sm text-gray-400 mb-1">Success Rate</div>
            <div className="text-2xl font-bold text-purple-400">
              {analysis.positive_clv_percentage.toFixed(1)}%
            </div>
          </GlassCard>
        </div>
      </div>

      {/* Best Opportunities */}
      {analysis.best_opportunities.length > 0 && (
        <div>
          <h3 className="text-xl font-bold text-white mb-4">🎯 Top 5 Best Opportunities</h3>
          <div className="space-y-2">
            {analysis.best_opportunities.map((opp, idx) => (
              <GlassCard key={idx} className="p-4" gradient={idx === 0 ? 'green' : undefined}>
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-white font-semibold">{opp.outcome}</span>
                      <span className="text-xs px-2 py-1 rounded bg-blue-500/20 text-blue-400 border border-blue-500/30">
                        {getMarketTypeName(opp.market_type)}
                      </span>
                      <span className="text-xs px-2 py-1 rounded bg-purple-500/20 text-purple-400 border border-purple-500/30">
                        {opp.bookmaker}
                      </span>
                    </div>
                    <div className="text-sm text-gray-400">
                      Entry: {opp.entry_odds.toFixed(2)} → Closing: {opp.closing_odds.toFixed(2)}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className={`text-2xl font-bold ${opp.clv > 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {opp.clv > 0 ? '+' : ''}{opp.clv.toFixed(2)}%
                    </div>
                    {idx === 0 && <div className="text-xs text-green-400 mt-1">Best Edge</div>}
                  </div>
                </div>
              </GlassCard>
            ))}
          </div>
        </div>
      )}

      {/* Breakdown by Market Type */}
      {Object.keys(analysis.by_market).length > 0 && (
        <div>
          <h3 className="text-xl font-bold text-white mb-4">📈 CLV by Market Type</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {Object.entries(analysis.by_market).map(([market, stats]) => (
              <GlassCard key={market} className="p-4" gradient={stats.avg_clv > 0 ? 'green' : 'red'}>
                <div className="text-sm text-gray-400 mb-2">{getMarketTypeName(market)}</div>
                <div className={`text-2xl font-bold mb-1 ${stats.avg_clv > 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {stats.avg_clv > 0 ? '+' : ''}{stats.avg_clv.toFixed(2)}%
                </div>
                <div className="text-xs text-gray-400">{stats.count} opportunities</div>
              </GlassCard>
            ))}
          </div>
        </div>
      )}

      {/* Breakdown by Bookmaker */}
      {Object.keys(analysis.by_bookmaker).length > 0 && (
        <div>
          <h3 className="text-xl font-bold text-white mb-4">🏪 CLV by Bookmaker</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {Object.entries(analysis.by_bookmaker)
              .sort(([, a], [, b]) => b.avg_clv - a.avg_clv)
              .map(([bookmaker, stats]) => (
                <GlassCard key={bookmaker} className="p-3">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-white font-medium">{bookmaker}</div>
                      <div className="text-xs text-gray-400">{stats.count} bets</div>
                    </div>
                    <div className={`text-xl font-bold ${stats.avg_clv > 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {stats.avg_clv > 0 ? '+' : ''}{stats.avg_clv.toFixed(2)}%
                    </div>
                  </div>
                </GlassCard>
              ))}
          </div>
        </div>
      )}

      {/* What We Learned */}
      <div>
        <h3 className="text-xl font-bold text-white mb-4">💡 Key Insights</h3>
        <GlassCard className="p-4" gradient="blue">
          <ul className="space-y-2 text-sm text-gray-300">
            {analysis.avg_clv > 0 ? (
              <li>✅ This game offered positive CLV opportunities on average ({analysis.avg_clv.toFixed(2)}%)</li>
            ) : (
              <li>❌ This game had negative CLV on average ({analysis.avg_clv.toFixed(2)}%)</li>
            )}
            <li>
              📊 {analysis.positive_clv_percentage.toFixed(1)}% of opportunities had positive CLV
              ({analysis.positive_clv_count} out of {analysis.total_opportunities})
            </li>
            {analysis.best_opportunities.length > 0 && (
              <li>
                🎯 Best opportunity was {analysis.best_opportunities[0].outcome} on {getMarketTypeName(analysis.best_opportunities[0].market_type)}
                {' '}at {analysis.best_opportunities[0].bookmaker} with {analysis.best_opportunities[0].clv.toFixed(2)}% CLV
              </li>
            )}
            {Object.keys(analysis.by_market).length > 0 && (
              <li>
                🏆 Best market type: {Object.entries(analysis.by_market)
                  .sort(([, a], [, b]) => b.avg_clv - a.avg_clv)[0][0]
                  .toUpperCase()} with {Object.entries(analysis.by_market)
                  .sort(([, a], [, b]) => b.avg_clv - a.avg_clv)[0][1].avg_clv.toFixed(2)}% avg CLV
              </li>
            )}
          </ul>
        </GlassCard>
      </div>
    </div>
  );
}
