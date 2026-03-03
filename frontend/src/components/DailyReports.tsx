import React, { useCallback, useEffect, useRef, useState } from 'react';

const API_BASE = 'http://localhost:8000/api';

interface DailyCLVReport {
  id: number;
  report_date: string;
  games_analyzed: number;
  total_opportunities: number;
  avg_clv: number;
  median_clv: number;
  positive_clv_count: number;
  positive_clv_percentage: number;
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

interface TrackedOpportunity {
  id: number;
  game_id: number;
  home_team: string;
  away_team: string;
  home_score: number | null;
  away_score: number | null;
  bookmaker: string;
  market_type: string;
  outcome_name: string;
  point_line: number | null;
  entry_odds: number;
  closing_odds: number;
  clv_percentage: number;
  bet_amount: number;
  result: 'win' | 'loss' | 'push' | 'pending';
  profit_loss: number | null;
  settled_at: string | null;
}

export const DailyReports: React.FC = () => {
  const [dailyReports, setDailyReports] = useState<DailyCLVReport[]>([]);
  const [expandedReportId, setExpandedReportId] = useState<number | null>(null);
  const [loadingOpportunities, setLoadingOpportunities] = useState(false);
  const [trackedOpportunities, setTrackedOpportunities] = useState<TrackedOpportunity[]>([]);
  // Cache fetched opportunities to avoid re-fetching on re-expand
  const opportunitiesCache = useRef<Map<number, TrackedOpportunity[]>>(new Map());

  useEffect(() => {
    const fetchReports = async () => {
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
    fetchReports();
  }, []);

  const fetchTrackedOpportunities = useCallback(async (reportId: number) => {
    // Return cached data if available
    if (opportunitiesCache.current.has(reportId)) {
      setTrackedOpportunities(opportunitiesCache.current.get(reportId)!);
      return;
    }
    try {
      setLoadingOpportunities(true);
      const res = await fetch(`${API_BASE}/daily-reports/${reportId}/opportunities`);
      if (res.ok) {
        const data = await res.json();
        const opps = data.opportunities || [];
        opportunitiesCache.current.set(reportId, opps);
        setTrackedOpportunities(opps);
      }
    } catch (err) {
      console.error('Error fetching tracked opportunities:', err);
      setTrackedOpportunities([]);
    } finally {
      setLoadingOpportunities(false);
    }
  }, []);

  const toggleReportExpanded = useCallback(
    (reportId: number) => {
      if (expandedReportId === reportId) {
        setExpandedReportId(null);
        setTrackedOpportunities([]);
      } else {
        setExpandedReportId(reportId);
        fetchTrackedOpportunities(reportId);
      }
    },
    [expandedReportId, fetchTrackedOpportunities]
  );

  return (
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
                  <div className="text-xl font-bold text-white">{report.win_rate?.toFixed(1)}%</div>
                  <div className="text-xs text-gray-500 mt-1">
                    {report.win_count}W-{report.loss_count}L
                    {(report.push_count ?? 0) > 0 && `-${report.push_count}P`}
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
                  <div className="text-xs text-gray-500 mt-1">return on investment</div>
                </div>
              </div>
            </div>
          )}

          <div className="mt-4">
            <h4 className="text-sm font-semibold text-gray-300 mb-3 uppercase tracking-wider">Top 3 Best Opportunities</h4>
            <div className="space-y-2">
              {report.best_opportunities.slice(0, 3).map((opp, oppIndex) => (
                <div key={oppIndex} className="flex items-center justify-between bg-white/5 rounded-lg p-3 border border-white/5">
                  <div className="flex items-center gap-3">
                    <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-yellow-500/30 to-orange-500/30 border border-yellow-500/50 flex items-center justify-center">
                      <span className="text-yellow-400 font-bold text-sm">#{oppIndex + 1}</span>
                    </div>
                    <div>
                      <div className="text-sm font-medium text-white">{opp.bookmaker} • {opp.market_type}</div>
                      <div className="text-xs text-gray-400">{opp.outcome}</div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-lg font-bold text-green-400">+{opp.clv.toFixed(2)}%</div>
                    <div className="text-xs text-gray-500">
                      {opp.entry_odds > 0 ? '+' : ''}{opp.entry_odds} → {opp.closing_odds > 0 ? '+' : ''}{opp.closing_odds}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-4 pt-4 border-t border-white/10">
            <div className="text-center">
              <div className="text-xs text-gray-400 mb-1">Best Book</div>
              <div className="text-sm font-medium text-blue-400">
                {Object.entries(report.by_bookmaker).sort(([, a], [, b]) => b.avg_clv - a.avg_clv)[0]?.[0] || 'N/A'}
              </div>
            </div>
            <div className="text-center">
              <div className="text-xs text-gray-400 mb-1">Best Market</div>
              <div className="text-sm font-medium text-purple-400">
                {Object.entries(report.by_market).sort(([, a], [, b]) => b.avg_clv - a.avg_clv)[0]?.[0] || 'N/A'}
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

          {report.settled_count !== undefined && report.settled_count > 0 && (
            <div className="mt-4 pt-4 border-t border-white/10">
              <button
                onClick={() => toggleReportExpanded(report.id)}
                className={`w-full px-4 py-3 rounded-lg transition-all duration-200 flex items-center justify-center gap-2 ${
                  expandedReportId === report.id
                    ? 'bg-purple-500/20 text-purple-300 border border-purple-500/50'
                    : 'bg-white/5 text-gray-300 hover:bg-white/10 border border-white/10'
                }`}
              >
                <svg
                  className={`w-4 h-4 transition-transform duration-200 ${expandedReportId === report.id ? 'rotate-180' : ''}`}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
                {expandedReportId === report.id ? 'Hide Tracked Bets' : `View ${report.settled_count} Tracked Bets`}
              </button>

              {expandedReportId === report.id && (
                <div className="mt-4">
                  {loadingOpportunities ? (
                    <div className="text-center py-8">
                      <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-purple-500 mx-auto"></div>
                      <p className="text-gray-400 mt-2 text-sm">Loading tracked bets...</p>
                    </div>
                  ) : trackedOpportunities.length > 0 ? (
                    <div className="space-y-2">
                      <h4 className="text-sm font-semibold text-purple-300 mb-3 uppercase tracking-wider">Tracked Bet Results</h4>
                      {trackedOpportunities.map((opp) => (
                        <div
                          key={opp.id}
                          className={`flex items-center justify-between rounded-lg p-3 border ${
                            opp.result === 'win'
                              ? 'bg-green-500/10 border-green-500/30'
                              : opp.result === 'loss'
                              ? 'bg-red-500/10 border-red-500/30'
                              : opp.result === 'push'
                              ? 'bg-gray-500/10 border-gray-500/30'
                              : 'bg-white/5 border-white/10'
                          }`}
                        >
                          <div className="flex items-center gap-3">
                            <div
                              className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                                opp.result === 'win'
                                  ? 'bg-green-500/30 border border-green-500/50'
                                  : opp.result === 'loss'
                                  ? 'bg-red-500/30 border border-red-500/50'
                                  : opp.result === 'push'
                                  ? 'bg-gray-500/30 border border-gray-500/50'
                                  : 'bg-yellow-500/30 border border-yellow-500/50'
                              }`}
                            >
                              {opp.result === 'win' ? (
                                <svg className="w-4 h-4 text-green-400" fill="currentColor" viewBox="0 0 20 20">
                                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                                </svg>
                              ) : opp.result === 'loss' ? (
                                <svg className="w-4 h-4 text-red-400" fill="currentColor" viewBox="0 0 20 20">
                                  <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                                </svg>
                              ) : opp.result === 'push' ? (
                                <span className="text-gray-400 font-bold text-sm">=</span>
                              ) : (
                                <span className="text-yellow-400 font-bold text-sm">?</span>
                              )}
                            </div>
                            <div>
                              <div className="text-sm font-medium text-white">
                                {opp.away_team} @ {opp.home_team}
                                {opp.home_score !== null && opp.away_score !== null && (
                                  <span className="text-gray-400 ml-2">({opp.away_score} - {opp.home_score})</span>
                                )}
                              </div>
                              <div className="text-xs text-gray-400">
                                {opp.bookmaker} • {opp.market_type} • {opp.outcome_name}
                                {opp.point_line !== null && (
                                  <span className="text-blue-400 ml-1">
                                    {opp.market_type === 'spreads'
                                      ? `(${opp.point_line > 0 ? '+' : ''}${opp.point_line})`
                                      : `(${opp.point_line})`}
                                  </span>
                                )}
                              </div>
                            </div>
                          </div>
                          <div className="text-right">
                            <div className={`text-lg font-bold ${opp.profit_loss !== null && opp.profit_loss >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                              {opp.profit_loss !== null
                                ? `${opp.profit_loss >= 0 ? '+' : ''}$${opp.profit_loss.toFixed(2)}`
                                : 'Pending'}
                            </div>
                            <div className="text-xs text-gray-500">
                              CLV: +{opp.clv_percentage.toFixed(2)}% | {opp.entry_odds > 0 ? '+' : ''}{opp.entry_odds}
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-8 text-gray-400">
                      <p>No tracked opportunities found for this report</p>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      ))}
      {dailyReports.length === 0 && (
        <div className="text-center py-12 text-gray-400">
          <p className="text-lg mb-2">No daily reports yet</p>
          <p className="text-sm">Reports are generated daily at 9:00 AM for completed games</p>
        </div>
      )}
    </div>
  );
};
