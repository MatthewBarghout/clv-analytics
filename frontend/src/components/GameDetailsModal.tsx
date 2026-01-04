import React, { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import { GlassCard } from './GlassCard';
import {
  decimalToAmerican,
  explainBet,
  explainLineMovement,
  formatTimestamp,
  getMarketTypeName,
  getImpliedProbability
} from '../utils/oddsTranslator';

interface OddsSnapshot {
  id: number;
  timestamp: string;
  bookmaker_name: string;
  market_type: string;
  outcomes: Array<{
    name: string;
    price: number;
    point?: number;
  }>;
}

interface ClosingLine {
  id: number;
  bookmaker_name: string;
  market_type: string;
  outcomes: Array<{
    name: string;
    price: number;
    point?: number;
  }>;
}

interface OutcomePrediction {
  outcome_name: string;
  opening_price: number;
  opening_point: number;
  predicted_movement: number;
  predicted_direction: string;
  direction_confidence: number;
  predicted_closing_price: number;
  predicted_closing_point: number;
  actual_closing_price: number | null;
  actual_closing_point: number | null;
  movement_error: number | null;
}

interface MLPrediction {
  snapshot_id: number;
  bookmaker_name: string;
  market_type: string;
  timestamp: string;
  outcomes: OutcomePrediction[];
}

interface GameDetailsModalProps {
  gameId: number;
  homeTeam: string;
  awayTeam: string;
  onClose: () => void;
}

const API_BASE = 'http://localhost:8000/api';

export function GameDetailsModal({ gameId, homeTeam, awayTeam, onClose }: GameDetailsModalProps) {
  const [snapshots, setSnapshots] = useState<OddsSnapshot[]>([]);
  const [closingLines, setClosingLines] = useState<ClosingLine[]>([]);
  const [predictions, setPredictions] = useState<MLPrediction[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedMarket, setSelectedMarket] = useState<'h2h' | 'spreads' | 'totals'>('h2h');

  useEffect(() => {
    fetchGameDetails();
  }, [gameId]);

  const fetchGameDetails = async () => {
    try {
      setLoading(true);
      const [snapshotsRes, closingRes, predictionsRes] = await Promise.all([
        fetch(`${API_BASE}/games/${gameId}/snapshots`),
        fetch(`${API_BASE}/games/${gameId}/closing-lines`),
        fetch(`${API_BASE}/ml/predictions/${gameId}`)
      ]);

      if (snapshotsRes.ok && closingRes.ok) {
        const [snapshotsData, closingData] = await Promise.all([
          snapshotsRes.json(),
          closingRes.json()
        ]);
        setSnapshots(snapshotsData);
        setClosingLines(closingData);
      }

      if (predictionsRes.ok) {
        const predictionsData = await predictionsRes.json();
        setPredictions(predictionsData.predictions || []);
      }
    } catch (err) {
      console.error('Error fetching game details:', err);
    } finally {
      setLoading(false);
    }
  };

  const filteredSnapshots = snapshots.filter(s => s.market_type === selectedMarket);
  const filteredClosing = closingLines.filter(c => c.market_type === selectedMarket);
  const filteredPredictions = predictions.filter(p => p.market_type === selectedMarket);

  // Group snapshots by bookmaker
  const snapshotsByBookmaker = filteredSnapshots.reduce((acc, snapshot) => {
    if (!acc[snapshot.bookmaker_name]) {
      acc[snapshot.bookmaker_name] = [];
    }
    acc[snapshot.bookmaker_name].push(snapshot);
    return acc;
  }, {} as Record<string, OddsSnapshot[]>);

  const modalContent = (
    <div
      className="fixed inset-0 flex items-center justify-center p-4"
      style={{
        zIndex: 999999,
        backgroundColor: 'rgba(255, 0, 0, 0.5)',
        border: '10px solid yellow'
      }}
      onClick={(e) => {
        console.log('🔴 Modal backdrop clicked');
        if (e.target === e.currentTarget) {
          onClose();
        }
      }}
    >
      <div
        className="max-w-6xl w-full max-h-[90vh] overflow-y-auto"
        style={{
          zIndex: 1000000,
          border: '5px solid lime'
        }}
      >
        <GlassCard className="relative bg-gray-900" style={{ border: '3px solid cyan' }}>
          {/* Close button */}
          <button
            onClick={onClose}
            className="absolute top-4 right-4 text-gray-400 hover:text-white transition-colors"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>

          {/* Header */}
          <div className="mb-6">
            <h2 className="text-3xl font-bold mb-2 bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">
              {awayTeam} @ {homeTeam}
            </h2>
            <p className="text-gray-400">Detailed Betting Lines & Movement</p>
          </div>

          {/* Market Type Selector */}
          <div className="flex gap-2 mb-6">
            {(['h2h', 'spreads', 'totals'] as const).map((market) => (
              <button
                key={market}
                onClick={() => setSelectedMarket(market)}
                className={`px-4 py-2 rounded-lg transition-all duration-200 ${
                  selectedMarket === market
                    ? 'bg-white/20 text-white border border-white/30'
                    : 'bg-white/5 text-gray-400 hover:text-white hover:bg-white/10 border border-white/10'
                }`}
              >
                {getMarketTypeName(market)}
              </button>
            ))}
          </div>

          {loading ? (
            <div className="text-center py-12">
              <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500 mx-auto"></div>
              <p className="text-gray-400 mt-4">Loading betting lines...</p>
            </div>
          ) : (
            <div className="space-y-6">
              {/* Info box */}
              <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
                <h3 className="text-blue-400 font-semibold mb-2">What am I looking at?</h3>
                <p className="text-sm text-gray-300">
                  {selectedMarket === 'h2h' && 'Moneyline: Bet on which team wins straight up. Favorites have negative odds (-), underdogs have positive odds (+).'}
                  {selectedMarket === 'spreads' && 'Point Spread: The favorite must win by more than the spread, the underdog must lose by less than the spread (or win).'}
                  {selectedMarket === 'totals' && 'Over/Under: Bet on whether the combined score will be over or under the total.'}
                </p>
              </div>

              {/* ML Predictions */}
              {filteredPredictions.length > 0 && (
                <div>
                  <h3 className="text-xl font-bold mb-4 text-purple-400">ML Model Predictions</h3>
                  <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4 mb-4">
                    <h4 className="text-blue-400 font-semibold mb-2">How to Read Predictions</h4>
                    <p className="text-sm text-gray-300">
                      For each outcome (team/over/under), the model predicts: <br/>
                      - <strong>Price</strong>: What the odds will be at closing <br/>
                      - <strong>Point</strong>: What the spread/total will be at closing (for spreads/totals only) <br/>
                      <br/>
                      <strong>When to bet:</strong> If the predicted line is worse than current, bet now!
                    </p>
                  </div>
                  <div className="space-y-4">
                    {filteredPredictions.map((prediction, idx) => (
                      <GlassCard key={idx} gradient="purple" className="p-4">
                        <div className="flex justify-between items-start mb-4">
                          <h4 className="font-bold text-white text-lg">{prediction.bookmaker_name}</h4>
                          <span className="text-xs text-purple-400 bg-purple-500/20 px-2 py-1 rounded">AI PREDICTION</span>
                        </div>

                        {/* Per-outcome predictions */}
                        <div className="space-y-3">
                          {prediction.outcomes.map((outcome, outIdx) => {
                            const hasPoint = outcome.opening_point !== 0;
                            const pointMovement = hasPoint ? outcome.predicted_closing_point - outcome.opening_point : 0;
                            const priceMovement = outcome.predicted_closing_price > outcome.opening_price;

                            return (
                              <div key={outIdx} className="bg-black/30 rounded-lg p-4">
                                <div className="flex justify-between items-start mb-3">
                                  <h5 className="font-bold text-white text-base">{outcome.outcome_name}</h5>
                                  {outcome.movement_error !== null && outcome.movement_error !== undefined && (
                                    <span className="text-xs text-green-400">
                                      Error: {outcome.movement_error.toFixed(3)}
                                    </span>
                                  )}
                                </div>

                                <div className="grid grid-cols-2 gap-4">
                                  {/* Current (Opening) */}
                                  <div>
                                    <div className="text-gray-400 text-xs mb-1">Current</div>
                                    <div className="text-white font-bold">
                                      {hasPoint && `${outcome.opening_point > 0 ? '+' : ''}${outcome.opening_point} `}
                                      at {decimalToAmerican(outcome.opening_price)}
                                    </div>
                                  </div>

                                  {/* Predicted Closing */}
                                  <div>
                                    <div className="text-gray-400 text-xs mb-1">Predicted Closing</div>
                                    <div className="text-purple-400 font-bold">
                                      {hasPoint && `${outcome.predicted_closing_point > 0 ? '+' : ''}${outcome.predicted_closing_point.toFixed(1)} `}
                                      at {decimalToAmerican(outcome.predicted_closing_price)}
                                    </div>
                                  </div>
                                </div>

                                {/* Movement indicator */}
                                {hasPoint && Math.abs(pointMovement) > 0.1 && (
                                  <div className="mt-3 pt-3 border-t border-white/10">
                                    <div className="text-xs">
                                      <span className="text-yellow-400 font-semibold">Line Movement:</span>{' '}
                                      <span className={pointMovement > 0 ? 'text-red-400' : 'text-green-400'}>
                                        {pointMovement > 0 ? '+' : ''}{pointMovement.toFixed(1)} points
                                      </span>
                                      {' '}
                                      <span className="text-gray-400">
                                        ({Math.abs(pointMovement) > 0.5 ? 'significant' : 'minor'} movement)
                                      </span>
                                    </div>
                                  </div>
                                )}

                                {!hasPoint && outcome.predicted_closing_price !== outcome.opening_price && (
                                  <div className="mt-3 pt-3 border-t border-white/10">
                                    <div className="text-xs">
                                      <span className="text-yellow-400 font-semibold">Odds Movement:</span>{' '}
                                      <span className={priceMovement ? 'text-green-400' : 'text-red-400'}>
                                        {priceMovement ? 'Getting better' : 'Getting worse'}
                                      </span>
                                    </div>
                                  </div>
                                )}

                                {/* Actual closing (if available) */}
                                {outcome.actual_closing_price !== null && (
                                  <div className="mt-3 pt-3 border-t border-white/10">
                                    <div className="text-xs text-gray-400">
                                      <strong>Actual Closing:</strong>{' '}
                                      {outcome.actual_closing_point !== null && outcome.actual_closing_point !== 0 &&
                                        `${outcome.actual_closing_point > 0 ? '+' : ''}${outcome.actual_closing_point} at `}
                                      {decimalToAmerican(outcome.actual_closing_price)}
                                    </div>
                                  </div>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      </GlassCard>
                    ))}
                  </div>
                </div>
              )}

              {/* Closing Lines (if available) */}
              {filteredClosing.length > 0 && (
                <div>
                  <h3 className="text-xl font-bold mb-4 text-green-400">🎯 Closing Lines (Final Odds)</h3>
                  <div className="space-y-3">
                    {filteredClosing.map((closing) => (
                      <GlassCard key={closing.id} gradient="green" className="p-4">
                        <div className="flex justify-between items-start mb-3">
                          <h4 className="font-bold text-white">{closing.bookmaker_name}</h4>
                          <span className="text-xs text-green-400 bg-green-500/20 px-2 py-1 rounded">CLOSING LINE</span>
                        </div>
                        <div className="space-y-2">
                          {closing.outcomes.map((outcome, idx) => (
                            <div key={idx} className="bg-black/20 rounded p-3">
                              <div className="font-semibold text-white mb-1">
                                {outcome.name} {outcome.point !== undefined && `${outcome.point > 0 ? '+' : ''}${outcome.point}`}
                              </div>
                              <div className="text-green-400 font-bold mb-1">
                                {decimalToAmerican(outcome.price)} ({getImpliedProbability(outcome.price).toFixed(1)}% probability)
                              </div>
                              <div className="text-sm text-gray-300">
                                {explainBet(selectedMarket, outcome)}
                              </div>
                            </div>
                          ))}
                        </div>
                      </GlassCard>
                    ))}
                  </div>
                </div>
              )}

              {/* Line Movement by Bookmaker */}
              {Object.entries(snapshotsByBookmaker).map(([bookmaker, bookmakerSnapshots]) => {
                const sortedSnapshots = [...bookmakerSnapshots].sort(
                  (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
                );
                const firstSnapshot = sortedSnapshots[0];
                const lastSnapshot = sortedSnapshots[sortedSnapshots.length - 1];
                const closingLine = filteredClosing.find(c => c.bookmaker_name === bookmaker);

                return (
                  <div key={bookmaker}>
                    <h3 className="text-xl font-bold mb-4 text-blue-400">📈 {bookmaker} - Line Movement</h3>

                    {/* Show movement summary */}
                    {firstSnapshot && lastSnapshot && firstSnapshot.id !== lastSnapshot.id && (
                      <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4 mb-4">
                        <h4 className="text-purple-400 font-semibold mb-2">📊 Movement Summary</h4>
                        {firstSnapshot.outcomes.map((firstOutcome, idx) => {
                          const lastOutcome = lastSnapshot.outcomes[idx];
                          if (firstOutcome && lastOutcome) {
                            return (
                              <p key={idx} className="text-sm text-gray-300 mb-1">
                                <span className="font-semibold">{firstOutcome.name}:</span>{' '}
                                {explainLineMovement(firstOutcome.price, lastOutcome.price, firstOutcome.name)}
                              </p>
                            );
                          }
                          return null;
                        })}
                      </div>
                    )}

                    <div className="space-y-3">
                      {sortedSnapshots.map((snapshot, snapshotIdx) => (
                        <GlassCard
                          key={snapshot.id}
                          className="p-4"
                          gradient={snapshotIdx === 0 ? 'blue' : undefined}
                        >
                          <div className="flex justify-between items-start mb-3">
                            <div className="text-sm text-gray-400">
                              {formatTimestamp(snapshot.timestamp)}
                            </div>
                            {snapshotIdx === 0 && (
                              <span className="text-xs text-blue-400 bg-blue-500/20 px-2 py-1 rounded">OPENING</span>
                            )}
                          </div>

                          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                            {snapshot.outcomes.map((outcome, idx) => {
                              const closingOutcome = closingLine?.outcomes[idx];
                              const hasClosing = closingOutcome && closingOutcome.price !== outcome.price;

                              return (
                                <div key={idx} className="bg-black/20 rounded p-3">
                                  <div className="font-semibold text-white mb-1">
                                    {outcome.name} {outcome.point !== undefined && `${outcome.point > 0 ? '+' : ''}${outcome.point}`}
                                  </div>
                                  <div className="text-blue-400 font-bold mb-1">
                                    {decimalToAmerican(outcome.price)}
                                  </div>

                                  {hasClosing && (
                                    <div className="text-xs text-gray-400 mb-1">
                                      Closed at: {decimalToAmerican(closingOutcome.price)}
                                      {closingOutcome.price > outcome.price ? (
                                        <span className="text-green-400 ml-1">↑ Better odds</span>
                                      ) : (
                                        <span className="text-red-400 ml-1">↓ Worse odds</span>
                                      )}
                                    </div>
                                  )}

                                  <div className="text-xs text-gray-300">
                                    {explainBet(selectedMarket, outcome, snapshotIdx === 0)}
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        </GlassCard>
                      ))}
                    </div>
                  </div>
                );
              })}

              {filteredSnapshots.length === 0 && (
                <div className="text-center py-12 text-gray-400">
                  <p>No betting lines collected for this market type yet.</p>
                </div>
              )}
            </div>
          )}
        </GlassCard>
      </div>
    </div>
  );

  const portalRoot = document.getElementById('portal-root');
  if (!portalRoot) {
    return null;
  }

  // Use inline styles for the wrapper to ensure visibility
  const finalModal = (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.85)',
        backdropFilter: 'blur(4px)',
        zIndex: 999999,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '16px'
      }}
      onClick={(e) => {
        if (e.target === e.currentTarget) {
          onClose();
        }
      }}
    >
      {modalContent.props.children}
    </div>
  );

  return createPortal(finalModal, portalRoot);
}
