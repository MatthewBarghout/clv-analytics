import React, { useCallback, useEffect, useState } from 'react';

const API_BASE = 'http://localhost:8000/api';

interface UserBet {
  id: number;
  game_description: string;
  game_date: string;
  bookmaker: string;
  market_type: string;
  bet_description: string;
  odds: number;
  stake: number;
  result: string;
  profit_loss: number | null;
  closing_odds: number | null;
  clv_percentage: number | null;
  notes: string | null;
  created_at: string;
  settled_at: string | null;
}

interface UserBetsSummary {
  total_bets: number;
  pending: number;
  settled: number;
  wins: number;
  losses: number;
  pushes: number;
  win_rate: number;
  total_profit: number;
  total_staked: number;
  roi: number;
}

const emptyBet = {
  game_description: '',
  game_date: '',
  bookmaker: 'DraftKings',
  market_type: 'h2h',
  bet_description: '',
  odds: '',
  stake: '100',
};

export const MyBets: React.FC = () => {
  const [userBets, setUserBets] = useState<UserBet[]>([]);
  const [userBetsSummary, setUserBetsSummary] = useState<UserBetsSummary | null>(null);
  const [loadingUserBets, setLoadingUserBets] = useState(false);
  const [showAddBetForm, setShowAddBetForm] = useState(false);
  const [newBet, setNewBet] = useState(emptyBet);

  const fetchUserBets = useCallback(async () => {
    try {
      setLoadingUserBets(true);
      const [betsRes, summaryRes] = await Promise.all([
        fetch(`${API_BASE}/user-bets`),
        fetch(`${API_BASE}/user-bets/summary`),
      ]);
      if (betsRes.ok) setUserBets(await betsRes.json());
      if (summaryRes.ok) setUserBetsSummary(await summaryRes.json());
    } catch (err) {
      console.error('Error fetching user bets:', err);
    } finally {
      setLoadingUserBets(false);
    }
  }, []);

  useEffect(() => {
    fetchUserBets();
  }, [fetchUserBets]);

  const handleAddBet = useCallback(async () => {
    try {
      const params = new URLSearchParams({
        game_description: newBet.game_description,
        game_date: newBet.game_date,
        bookmaker: newBet.bookmaker,
        market_type: newBet.market_type,
        bet_description: newBet.bet_description,
        odds: newBet.odds,
        stake: newBet.stake,
      });
      const res = await fetch(`${API_BASE}/user-bets?${params}`, { method: 'POST' });
      if (res.ok) {
        setShowAddBetForm(false);
        setNewBet(emptyBet);
        fetchUserBets();
      }
    } catch (err) {
      console.error('Error adding bet:', err);
    }
  }, [newBet, fetchUserBets]);

  const handleSettleBet = useCallback(async (betId: number, result: 'win' | 'loss' | 'push') => {
    try {
      const res = await fetch(`${API_BASE}/user-bets/${betId}?result=${result}`, { method: 'PUT' });
      if (res.ok) fetchUserBets();
    } catch (err) {
      console.error('Error settling bet:', err);
    }
  }, [fetchUserBets]);

  const handleDeleteBet = useCallback(async (betId: number) => {
    try {
      const res = await fetch(`${API_BASE}/user-bets/${betId}`, { method: 'DELETE' });
      if (res.ok) fetchUserBets();
    } catch (err) {
      console.error('Error deleting bet:', err);
    }
  }, [fetchUserBets]);

  return (
    <div className="space-y-6">
      {userBetsSummary && (
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          {[
            { label: 'Pending', value: userBetsSummary.pending, color: 'text-yellow-400' },
            { label: 'Record', value: `${userBetsSummary.wins}-${userBetsSummary.losses}`, color: 'text-white' },
            { label: 'Win Rate', value: `${userBetsSummary.win_rate}%`, color: 'text-blue-400' },
            { label: 'Profit', value: `${userBetsSummary.total_profit >= 0 ? '+' : ''}$${userBetsSummary.total_profit}`, color: userBetsSummary.total_profit >= 0 ? 'text-green-400' : 'text-red-400' },
            { label: 'ROI', value: `${userBetsSummary.roi >= 0 ? '+' : ''}${userBetsSummary.roi}%`, color: userBetsSummary.roi >= 0 ? 'text-green-400' : 'text-red-400' },
          ].map((card) => (
            <div key={card.label} className="bg-white/5 rounded-lg p-4 border border-white/10">
              <div className="text-xs text-gray-400 mb-1 uppercase">{card.label}</div>
              <div className={`text-2xl font-bold ${card.color}`}>{card.value}</div>
            </div>
          ))}
        </div>
      )}

      <div className="flex justify-end">
        <button
          onClick={() => setShowAddBetForm(!showAddBetForm)}
          className="px-4 py-2 bg-green-500/20 hover:bg-green-500/30 text-green-400 rounded-lg border border-green-500/50 transition-all"
        >
          {showAddBetForm ? 'Cancel' : '+ Add Bet'}
        </button>
      </div>

      {showAddBetForm && (
        <div className="bg-white/5 rounded-lg p-6 border border-white/10">
          <h3 className="text-lg font-semibold text-white mb-4">Add New Bet</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <input
              type="text"
              placeholder="Game (e.g., Bulls @ Celtics)"
              value={newBet.game_description}
              onChange={(e) => setNewBet({ ...newBet, game_description: e.target.value })}
              className="bg-gray-800 border border-gray-700 rounded px-3 py-2 text-white"
            />
            <input
              type="datetime-local"
              value={newBet.game_date}
              onChange={(e) => setNewBet({ ...newBet, game_date: e.target.value })}
              className="bg-gray-800 border border-gray-700 rounded px-3 py-2 text-white"
            />
            <select
              value={newBet.bookmaker}
              onChange={(e) => setNewBet({ ...newBet, bookmaker: e.target.value })}
              className="bg-gray-800 border border-gray-700 rounded px-3 py-2 text-white"
            >
              <option value="DraftKings">DraftKings</option>
              <option value="FanDuel">FanDuel</option>
              <option value="BetMGM">BetMGM</option>
              <option value="Caesars">Caesars</option>
              <option value="theScore Bet">theScore Bet</option>
            </select>
            <select
              value={newBet.market_type}
              onChange={(e) => setNewBet({ ...newBet, market_type: e.target.value })}
              className="bg-gray-800 border border-gray-700 rounded px-3 py-2 text-white"
            >
              <option value="h2h">Moneyline</option>
              <option value="spreads">Spread</option>
              <option value="totals">Totals</option>
            </select>
            <input
              type="text"
              placeholder="Bet (e.g., Bulls -1.5)"
              value={newBet.bet_description}
              onChange={(e) => setNewBet({ ...newBet, bet_description: e.target.value })}
              className="bg-gray-800 border border-gray-700 rounded px-3 py-2 text-white"
            />
            <input
              type="number"
              placeholder="Odds (e.g., +140 or -110)"
              value={newBet.odds}
              onChange={(e) => setNewBet({ ...newBet, odds: e.target.value })}
              className="bg-gray-800 border border-gray-700 rounded px-3 py-2 text-white"
            />
            <input
              type="number"
              placeholder="Stake"
              value={newBet.stake}
              onChange={(e) => setNewBet({ ...newBet, stake: e.target.value })}
              className="bg-gray-800 border border-gray-700 rounded px-3 py-2 text-white"
            />
            <button
              onClick={handleAddBet}
              className="px-4 py-2 bg-green-500 hover:bg-green-600 text-white rounded-lg font-medium transition-all"
            >
              Add Bet
            </button>
          </div>
        </div>
      )}

      {loadingUserBets ? (
        <div className="text-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-yellow-500 mx-auto"></div>
        </div>
      ) : userBets.length > 0 ? (
        <div className="bg-white/5 rounded-lg border border-white/10 overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-gray-900">
              <tr className="border-b border-gray-700">
                {['Date', 'Game', 'Bet', 'Book', 'Odds', 'Stake', 'Status', 'P&L', 'Actions'].map((h, i) => (
                  <th key={h} className={`py-3 px-4 text-gray-400 ${i < 3 ? 'text-left' : i === 7 ? 'text-right' : 'text-center'}`}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {userBets.map((bet) => (
                <tr key={bet.id} className="border-b border-gray-800 hover:bg-white/5">
                  <td className="py-3 px-4 text-gray-400">{new Date(bet.game_date).toLocaleDateString()}</td>
                  <td className="py-3 px-4 text-white">{bet.game_description}</td>
                  <td className="py-3 px-4">
                    <span className="text-purple-400">{bet.market_type}</span>
                    <span className="text-white ml-2">{bet.bet_description}</span>
                  </td>
                  <td className="text-center py-3 px-4 text-gray-300">{bet.bookmaker}</td>
                  <td className="text-center py-3 px-4 text-blue-400 font-mono">
                    {bet.odds > 0 ? '+' : ''}{bet.odds}
                  </td>
                  <td className="text-center py-3 px-4 text-white">${bet.stake}</td>
                  <td className="text-center py-3 px-4">
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      bet.result === 'win' ? 'bg-green-500/20 text-green-400' :
                      bet.result === 'loss' ? 'bg-red-500/20 text-red-400' :
                      bet.result === 'push' ? 'bg-gray-500/20 text-gray-400' :
                      'bg-yellow-500/20 text-yellow-400'
                    }`}>
                      {bet.result.toUpperCase()}
                    </span>
                  </td>
                  <td className={`text-right py-3 px-4 font-medium ${
                    bet.profit_loss === null ? 'text-gray-500' :
                    bet.profit_loss >= 0 ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {bet.profit_loss !== null ? `${bet.profit_loss >= 0 ? '+' : ''}$${bet.profit_loss.toFixed(0)}` : '-'}
                  </td>
                  <td className="text-center py-3 px-4">
                    {bet.result === 'pending' ? (
                      <div className="flex gap-1 justify-center">
                        {(['win', 'loss', 'push'] as const).map((r) => (
                          <button
                            key={r}
                            onClick={() => handleSettleBet(bet.id, r)}
                            className={`px-2 py-1 text-xs rounded ${
                              r === 'win' ? 'bg-green-500/20 hover:bg-green-500/30 text-green-400' :
                              r === 'loss' ? 'bg-red-500/20 hover:bg-red-500/30 text-red-400' :
                              'bg-gray-500/20 hover:bg-gray-500/30 text-gray-400'
                            }`}
                          >
                            {r === 'win' ? 'W' : r === 'loss' ? 'L' : 'P'}
                          </button>
                        ))}
                        <button
                          onClick={() => handleDeleteBet(bet.id)}
                          className="px-2 py-1 text-xs bg-red-500/10 hover:bg-red-500/20 text-red-400 rounded"
                        >
                          X
                        </button>
                      </div>
                    ) : (
                      <span className="text-gray-500 text-xs">Settled</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="text-center py-12 text-gray-400">
          <div className="text-6xl mb-4">🎰</div>
          <p className="text-lg mb-2">No bets tracked yet</p>
          <p className="text-sm">Click "+ Add Bet" to start tracking your bets</p>
        </div>
      )}
    </div>
  );
};
