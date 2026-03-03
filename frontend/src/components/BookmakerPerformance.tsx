import React, { useMemo } from 'react';
import { GlassCard } from './GlassCard';

interface BookmakerStats {
  bookmaker_name: string;
  total_snapshots: number;
  avg_clv: number | null;
  positive_clv_percentage: number;
}

interface Props {
  bookmakers: BookmakerStats[];
}

export const BookmakerPerformance: React.FC<Props> = React.memo(({ bookmakers }) => {
  const sorted = useMemo(
    () => [...bookmakers].sort((a, b) => (b.avg_clv ?? -Infinity) - (a.avg_clv ?? -Infinity)),
    [bookmakers]
  );

  return (
    <GlassCard className="mb-8">
      <h2 className="text-2xl font-bold mb-6 bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">
        Bookmaker Performance
      </h2>
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-gray-700/50">
              <th className="text-left py-4 px-4 text-gray-400 font-semibold uppercase text-xs tracking-wider">Bookmaker</th>
              <th className="text-center py-4 px-4 text-gray-400 font-semibold uppercase text-xs tracking-wider">Avg CLV</th>
              <th className="text-center py-4 px-4 text-gray-400 font-semibold uppercase text-xs tracking-wider">Positive CLV %</th>
              <th className="text-center py-4 px-4 text-gray-400 font-semibold uppercase text-xs tracking-wider">Total Snapshots</th>
              <th className="text-right py-4 px-4 text-gray-400 font-semibold uppercase text-xs tracking-wider">Performance</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((bookmaker, index) => (
              <tr
                key={bookmaker.bookmaker_name}
                className="border-b border-gray-700/30 hover:bg-white/5 transition-colors duration-200"
                style={{ animationDelay: `${index * 50}ms` }}
              >
                <td className="py-4 px-4">
                  <span className="font-medium text-white">{bookmaker.bookmaker_name}</span>
                </td>
                <td className="text-center py-4 px-4">
                  <span
                    className={`font-bold text-lg ${
                      (bookmaker.avg_clv || 0) > 0 ? 'text-clv-positive-400' : 'text-clv-negative-400'
                    }`}
                  >
                    {bookmaker.avg_clv?.toFixed(2) || 'N/A'}%
                  </span>
                </td>
                <td className="text-center py-4 px-4">
                  <div className="flex items-center justify-center gap-2">
                    <div className="w-24 h-2 bg-gray-700 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-clv-positive-600 to-clv-positive-400 rounded-full transition-all duration-500"
                        style={{ width: `${bookmaker.positive_clv_percentage}%` }}
                      />
                    </div>
                    <span className="text-clv-positive-400 font-medium text-sm">
                      {bookmaker.positive_clv_percentage.toFixed(1)}%
                    </span>
                  </div>
                </td>
                <td className="text-center py-4 px-4">
                  <span className="text-blue-400 font-medium">{bookmaker.total_snapshots.toLocaleString()}</span>
                </td>
                <td className="text-right py-4 px-4">
                  {(bookmaker.avg_clv || 0) > 0 ? (
                    <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-clv-positive-500/20 text-clv-positive-400 border border-clv-positive-500/30">
                      Favorable
                    </span>
                  ) : (
                    <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-clv-negative-500/20 text-clv-negative-400 border border-clv-negative-500/30">
                      Unfavorable
                    </span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </GlassCard>
  );
});
