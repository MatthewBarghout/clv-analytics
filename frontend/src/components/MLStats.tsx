import React, { useMemo } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { GlassCard } from './GlassCard';
import { AnimatedCounter } from './AnimatedCounter';
import { CHART_COLORS, axisProps, gridProps, tooltipStyle } from '../charts/theme';

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

interface Props {
  mlStats: MLModelStats | null;
  featureImportance: FeatureImportance[];
}


export const MLStats: React.FC<Props> = React.memo(({ mlStats, featureImportance }) => {
  const topFeatures = useMemo(() => featureImportance.slice(0, 7), [featureImportance]);

  // null means the stats request failed or timed out — which is not the same thing
  // as the model being untrained, and saying so sent people chasing a training bug
  // that did not exist.
  if (mlStats === null) {
    return (
      <GlassCard className="mb-8 border-line-strong">
        <h3 className="text-lg font-semibold text-ink mb-1">Model stats unavailable</h3>
        <p className="text-sm text-ink-muted">
          Could not load model metrics. The model itself is unaffected — evaluating it is
          expensive on a cold cache, so the first request after a retrain can take a while.
        </p>
      </GlassCard>
    );
  }

  if (!mlStats.is_trained) {
    return (
      <GlassCard className="mb-8 border-warn/30 bg-warn/5">
        <h3 className="text-lg font-semibold text-warn mb-1">ML Model Not Trained</h3>
        <p className="text-sm text-ink-muted">
          Run{' '}
          <code className="px-2 py-1 bg-panel-raised rounded text-warn">
            poetry run python -m scripts.train_movement_model --walk-forward
          </code>{' '}
          to train the per-sport line movement models.
        </p>
      </GlassCard>
    );
  }

  return (
    <GlassCard className="mb-8">
      <h2 className="text-2xl font-bold mb-6 bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">
        ML Model Performance
      </h2>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4 mb-6">
        <div className="bg-panel-raised rounded-lg p-4 border border-line">
          <h3 className="text-gray-400 text-xs font-medium mb-1 uppercase tracking-wider">Movement MAE</h3>
          <AnimatedCounter value={mlStats.movement_mae || 0} decimals={4} className="text-2xl font-bold text-purple-400" />
          <p className="text-xs text-gray-500 mt-1">Avg prediction error</p>
        </div>
        <div className="bg-panel-raised rounded-lg p-4 border border-line">
          <h3 className="text-gray-400 text-xs font-medium mb-1 uppercase tracking-wider">Directional Accuracy</h3>
          <AnimatedCounter value={(mlStats.directional_accuracy || 0) * 100} decimals={1} suffix="%" className="text-2xl font-bold text-green-400" />
          <p className="text-xs text-gray-500 mt-1">Correct direction</p>
        </div>
        <div className="bg-panel-raised rounded-lg p-4 border border-line">
          <h3 className="text-gray-400 text-xs font-medium mb-1 uppercase tracking-wider">Improvement</h3>
          <AnimatedCounter value={mlStats.improvement_vs_baseline || 0} decimals={1} suffix="%" className="text-2xl font-bold text-blue-400" />
          <p className="text-xs text-gray-500 mt-1">vs baseline</p>
        </div>
        <div className="bg-panel-raised rounded-lg p-4 border border-line">
          <h3 className="text-gray-400 text-xs font-medium mb-1 uppercase tracking-wider">Precision</h3>
          <AnimatedCounter value={(mlStats.directional_precision || 0) * 100} decimals={1} suffix="%" className="text-2xl font-bold text-cyan-400" />
          <p className="text-xs text-gray-500 mt-1">Direction precision</p>
        </div>
        <div className="bg-panel-raised rounded-lg p-4 border border-line">
          <h3 className="text-gray-400 text-xs font-medium mb-1 uppercase tracking-wider">Training Records</h3>
          <AnimatedCounter value={mlStats.training_records || 0} className="text-2xl font-bold text-orange-400" />
          <p className="text-xs text-gray-500 mt-1">Data points</p>
        </div>
      </div>

      {topFeatures.length > 0 && (
        <div className="mt-6">
          <h3 className="text-lg font-semibold mb-4 text-gray-300">Feature Importance</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={topFeatures} layout="vertical" margin={{ top: 5, right: 30, left: 100, bottom: 5 }}>
              <CartesianGrid {...gridProps} />
              <XAxis type="number" {...axisProps} />
              <YAxis type="category" dataKey="feature_name" {...axisProps} />
              <Tooltip
                contentStyle={tooltipStyle}
                labelStyle={{ color: '#F3F4F6' }}
                itemStyle={{ color: CHART_COLORS.tick }}
                formatter={(value: number | undefined) => [
                  (value ? (value * 100).toFixed(2) : '0') + '%',
                  'Importance',
                ]}
              />
              <Bar dataKey="importance" fill="#3b82f6" radius={[0, 4, 4, 0]} isAnimationActive animationDuration={800} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {mlStats.last_trained && (
        <div className="mt-4 text-xs text-gray-500 text-center">
          Last trained: {new Date(mlStats.last_trained).toLocaleString()}
        </div>
      )}
    </GlassCard>
  );
});
