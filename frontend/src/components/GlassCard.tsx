import React from 'react';

interface GlassCardProps {
  children: React.ReactNode;
  className?: string;
  gradient?: 'green' | 'red' | 'blue' | 'purple';
}

export function GlassCard({ children, className = '', gradient }: GlassCardProps) {
  const gradients = {
    green: 'from-green-500/10 to-emerald-500/5',
    red: 'from-red-500/10 to-rose-500/5',
    blue: 'from-blue-500/10 to-cyan-500/5',
    purple: 'from-purple-500/10 to-pink-500/5'
  };

  return (
    <div
      className={`
        relative backdrop-blur-xl bg-white/5
        rounded-2xl p-6 border border-white/10
        shadow-2xl shadow-black/20
        transition-all duration-300
        hover:shadow-3xl hover:border-white/20
        ${gradient ? `bg-gradient-to-br ${gradients[gradient]}` : ''}
        ${className}
      `}
    >
      {/* Subtle inner glow */}
      <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-white/5 to-transparent pointer-events-none" />

      {/* Content */}
      <div className="relative z-10">
        {children}
      </div>
    </div>
  );
}
