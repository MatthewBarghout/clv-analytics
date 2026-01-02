import React from 'react';

interface TimeRangeSelectorProps {
  selected: '7d' | '30d' | '90d' | 'all';
  onChange: (range: '7d' | '30d' | '90d' | 'all') => void;
}

export function TimeRangeSelector({ selected, onChange }: TimeRangeSelectorProps) {
  const options = [
    { value: '7d' as const, label: '7 Days' },
    { value: '30d' as const, label: '30 Days' },
    { value: '90d' as const, label: '90 Days' },
    { value: 'all' as const, label: 'All Time' }
  ];

  return (
    <div className="inline-flex rounded-lg bg-white/5 p-1 backdrop-blur-sm border border-white/10">
      {options.map((option) => (
        <button
          key={option.value}
          onClick={() => onChange(option.value)}
          className={`
            px-4 py-2 rounded-md text-sm font-medium transition-all duration-200
            ${selected === option.value
              ? 'bg-white/20 text-white shadow-lg'
              : 'text-gray-400 hover:text-white hover:bg-white/10'
            }
          `}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}
