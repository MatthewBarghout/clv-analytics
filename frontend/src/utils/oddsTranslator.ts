/**
 * Utilities for translating betting odds and explaining bets to users
 */

/**
 * Convert decimal odds to American odds format
 */
export function decimalToAmerican(decimal: number): string {
  if (decimal >= 2.0) {
    // Positive American odds (underdog)
    const american = Math.round((decimal - 1) * 100);
    return `+${american}`;
  } else {
    // Negative American odds (favorite)
    const american = Math.round(-100 / (decimal - 1));
    return `${american}`;
  }
}

/**
 * Calculate implied probability from decimal odds
 */
export function getImpliedProbability(decimal: number): number {
  return (1 / decimal) * 100;
}

/**
 * Get a human-readable explanation of what a bet means
 */
export function explainBet(
  marketType: string,
  outcome: any,
  isOpening: boolean = false
): string {
  const lineType = isOpening ? 'Opening line' : 'Line';

  if (marketType === 'h2h') {
    const american = decimalToAmerican(outcome.price);
    const prob = getImpliedProbability(outcome.price).toFixed(1);

    if (outcome.price < 2.0) {
      return `${lineType}: ${outcome.name} ${american} (Favorite - ${prob}% implied probability). Bet $${Math.abs(parseInt(american))} to win $100.`;
    } else {
      return `${lineType}: ${outcome.name} ${american} (Underdog - ${prob}% implied probability). Bet $100 to win $${parseInt(american.replace('+', ''))}.`;
    }
  } else if (marketType === 'spreads') {
    const american = decimalToAmerican(outcome.price);
    const spread = outcome.point > 0 ? `+${outcome.point}` : `${outcome.point}`;
    return `${lineType}: ${outcome.name} ${spread} at ${american}. ${outcome.name} must ${outcome.point > 0 ? 'lose by less than' : 'win by more than'} ${Math.abs(outcome.point)} points.`;
  } else if (marketType === 'totals') {
    const american = decimalToAmerican(outcome.price);
    const direction = outcome.name === 'Over' ? 'more' : 'fewer';
    return `${lineType}: ${outcome.name} ${outcome.point} at ${american}. Combined score must be ${direction} than ${outcome.point} points.`;
  }

  return 'Unknown bet type';
}

/**
 * Format a timestamp to readable time
 */
export function formatTimestamp(timestamp: string): string {
  return new Date(timestamp).toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  });
}

/**
 * Explain line movement
 */
export function explainLineMovement(
  openingOdds: number,
  closingOdds: number,
  teamName: string
): string {
  if (openingOdds === closingOdds) {
    return `No movement - line stayed at ${decimalToAmerican(openingOdds)}`;
  }

  const direction = closingOdds > openingOdds ? 'moved up' : 'moved down';
  const openingAmerican = decimalToAmerican(openingOdds);
  const closingAmerican = decimalToAmerican(closingOdds);

  if (closingOdds > openingOdds) {
    return `Line ${direction} from ${openingAmerican} to ${closingAmerican} - ${teamName} became LESS favored (better value for bettors)`;
  } else {
    return `Line ${direction} from ${openingAmerican} to ${closingAmerican} - ${teamName} became MORE favored (worse value for bettors)`;
  }
}

/**
 * Get market type display name
 */
export function getMarketTypeName(marketType: string): string {
  const names: Record<string, string> = {
    'h2h': 'Moneyline',
    'spreads': 'Point Spread',
    'totals': 'Over/Under'
  };
  return names[marketType] || marketType;
}
