#!/usr/bin/env python3
"""
Analyze historical line movement patterns to determine realistic prediction caps.

This script analyzes actual closing line movements by market type and calculates
statistics to inform model prediction constraints.
"""
import json
import logging
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from src.models.database import ClosingLine, OddsSnapshot

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")
POSTGRES_DB = os.getenv("POSTGRES_DB")

DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"


def analyze_movement_statistics(session: Session) -> Dict[str, Dict[str, float]]:
    """
    Calculate comprehensive movement statistics by market type.

    Args:
        session: Database session

    Returns:
        Dictionary of statistics by market type
    """
    logger.info("Analyzing historical line movements...")

    # Query all snapshots with closing lines
    stmt = select(OddsSnapshot, ClosingLine).join(
        ClosingLine,
        (OddsSnapshot.game_id == ClosingLine.game_id) &
        (OddsSnapshot.bookmaker_id == ClosingLine.bookmaker_id) &
        (OddsSnapshot.market_type == ClosingLine.market_type)
    )

    results = session.execute(stmt).all()
    logger.info(f"Found {len(results)} snapshot-closing line pairs")

    # Collect movements by market type
    movements_by_market: Dict[str, List[Dict]] = {
        'h2h': [],
        'spreads': [],
        'totals': []
    }

    for snapshot, closing in results:
        market_type = snapshot.market_type

        # Match outcomes between snapshot and closing
        for snap_outcome in snapshot.outcomes:
            for close_outcome in closing.outcomes:
                if snap_outcome.get('name') == close_outcome.get('name'):
                    opening_price = snap_outcome.get('price')
                    closing_price = close_outcome.get('price')

                    if opening_price is not None and closing_price is not None:
                        price_movement = float(closing_price) - float(opening_price)

                        movement_data = {
                            'price_movement': price_movement,
                            'abs_price_movement': abs(price_movement)
                        }

                        # Add point movement for spreads/totals
                        if market_type in ['spreads', 'totals']:
                            opening_point = snap_outcome.get('point')
                            closing_point = close_outcome.get('point')

                            if opening_point is not None and closing_point is not None:
                                point_movement = float(closing_point) - float(opening_point)
                                movement_data['point_movement'] = point_movement
                                movement_data['abs_point_movement'] = abs(point_movement)

                        movements_by_market[market_type].append(movement_data)
                    break

    # Calculate statistics for each market type
    statistics = {}

    for market_type, movements in movements_by_market.items():
        if not movements:
            logger.warning(f"No movements found for {market_type}")
            continue

        # Convert to DataFrame for easy statistics
        df = pd.DataFrame(movements)

        # Price movement statistics
        price_movements = df['abs_price_movement'].values

        stats = {
            'count': len(movements),
            'mean_abs_movement': float(np.mean(price_movements)),
            'median_abs_movement': float(np.median(price_movements)),
            'std_dev': float(np.std(price_movements)),
            'p50': float(np.percentile(price_movements, 50)),
            'p75': float(np.percentile(price_movements, 75)),
            'p90': float(np.percentile(price_movements, 90)),
            'p95': float(np.percentile(price_movements, 95)),
            'p99': float(np.percentile(price_movements, 99)),
            'max': float(np.max(price_movements)),
        }

        # Add point movement statistics for spreads/totals
        if market_type in ['spreads', 'totals'] and 'abs_point_movement' in df.columns:
            point_movements = df['abs_point_movement'].values
            stats['point_stats'] = {
                'mean_abs_movement': float(np.mean(point_movements)),
                'median_abs_movement': float(np.median(point_movements)),
                'p95': float(np.percentile(point_movements, 95)),
                'max': float(np.max(point_movements)),
            }

        statistics[market_type] = stats

    return statistics


def print_statistics_table(statistics: Dict[str, Dict[str, float]]) -> None:
    """Print formatted statistics table."""
    logger.info("\n" + "=" * 80)
    logger.info("HISTORICAL LINE MOVEMENT ANALYSIS")
    logger.info("=" * 80)

    for market_type, stats in statistics.items():
        logger.info(f"\n{market_type.upper()} Markets:")
        logger.info("-" * 80)
        logger.info(f"  Observations:           {stats['count']}")
        logger.info(f"  Mean Absolute Movement: {stats['mean_abs_movement']:.4f}")
        logger.info(f"  Median Absolute:        {stats['median_abs_movement']:.4f}")
        logger.info(f"  Std Deviation:          {stats['std_dev']:.4f}")
        logger.info(f"  ")
        logger.info(f"  Percentiles:")
        logger.info(f"    50th (Median):        {stats['p50']:.4f}")
        logger.info(f"    75th:                 {stats['p75']:.4f}")
        logger.info(f"    90th:                 {stats['p90']:.4f}")
        logger.info(f"    95th (Recommended):   {stats['p95']:.4f}  ← SUGGESTED CAP")
        logger.info(f"    99th:                 {stats['p99']:.4f}")
        logger.info(f"  ")
        logger.info(f"  Maximum Observed:       {stats['max']:.4f}")

        if 'point_stats' in stats:
            ps = stats['point_stats']
            logger.info(f"  ")
            logger.info(f"  Point Movement:")
            logger.info(f"    Mean Absolute:        {ps['mean_abs_movement']:.4f}")
            logger.info(f"    Median Absolute:      {ps['median_abs_movement']:.4f}")
            logger.info(f"    95th Percentile:      {ps['p95']:.4f}")
            logger.info(f"    Maximum:              {ps['max']:.4f}")

    logger.info("\n" + "=" * 80)
    logger.info("RECOMMENDED PREDICTION CAPS (95th Percentile):")
    logger.info("=" * 80)
    for market_type, stats in statistics.items():
        logger.info(f"  {market_type:12s}: ±{stats['p95']:.4f}")
    logger.info("=" * 80 + "\n")


def save_statistics(statistics: Dict[str, Dict[str, float]], output_path: Path) -> None:
    """Save statistics to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(statistics, f, indent=2)

    logger.info(f"Statistics saved to: {output_path}")


def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("STARTING MOVEMENT PATTERN ANALYSIS")
    logger.info("=" * 80)

    # Setup database connection
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()

    try:
        # Analyze movements
        statistics = analyze_movement_statistics(session)

        # Print results
        print_statistics_table(statistics)

        # Save to file
        output_path = Path("models/movement_statistics.json")
        save_statistics(statistics, output_path)

        logger.info("✓ Analysis complete!")

    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise

    finally:
        session.close()


if __name__ == "__main__":
    main()
