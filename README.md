# CLV Analytics Platform

A quantitative sports betting analytics system that tracks and calculates Closing Line Value (CLV) to identify profitable betting opportunities. Built to demonstrate systematic edge detection and data-driven decision making in sports markets.

## What is CLV?

Closing Line Value measures whether you're getting better odds than the market's final assessment. If you consistently beat the closing line, you have an edge. It's the single most reliable indicator of long-term profitability in sports betting - think of it as alpha in traditional markets.

**Example:**
- You bet Lakers at +200 (3.0 decimal) at 10am
- Line closes at +180 (2.8 decimal)
- Your CLV: +2.38% - you beat the market

Do this consistently across hundreds of bets, and you're printing money.

## The System

This platform automates the entire workflow:

1. **Data Collection** - Polls The Odds API twice daily (opening + closing lines)
2. **Storage** - PostgreSQL time-series database captures odds movement
3. **Analysis** - Calculates CLV by comparing entry vs closing odds
4. **Visualization** - Real-time dashboard shows performance metrics

The goal: prove you can identify +EV opportunities before the market corrects.

## Tech Stack

**Backend:**
- FastAPI (Python) - API endpoints for CLV calculations
- SQLAlchemy 2.0 - ORM with proper indexing for time-series queries
- PostgreSQL - JSONB for flexible odds storage
- Alembic - Database migrations

**Frontend:**
- React + TypeScript - Type-safe component architecture
- Vite - Fast build tooling
- Recharts - Data visualization
- Tailwind CSS - Styling

**Infrastructure:**
- Docker Compose - Local PostgreSQL instance
- macOS launchd - Scheduled collection (10am/6pm ET)
- Poetry - Python dependency management

## Quick Start

```bash
# Setup
docker-compose up -d
alembic upgrade head
cp .env.example .env
# Add your ODDS_API_KEY

# Start everything
./scripts/start_dashboard.sh
```

Dashboard: http://localhost:5173
API Docs: http://localhost:8000/docs

## Key Features

### Automated Data Collection
Scheduled jobs collect NBA odds from 4 major sportsbooks (Pinnacle, FanDuel, DraftKings, ESPNbet) at opening and closing. Handles rate limiting, retries, and stores 3 market types (moneyline, spreads, totals).

### CLV Calculation Engine
Converts decimal odds to implied probabilities, compares entry vs closing, outputs percentage edge. Aggregates by bookmaker and market type to identify the sharpest books.

### Performance Dashboard
- **Mean CLV** - Overall edge across all analyzed bets
- **Positive CLV %** - Hit rate on beating the closing line
- **Trend Analysis** - CLV over time to spot improving/declining performance
- **Bookmaker Comparison** - Which books consistently offer the best prices

### Database Schema
Designed for analytical queries:
- Composite indexes on (game_id, timestamp) for fast time-series lookups
- JSONB outcomes field for flexible market types
- Proper foreign keys with cascades
- Closing lines tracked separately for clean CLV calculation

## Sample Queries

```python
# Find all bets that beat closing by >2%
from src.analyzers.clv_calculator import CLVCalculator

calc = CLVCalculator()
clv = calc.calculate_clv(entry_odds=2.1, closing_odds=1.95)
# Returns: +3.66%
```

## Why This Matters

Most retail bettors chase wins. Sharp bettors chase CLV.

This system proves I understand:
- **Market efficiency** - Closing lines are the best estimate of true odds
- **Edge detection** - Consistent +CLV = long-term profit
- **Data engineering** - Proper schema design for analytical workloads
- **Systematic thinking** - Automation > manual processes

Same principles apply whether you're trading SPY options or betting NBA spreads.

## Project Structure

```
src/
├── api/              # FastAPI endpoints
├── analyzers/        # CLV calculation logic
├── collectors/       # Odds API client + data processor
└── models/           # SQLAlchemy database models

frontend/
└── src/
    └── Dashboard.tsx # React analytics dashboard

scripts/
├── collect_odds.py   # Main collection script
└── start_dashboard.sh # One-command startup

migrations/           # Alembic database migrations
launchd/             # macOS scheduling configs
```

## Roadmap

- [ ] ML model to predict line movement
- [ ] Bet tracking + P&L analysis
- [ ] Arbitrage opportunity detection
- [ ] Kelly Criterion position sizing
- [ ] Live odds monitoring

## Results

After collecting data for a few days, you can see which bookmakers consistently offer the best lines and optimal bet timing. The data speaks for itself - some books are beatable, most aren't.

## Notes

Built as a learning project to explore sports betting analytics and quantitative edge detection. Not financial advice. Bet responsibly.

---

**Stack:** Python • TypeScript • React • PostgreSQL • FastAPI • Docker
**Concepts:** Time-series data • Market efficiency • Statistical edge • Automated systems
