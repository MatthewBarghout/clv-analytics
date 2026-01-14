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
- scikit-learn - Machine learning for closing line prediction

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

# Install dependencies
poetry install

# Train ML model (optional, enables predictions)
poetry run python scripts/train_model.py

# Start backend + frontend
./start.sh
```

Dashboard: http://localhost:5173
API Docs: http://localhost:8000/docs

## Key Features

### Automated Data Collection
Scheduled jobs collect NBA odds from 4 major sportsbooks (Pinnacle, FanDuel, DraftKings, ESPNbet) at opening and closing. Handles rate limiting, retries, and stores 3 market types (moneyline, spreads, totals).

### CLV Calculation Engine
Converts decimal odds to implied probabilities, compares entry vs closing, outputs percentage edge. Aggregates by bookmaker and market type to identify the sharpest books.

### Machine Learning Predictions
Random Forest model predicts closing line odds from opening snapshots. Trained on historical data to forecast line movement before games start.

**Model Performance:**
- R² Score: 0.9936 (99.36% accuracy)
- MAE: 0.0167 (average error < 2%)
- Feature Importance: Opening odds (95.5%), hours-to-game (3.8%), time factors (0.7%)

**API Endpoints:**
- `GET /api/ml/stats` - Model performance metrics
- `GET /api/ml/predictions/{game_id}` - Predicted vs actual closing lines
- `POST /api/ml/retrain` - Retrain with latest data
- `GET /api/ml/feature-importance` - Feature importance rankings

Use predictions to identify opportunities where current odds differ significantly from projected closing lines.

### Performance Dashboard
- **Mean CLV** - Overall edge across all analyzed bets
- **Positive CLV %** - Hit rate on beating the closing line
- **Trend Analysis** - CLV over time to spot improving/declining performance
- **Bookmaker Comparison** - Which books consistently offer the best prices
- **ML Model Metrics** - Real-time model performance and feature importance

### Database Schema
Designed for analytical queries:
- Composite indexes on (game_id, timestamp) for fast time-series lookups
- JSONB outcomes field for flexible market types
- Proper foreign keys with cascades
- Closing lines tracked separately for clean CLV calculation

## Sample Usage

```python
# Calculate CLV for a bet
from src.analyzers.clv_calculator import CLVCalculator

calc = CLVCalculator()
clv = calc.calculate_clv(entry_odds=2.1, closing_odds=1.95)
# Returns: +3.66%

# Predict closing line from opening odds
from src.analyzers.ml_predictor import ClosingLinePredictor
from src.analyzers.features import FeatureEngineer

predictor = ClosingLinePredictor()
predictor.load_model("models/closing_line_predictor.pkl")

# Get prediction for a new snapshot
predicted_closing = predictor.predict_closing_line(snapshot, game)
# Returns: -108 (predicted closing odds)
```

## Why This Matters

Most retail bettors chase wins. Sharp bettors chase CLV.

This system proves I understand:
- **Market efficiency** - Closing lines are the best estimate of true odds
- **Edge detection** - Consistent +CLV = long-term profit
- **Machine learning** - Predictive models for market movement (99.36% R² on closing line predictions)
- **Data engineering** - Proper schema design for analytical workloads
- **Systematic thinking** - Automation > manual processes

Same principles apply whether you're trading SPY options or betting NBA spreads. The ML model demonstrates quantitative prediction skills applicable to any financial market.

## Project Structure

```
src/
├── api/              # FastAPI endpoints
│   ├── main.py       # Main API app
│   └── ml_endpoints.py # ML prediction endpoints
├── analyzers/        # Analytics & ML
│   ├── clv_calculator.py # CLV calculation logic
│   ├── features.py   # ML feature engineering
│   └── ml_predictor.py # Random Forest model
├── collectors/       # Odds API client + data processor
└── models/           # SQLAlchemy database models

frontend/
└── src/
    └── Dashboard.tsx # React analytics dashboard w/ ML metrics

scripts/
├── collect_odds.py   # Main collection script
├── train_model.py    # ML model training
└── start_dashboard.sh # One-command startup

models/              # Trained ML models (git-ignored)
├── closing_line_predictor.pkl

migrations/          # Alembic database migrations
launchd/            # macOS scheduling configs
start.sh            # Launch backend + frontend
```

## Automated Daily Reporting

The system generates comprehensive daily reports at 9 AM showing previous day's performance with complete ROI tracking.

**Daily Report Includes:**
- CLV analysis for all completed games
- Top 10 best betting opportunities by CLV percentage
- Detailed bet tracking with win/loss results and profit calculations
- Performance metrics: win rate, total profit, ROI percentage
- Breakdown by bookmaker and market type
- ML-predicted +EV opportunities for upcoming games

**Automation Schedule:**
- 2:00 AM - Fetch final game scores
- 3:00 AM - Calculate ROI on tracked bets
- 9:00 AM - Generate daily report with full performance breakdown
- 9:30 AM - Track new opportunities from report
- 10:30 AM - Schedule closing line collection batches

The workflow eliminates manual tracking and provides immediate performance feedback on yesterday's betting opportunities.

## Roadmap

- [x] ML model to predict line movement
- [x] Automated daily CLV reports with ROI tracking
- [x] Bet performance tracking and settlement
- [ ] Direct book scraping (Kalshi, Polymarket) for higher frequency data
- [ ] Arbitrage opportunity detection
- [ ] Kelly Criterion position sizing
- [ ] Enhanced ML model tuning
- [ ] Live odds monitoring with real-time alerts

## Results

After collecting data for a few days, you can see which bookmakers consistently offer the best lines and optimal bet timing. The data speaks for itself - some books are beatable, most aren't.

## Notes

Built as a learning project to explore sports betting analytics and quantitative edge detection. Not financial advice. Bet responsibly.

---

**Stack:** Python • TypeScript • React • PostgreSQL • FastAPI • Docker
**Concepts:** Time-series data • Market efficiency • Statistical edge • Automated systems
