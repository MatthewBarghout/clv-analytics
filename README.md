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
- XGBoost + scikit-learn - Ensemble ML for line movement prediction

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
poetry run python -m scripts.train_movement_model

# Start backend + frontend
./start.sh
```

Dashboard: http://localhost:5173
API Docs: http://localhost:8000/docs

## Key Features

### Automated Data Collection
Scheduled jobs collect NBA odds from 4 major sportsbooks (Pinnacle, FanDuel, DraftKings, theScore Bet) at opening and closing. Dynamic scheduler creates launchd batches 30 minutes before each game to capture closing lines. Handles rate limiting, retries, and stores 3 market types (moneyline, spreads, totals).

### CLV Calculation Engine
Converts decimal odds to implied probabilities, compares entry vs closing, outputs percentage edge. Aggregates by bookmaker and market type to identify the sharpest books.

### Machine Learning Predictions
Ensemble model (XGBoost + Random Forest) predicts line movement direction and magnitude. Identifies +EV opportunities by forecasting which lines will move favorably before games start.

**Model Performance:**
- Direction Accuracy: 65.1% (vs 40.7% baseline)
- MAE: 0.0496 (price movement prediction)
- R² Score: 0.58
- Improvement vs Baseline: 22.9%

**Top Features:**
- Consensus line (13.9%)
- Opening price (12.1%)
- Line spread (8.4%)
- Distance from consensus (6.2%)
- Hours to game (4.7%)

**Advanced ML Features:**
- Temporal features: movement velocity, price volatility, cumulative movement, direction changes
- Bookmaker features: sharp book detection (Pinnacle), relative positioning, steam move signals
- Market-specific direction thresholds: h2h=0.02, spreads=0.01, totals=0.01
- Per-market ensemble weights optimized via grid search
- Walk-forward (expanding window) validation for time-series integrity
- Auto-retraining when performance degrades >10%

**API Endpoints:**
- `GET /api/ml/stats` - Model performance metrics
- `GET /api/ml/predictions/{game_id}` - Predicted vs actual closing lines
- `POST /api/ml/retrain` - Retrain with latest data
- `GET /api/ml/feature-importance` - Feature importance rankings
- `GET /api/ml/opportunities` - Comprehensive filterable opportunities
- `GET /api/ml/upcoming-opportunities` - Games starting in next N hours

Use predictions to identify +EV opportunities where the model predicts favorable line movement.

### Bet Sizing Strategies
Multiple position sizing strategies for bankroll management:

- **Fixed Unit** - Flat amount per bet (e.g., $100)
- **Fractional** - Percentage of bankroll (e.g., 2%)
- **Kelly Criterion** - Optimal sizing based on edge: f* = (bp - q) / b
- **Half-Kelly** - Conservative Kelly at 50% for reduced variance
- **Confidence-Weighted** - Scale bet size by ML model confidence

**Risk Metrics:**
- Sharpe Ratio - Risk-adjusted returns
- Sortino Ratio - Downside-focused risk metric
- Max drawdown tracking

### Enhanced Bankroll Simulation
Advanced simulation with strategy selection and filtering:

**Parameters:**
- `strategy` - fixed, fractional, kelly, half_kelly, confidence
- `bookmaker_filter` - Filter by specific bookmaker
- `market_filter` - Filter by market type (h2h, spreads, totals)
- `clv_threshold` - Minimum CLV to include

**Performance Breakdown:**
- By bookmaker - Which books are most profitable
- By market type - Which markets perform best
- By month - Track performance over time
- By CLV bucket - Performance at different edge levels

### Opportunities Explorer
Comprehensive interface for discovering and tracking betting opportunities:

- Filterable data table with status, CLV, bookmaker, market filters
- Date range selection
- Pagination for large datasets
- Export to CSV
- Track pending vs settled bets
- Sort by CLV, confidence, EV score, or date

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

# Predict line movement direction
from src.analyzers.movement_predictor import LineMovementPredictor

predictor = LineMovementPredictor()
predictor.load_model("models/line_movement_predictor.pkl")

# Get prediction for current odds
direction, magnitude, confidence = predictor.predict(features)
# Returns: ("UP", 0.15, 0.73) - line moving up 15% with 73% confidence
```

## Why This Matters

Most retail bettors chase wins. Sharp bettors chase CLV.

This system proves I understand:
- **Market efficiency** - Closing lines are the best estimate of true odds
- **Edge detection** - Consistent +CLV = long-term profit
- **Machine learning** - Ensemble models predicting line movement (65% accuracy, 23% improvement over baseline)
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
│   ├── features.py   # ML feature engineering (temporal + bookmaker features)
│   ├── movement_predictor.py # Ensemble line movement model (per-market weights)
│   ├── bet_settlement.py # Bet outcome tracking
│   └── bet_sizing.py # Kelly, fractional, confidence-weighted sizing
├── collectors/       # Data collection
│   ├── odds_api_client.py # The Odds API client
│   ├── odds_proccessor.py # Data processing & storage
│   └── nba_scores_client.py # NBA.com scores fetcher
├── models/           # SQLAlchemy database models
└── utils/            # Utilities (notifications, etc.)

frontend/
└── src/
    ├── Dashboard.tsx # React analytics dashboard w/ ML metrics
    └── components/
        └── OpportunitiesExplorer.tsx # Filterable opportunities table

scripts/
├── collect_odds.py          # Odds collection (opening + closing)
├── schedule_game_batches.py # Dynamic launchd scheduler
├── analyze_daily_clv.py     # Daily CLV report generation
├── fetch_game_scores.py     # NBA score fetching
├── train_movement_model.py  # ML model training (walk-forward validation)
├── track_opportunity_performance.py # Bet tracking
├── update_report_profit_stats.py    # ROI calculations
└── auto_retrain.py          # Auto-retraining on performance degradation

models/              # Trained ML models (git-ignored)
└── line_movement_predictor.pkl

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
- [x] Enhanced ML model (XGBoost + Random Forest ensemble)
- [x] Dynamic game-time batch scheduling
- [x] Kelly Criterion position sizing with multiple strategies
- [x] Advanced ML features (temporal, bookmaker, walk-forward validation)
- [x] Comprehensive opportunities explorer with filtering
- [x] Auto-retraining on model degradation
- [ ] Direct book scraping (Kalshi, Polymarket) for higher frequency data
- [ ] Arbitrage opportunity detection
- [ ] Live odds monitoring with real-time alerts

## Results

After collecting data for a few days, you can see which bookmakers consistently offer the best lines and optimal bet timing. The data speaks for itself - some books are beatable, most aren't.

## Notes

Built as a learning project to explore sports betting analytics and quantitative edge detection. Not financial advice. Bet responsibly.

---

**Stack:** Python • TypeScript • React • PostgreSQL • FastAPI • Docker
**Concepts:** Time-series data • Market efficiency • Statistical edge • Automated systems
