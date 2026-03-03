# CLV Analytics Platform

A quantitative sports betting analytics system built to detect and exploit Closing Line Value (CLV) in real-time. The platform ingests live odds from multiple sportsbooks, applies an ensemble ML model to predict line movement, surfaces +EV opportunities with systematic bankroll management, and monitors live arbitrage between traditional sportsbooks and prediction markets (Kalshi, Polymarket).

---

## What is CLV?

Closing Line Value measures whether you secured better odds than the market's final assessment. Consistently beating the closing line is the single most reliable indicator of long-term profitability in sports betting — equivalent to consistently buying below fair value in financial markets.

**Example:**
- You bet Lakers +200 (3.00 decimal) at 10:00 AM
- Line closes at +180 (2.80 decimal)
- Your CLV: **+2.38%** — you outpriced the market

Replicate this across hundreds of bets with disciplined sizing, and the edge compounds.

---

## System Architecture

```
Odds API  ──►  Collector  ──►  PostgreSQL  ──►  FastAPI  ──►  React Dashboard
                                    │
                          ┌─────────┴─────────┐
                     ML Pipeline          Arb Engine
                  (XGBoost + RF)     (Kalshi + Polymarket)
                          │
                   Opportunity Engine
                 (CLV + EV + Tracking)
```

**Data Flow:**
1. Scheduled collectors poll opening and closing lines from multiple sportsbooks
2. Background scheduler polls Kalshi and Polymarket every 5 minutes for prediction market prices
3. Lines stored in PostgreSQL with composite indexes for time-series queries
4. ML ensemble predicts line movement direction and magnitude
5. CLV engine computes realized edge vs closing odds
6. Arb calculator identifies spread between sportsbook and prediction market implied odds
7. Dashboard surfaces opportunities, performance analytics, bankroll projections, and live arb feeds

---

## Tech Stack

**Backend**
| Component | Technology |
|---|---|
| API | FastAPI (Python) with in-memory TTL caching |
| ORM | SQLAlchemy 2.0 with eager-loaded joins (no N+1 queries) |
| Database | PostgreSQL with composite indexes on hot query paths |
| Migrations | Alembic |
| ML | XGBoost + scikit-learn ensemble, isotonic calibration, walk-forward validation |
| Scheduling | APScheduler (background polling) + macOS launchd (collection windows) |

**Frontend**
| Component | Technology |
|---|---|
| UI | React 18 + TypeScript |
| Build | Vite |
| Charts | Recharts |
| Styling | Tailwind CSS |
| Performance | React.memo + useMemo + useCallback throughout; 500ms debounce on sim inputs |

**Infrastructure**
| Component | Technology |
|---|---|
| Local DB | Docker Compose |
| Dependencies | Poetry |

---

## Quick Start

```bash
# 1. Start database
docker-compose up -d

# 2. Run migrations
alembic upgrade head

# 3. Configure environment
cp .env.example .env
# Set ODDS_API_KEY, KALSHI_API_KEY (optional) in .env

# 4. Install Python dependencies
poetry install

# 5. (Optional) Train ML model — enables predictions
poetry run python -m scripts.train_movement_model

# 6. Launch backend + frontend
./start.sh
```

- **Dashboard:** http://localhost:5173
- **API Docs:** http://localhost:8000/docs

---

## Key Features

### Machine Learning — Line Movement Prediction

Ensemble model combining XGBoost and Random Forest to forecast line movement direction and magnitude before games start. Tightened thresholds reduce over-signaling; only high-confidence, high-EV opportunities surface.

**Prediction Thresholds (revised):**
| Parameter | Value |
|---|---|
| Minimum confidence | 62% |
| Minimum EV score | 2.0 |
| Minimum predicted movement | ±2.5% (was ±1%) |
| Minimum time to game | 1 hour |
| Prediction cap (h2h) | ±8% (was ±10%) |
| Prediction cap (spreads/totals) | ±4% (was ±5%) |

**Top Predictive Features:**
1. Consensus line
2. Opening price
3. Line spread
4. Distance from consensus
5. Hours to game

**Model Details:**
- Temporal features: movement velocity, price volatility, cumulative drift, direction change counts
- Bookmaker features: sharp book detection (Pinnacle weighting), steam move signals, relative positioning
- Per-market ensemble weights optimized via grid search on validation set
- Walk-forward (expanding window) cross-validation for time-series integrity

### Best EV+ Opportunities

Today-only filtered view of the highest-confidence, highest-EV picks from the current day's upcoming slate.

- Filtered to today's games only (`today_only=true`)
- Minimum confidence 62%, minimum EV score 2.0, at least 1 hour to game
- Quarter-Kelly stake sizing recommendation per pick
- "Save for Tracking" button snapshots today's picks to the `BestEVPick` table
- Picks are settled automatically after games complete

### Prediction Market Arbitrage — Markets Tab

Live monitoring of spread between traditional sportsbook odds and prediction market (Kalshi, Polymarket) implied probabilities.

- Polls Kalshi and Polymarket every 5 minutes via APScheduler
- Auto-refreshes in the dashboard every 60 seconds
- Color-coded by spread: green ≥2%, yellow 1–2%, gray <1%
- Summary bar: total opportunities, strong arb count, best spread
- 7-day historical trend chart of average and max spreads
- Manual "Refresh Now" triggers a fresh backend poll

**Arb logic:**
```
arb_spread = (PM implied probability − SB implied probability) × 100
Positive spread = PM assigns higher probability than sportsbook → value on SB side
```

**Environment variable:**
```
KALSHI_API_KEY=<your-key>    # Required for Kalshi auth. Polymarket is public.
```

### Bankroll Simulation — Aligned with Best EV+

Multi-strategy simulator with a critical alignment fix: the simulation defaults to **Best EV+ picks only**, ensuring the bankroll curve reflects the same bets displayed in the Best EV+ tab.

**Data Source Toggle:**
| Source | Description |
|---|---|
| Best EV+ Picks Only | Only picks saved via the daily Best EV+ snapshot (default) |
| All Tracked Bets | Historical CLV-tracked opportunities from all reports |

**Sizing Strategies:**
| Strategy | Description |
|---|---|
| Fixed Unit | Flat dollar amount per bet |
| Fractional | Fixed % of current bankroll |
| Kelly Criterion | f* = (bp − q) / b — mathematically optimal |
| Half Kelly | 50% Kelly for reduced variance |
| Confidence-Weighted | Scale by ML model confidence score |

**Risk Metrics:** Sharpe Ratio, Sortino Ratio, max drawdown, P&L curve, per-bookmaker and per-market breakdown.

### CLV Calculation Engine

Converts decimal odds to implied probabilities, computes entry vs closing edge, and aggregates by bookmaker and market type to identify the sharpest books and best entry windows.

```python
from src.analyzers.clv_calculator import CLVCalculator

calc = CLVCalculator()
clv = calc.calculate_clv(entry_odds=2.1, closing_odds=1.95)
# Returns: +3.66%
```

### Daily Performance Reports

Automated reports generated at 9 AM showing previous day's complete performance.

**Report Contents:**
- CLV analysis for all completed games
- Top opportunities by CLV percentage
- Win/loss/push breakdown with profit calculations
- ROI by bookmaker and market type

### My Bets Tracker

Manual bet tracking interface with personal P&L dashboard.

- Add bets with game, bookmaker, market type, odds, and stake
- One-click settle as win / loss / push
- Summary cards: record, win rate, total profit, ROI

### Automated Data Collection

```
2:00 AM   — Fetch final game scores
3:00 AM   — Calculate ROI on tracked bets
9:00 AM   — Generate daily CLV report; save Best EV+ picks
9:30 AM   — Track new opportunities from report
10:30 AM  — Schedule closing line collection batches
Every 5m  — Poll Kalshi + Polymarket for arb opportunities (APScheduler)
```

---

## API Reference

### Core Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/stats` | CLV statistics (cached 5 min) |
| `GET` | `/api/games` | Game list with odds snapshots |
| `GET` | `/api/bookmakers` | Bookmaker stats (cached 10 min) |
| `GET` | `/api/clv-history` | CLV trend over time (cached 5 min) |
| `GET` | `/api/daily-reports` | Daily CLV reports |
| `GET` | `/api/bankroll-simulation` | P&L simulation (`source=best_ev\|all`) |
| `GET` | `/api/user-bets` | Personal bet tracking |

### ML Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/ml/stats` | Model performance metrics |
| `GET` | `/api/ml/best-opportunities` | Today's top +EV picks (`today_only`, `min_ev_score`, `min_confidence`) |
| `GET` | `/api/ml/best-ev-history` | Historical Best EV picks with settled results |
| `POST` | `/api/ml/save-daily-picks` | Snapshot today's picks to tracking table |
| `POST` | `/api/ml/settle-picks` | Settle pending picks against outcomes |
| `GET` | `/api/ml/predictions/{game_id}` | Predicted vs actual line movement |
| `POST` | `/api/ml/retrain` | Retrain with latest data |
| `GET` | `/api/ml/feature-importance` | Feature importance rankings |
| `GET` | `/api/ml/opportunities` | All filterable ML opportunities |

### Prediction Market Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/arb-opportunities` | Active arb opportunities (sorted by spread) |
| `GET` | `/api/arb-history` | Historical arb data with date range filter |
| `POST` | `/api/arb/refresh` | Manually trigger a fresh poll |

---

## Database Schema

```
Game ──── Team (home + away)
 │
 ├── OddsSnapshot (time-series per bookmaker)
 ├── ClosingLine (final line per bookmaker)
 └── BettingOutcome (win/loss/push result)

DailyCLVReport ──── OpportunityPerformance (CLV-tracked bets)

BestEVPick (ML-selected daily picks with lifecycle tracking)

PredictionMarketArb (Kalshi/Polymarket vs sportsbook spreads)

UserBet (personal tracking, independent of ML pipeline)
```

**Indexes (composite, on hot paths):**
- `Game(completed, commence_time)`
- `OpportunityPerformance(report_id, game_id, result)`
- `OpportunityPerformance(settled_at, result)`
- `BestEVPick(report_date)`, `BestEVPick(result, report_date)`
- `PredictionMarketArb(timestamp, is_active)`, `PredictionMarketArb(market_source, arb_spread)`
- `UserBet(result, game_date)`

---

## Project Structure

```
src/
├── api/
│   ├── main.py               # Core FastAPI app — CLV, games, reports, arb endpoints
│   └── ml_endpoints.py       # ML router — predictions, opportunities, Best EV tracking
├── analyzers/
│   ├── clv_calculator.py     # CLV calculation logic
│   ├── features.py           # ML feature engineering (temporal + bookmaker)
│   ├── movement_predictor.py # Ensemble model with tightened prediction caps
│   ├── bet_settlement.py     # Bet outcome tracking
│   └── bet_sizing.py         # Kelly, fractional, confidence-weighted sizing
├── collectors/
│   ├── odds_api_client.py    # The Odds API client
│   ├── odds_proccessor.py    # Processing & storage pipeline
│   ├── kalshi_client.py      # Kalshi prediction market REST API client
│   ├── polymarket_client.py  # Polymarket CLOB API client (public)
│   ├── arb_calculator.py     # Arb spread calculation and opportunity detection
│   └── nba_scores_client.py  # NBA.com scores fetcher
├── models/
│   └── database.py           # SQLAlchemy models + composite indexes
└── utils/                    # Notifications, utilities

frontend/src/
├── Dashboard.tsx             # Orchestrator — state, fetching, tab routing
└── components/
    ├── MLStats.tsx                # ML model performance + feature importance chart
    ├── CLVOverview.tsx            # Market breakdown + CLV trend over time
    ├── BookmakerPerformance.tsx   # Bookmaker comparison table
    ├── DailyReports.tsx           # Daily reports with opportunity expansion
    ├── BankrollSimulator.tsx      # Simulation UI with source toggle + debounce
    ├── BestEVOpportunities.tsx    # Today's best picks with Kelly sizing
    ├── ArbOpportunities.tsx       # Live Kalshi/Polymarket arb tab
    ├── MyBets.tsx                 # Personal bet tracking dashboard
    ├── OpportunitiesExplorer.tsx  # Filterable opportunities table
    ├── GameDetailsModal.tsx       # Game odds snapshot detail view
    ├── GameAnalysis.tsx           # Per-game CLV analysis
    ├── GlassCard.tsx              # Reusable card component
    ├── TimeRangeSelector.tsx      # Date range filter control
    └── AnimatedCounter.tsx        # Animated metric display

scripts/
├── collect_odds.py                  # Opening + closing line collection
├── schedule_game_batches.py         # Dynamic launchd batch scheduler
├── analyze_daily_clv.py             # Daily report generation
├── fetch_game_scores.py             # Score ingestion
├── track_opportunity_performance.py # Bet lifecycle tracking
├── update_report_profit_stats.py    # ROI calculations
├── train_movement_model.py          # Walk-forward model training
└── auto_retrain.py                  # Auto-retraining on performance degradation

models/                     # Trained ML artifacts (git-ignored)
migrations/                 # Alembic schema migrations
launchd/                    # macOS scheduling plists
```

---

## Performance Optimizations

### Backend
- **N+1 elimination** — All ORM loops use pre-fetched bulk dictionaries instead of per-row queries
- **In-memory TTL cache** — Stats, bookmaker data, and CLV history cached in-process (5–10 min TTL)
- **Composite indexes** — Added on all major filter + sort columns in analytical queries
- **Efficient aggregation** — `statistics.median()`, single-pass accumulators, `SELECT COUNT(*)`

### Frontend
- **Component splitting** — Dashboard split into 8 self-contained components
- **Memoization** — `React.memo` on all display components; `useMemo` on sort/chart computations
- **Debounced simulation** — Bankroll sim fires 500ms after last input change
- **Opportunity cache** — Expanded report rows cache in `Map` ref — no duplicate requests

---

## Changelog

### Latest Release
- **Prediction Market Arbitrage:** Live Kalshi + Polymarket scraping, 5-minute background polling, Markets tab with pulsing indicator when opportunities exist
- **ML Accuracy Improvements:** Raised minimum movement threshold to ±2.5%, confidence floor to 62%, EV score floor to 2.0; tightened prediction caps (h2h: ±8%, spreads/totals: ±4%) to eliminate low-conviction noise
- **Best EV+ / Bankroll Alignment:** Best EV+ tab shows today-only picks; bankroll sim defaults to Best EV+ picks as data source, ensuring 1:1 alignment between what you see and what is simulated
- **BestEVPick Tracking:** Daily picks saved to dedicated table with lifecycle settlement against game outcomes

---

**Stack:** Python · TypeScript · React · PostgreSQL · FastAPI · XGBoost · Docker
**Concepts:** Time-series analytics · Market efficiency · Statistical edge detection · Quantitative position sizing · Prediction market arbitrage · Automated systems
