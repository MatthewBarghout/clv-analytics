# CLV Analytics Platform

A quantitative sports betting analytics system built to detect and exploit Closing Line Value (CLV) in real-time. The platform ingests live odds from multiple sportsbooks, applies an ensemble ML model to predict line movement, and surfaces +EV opportunities with systematic bankroll management — the same feedback loop used by professional sharp bettors.

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
                             ML Pipeline
                          (XGBoost + RF Ensemble)
                                    │
                             Opportunity Engine
                          (CLV + EV Calculation)
```

**Data Flow:**
1. Scheduled collectors poll opening and closing lines from 4 major sportsbooks
2. Lines stored in PostgreSQL time-series schema with composite indexes
3. ML ensemble predicts line movement direction and magnitude
4. CLV engine computes realized edge vs closing odds
5. Dashboard surfaces opportunities, performance analytics, and bankroll projections

---

## Tech Stack

**Backend**
| Component | Technology |
|---|---|
| API | FastAPI (Python) with in-memory TTL caching |
| ORM | SQLAlchemy 2.0 with eager-loaded joins (no N+1 queries) |
| Database | PostgreSQL with composite indexes on hot query paths |
| Migrations | Alembic |
| ML | XGBoost + scikit-learn ensemble with walk-forward validation |

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
| Scheduling | macOS launchd (10 AM / 6 PM ET collection windows) |
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
# Set ODDS_API_KEY in .env

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

Ensemble model combining XGBoost and Random Forest to forecast line movement direction and magnitude before games start.

**Current Performance:**
| Metric | Value |
|---|---|
| Direction Accuracy | 65.1% |
| Baseline | 40.7% |
| Improvement | +22.9% |
| MAE (price movement) | 0.0496 |
| R² Score | 0.58 |

**Top Predictive Features:**
1. Consensus line — 13.9%
2. Opening price — 12.1%
3. Line spread — 8.4%
4. Distance from consensus — 6.2%
5. Hours to game — 4.7%

**Model Details:**
- Temporal features: movement velocity, price volatility, cumulative drift, direction change counts
- Bookmaker features: sharp book detection (Pinnacle weighting), steam move signals, relative positioning
- Market-specific thresholds: h2h=0.02, spreads=0.01, totals=0.01
- Per-market ensemble weights optimized via grid search
- Walk-forward (expanding window) cross-validation for time-series integrity
- Auto-retraining triggered when live accuracy degrades >10%

### CLV Calculation Engine

Converts decimal odds to implied probabilities, computes entry vs closing edge, and aggregates by bookmaker and market type to identify the sharpest books and best entry windows.

```python
from src.analyzers.clv_calculator import CLVCalculator

calc = CLVCalculator()
clv = calc.calculate_clv(entry_odds=2.1, closing_odds=1.95)
# Returns: +3.66%
```

### Bankroll Simulation

Multi-strategy simulator projects long-term performance using historical settled bets and ML-predicted opportunities.

**Sizing Strategies:**
| Strategy | Description |
|---|---|
| Fixed Unit | Flat dollar amount per bet |
| Fractional | Fixed % of current bankroll |
| Kelly Criterion | f* = (bp − q) / b — mathematically optimal |
| Half Kelly | 50% Kelly for reduced variance |
| Confidence-Weighted | Scale by ML model confidence score |

**Risk Metrics:** Sharpe Ratio, Sortino Ratio, max drawdown, P&L curve, monthly breakdown

**Filters:** Bookmaker, market type, minimum CLV threshold, date range

Inputs are debounced (500ms) so the simulation fires once per input burst, not on every keystroke.

### Opportunities Explorer

Filterable table of all ML-detected +EV opportunities with full lifecycle tracking.

- Filter by status, CLV%, bookmaker, market, date range
- Sort by CLV, confidence, EV score, or date
- Pagination for large datasets
- Export to CSV
- Track pending vs settled with outcome recording

### Daily Performance Reports

Automated reports generated at 9 AM showing previous day's complete performance.

**Report Contents:**
- CLV analysis for all completed games
- Top 10 opportunities by CLV percentage
- Win/loss/push breakdown with profit calculations
- ROI by bookmaker and market type
- ML-predicted +EV opportunities for upcoming slate

### My Bets Tracker

Manual bet tracking interface with personal P&L dashboard.

- Add bets with game, bookmaker, market type, odds, and stake
- One-click settle as win / loss / push
- Summary cards: record, win rate, total profit, ROI
- Sortable bet history table

### Automated Data Collection

```
2:00 AM   — Fetch final game scores
3:00 AM   — Calculate ROI on tracked bets
9:00 AM   — Generate daily CLV report
9:30 AM   — Track new opportunities from report
10:30 AM  — Schedule closing line collection batches
```

Dynamic launchd scheduler creates collection batches 30 minutes before each game to capture closing lines with minimal latency. Handles rate limiting, retries, and stores three market types (moneyline, spreads, totals) across four books (Pinnacle, FanDuel, DraftKings, theScore Bet).

---

## API Reference

### Core Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/stats` | CLV statistics (cached 5 min) |
| `GET` | `/api/games` | Game list with odds snapshots |
| `GET` | `/api/bookmakers` | Bookmaker stats (cached 10 min) |
| `GET` | `/api/clv-history` | CLV trend over time (cached 5 min) |
| `GET` | `/api/reports` | Daily CLV reports |
| `GET` | `/api/user-bets` | Personal bet tracking |

### ML Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/ml/stats` | Model performance metrics |
| `GET` | `/api/ml/predictions/{game_id}` | Predicted vs actual lines |
| `POST` | `/api/ml/retrain` | Retrain with latest data |
| `GET` | `/api/ml/feature-importance` | Feature rankings |
| `GET` | `/api/ml/opportunities` | All filterable ML opportunities |
| `GET` | `/api/ml/upcoming-opportunities` | Games starting in next N hours |
| `GET` | `/api/ml/best-opportunities` | Top +EV picks for bankroll sim |

---

## Database Schema

Designed for high-throughput analytical queries:

```
Game ──── Team (home + away)
 │
 ├── OddsSnapshot (time-series per bookmaker)
 ├── ClosingLine (final line per bookmaker)
 └── BettingOutcome (win/loss/push result)

DailyCLVReport ──── OpportunityPerformance (tracked bets per report)

UserBet (personal tracking, independent of ML pipeline)
```

**Indexes (composite, on hot paths):**
- `Game(completed, commence_time)` — active game filters
- `OpportunityPerformance(report_id, game_id, result)` — report expansion queries
- `OpportunityPerformance(settled_at, result)` — settlement lookups
- `UserBet(result, game_date)` — personal bet summary queries

---

## Project Structure

```
src/
├── api/
│   ├── main.py               # Core FastAPI app — CLV, games, reports, user bets
│   └── ml_endpoints.py       # ML router — predictions, opportunities, retraining
├── analyzers/
│   ├── clv_calculator.py     # CLV calculation logic
│   ├── features.py           # ML feature engineering (temporal + bookmaker)
│   ├── movement_predictor.py # Ensemble line movement model (per-market weights)
│   ├── bet_settlement.py     # Bet outcome tracking
│   └── bet_sizing.py         # Kelly, fractional, confidence-weighted sizing
├── collectors/
│   ├── odds_api_client.py    # The Odds API client
│   ├── odds_proccessor.py    # Processing & storage pipeline
│   └── nba_scores_client.py  # NBA.com scores fetcher
├── models/
│   └── database.py           # SQLAlchemy models + composite indexes
└── utils/                    # Notifications, utilities

frontend/src/
├── Dashboard.tsx             # Orchestrator — state, fetching, tab routing
└── components/
    ├── MLStats.tsx           # ML model performance + feature importance chart
    ├── CLVOverview.tsx       # Market breakdown + CLV trend over time
    ├── BookmakerPerformance.tsx  # Bookmaker comparison table
    ├── DailyReports.tsx      # Daily reports with opportunity expansion (cached)
    ├── BankrollSimulator.tsx # Full simulation UI with debounced fetching
    ├── MyBets.tsx            # Personal bet tracking dashboard
    ├── OpportunitiesExplorer.tsx # Filterable opportunities table
    ├── GameDetailsModal.tsx  # Game odds snapshot detail view
    ├── GameAnalysis.tsx      # Per-game CLV analysis
    ├── GlassCard.tsx         # Reusable card component
    ├── TimeRangeSelector.tsx # Date range filter control
    └── AnimatedCounter.tsx   # Animated metric display

scripts/
├── collect_odds.py                  # Opening + closing line collection
├── schedule_game_batches.py         # Dynamic launchd batch scheduler
├── analyze_daily_clv.py             # Daily report generation
├── fetch_game_scores.py             # NBA score ingestion
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
- **N+1 elimination** — All ORM loops use pre-fetched bulk dictionaries (`SELECT ... WHERE id IN (...)`) instead of per-row queries; endpoints that previously ran 100–1000+ queries now run 2–5
- **In-memory TTL cache** — Stats, bookmaker data, and CLV history cached in-process with configurable TTL (5–10 min); repeat calls return in <1ms
- **Composite indexes** — Added on all major filter + sort columns used in analytical queries
- **Efficient aggregation** — `statistics.median()` replaces sorted array indexing; single-pass accumulators replace multi-pass generator chains; `SELECT COUNT(*)` replaces `len(execute().all())`

### Frontend
- **Component splitting** — Dashboard refactored from a 2,168-line monolith into 6 self-contained components; React only re-renders the affected component on state changes
- **Memoization** — `React.memo` on all display components; `useMemo` on all sort/map/chart-data computations; `useCallback` on all event handlers
- **Debounced simulation** — Bankroll sim API call fires 500ms after the last input change, not on every keystroke
- **Opportunity cache** — Expanded report rows cache fetched opportunities in a `Map` ref; re-expanding never triggers a duplicate network request

---

## Roadmap

- [x] Automated odds collection (opening + closing lines)
- [x] PostgreSQL time-series schema with proper indexing
- [x] CLV calculation engine
- [x] Ensemble ML model (XGBoost + Random Forest)
- [x] Walk-forward validation for time-series integrity
- [x] Advanced temporal + bookmaker features
- [x] Kelly Criterion and multi-strategy position sizing
- [x] Dynamic game-time batch scheduling
- [x] Comprehensive opportunities explorer
- [x] Auto-retraining on performance degradation
- [x] Bankroll simulation with multi-strategy support
- [x] Daily automated reports with full ROI tracking
- [x] N+1 query elimination + composite indexes
- [x] In-memory TTL caching on analytical endpoints
- [x] Frontend component split + full memoization
- [ ] Kalshi + Polymarket scraping — prediction market arbitrage detection
- [ ] Live odds WebSocket feed with real-time alerts
- [ ] Multi-sport expansion (NFL, NHL, MLB)

---

**Stack:** Python · TypeScript · React · PostgreSQL · FastAPI · XGBoost · Docker
**Concepts:** Time-series analytics · Market efficiency · Statistical edge detection · Quantitative position sizing · Automated systems
