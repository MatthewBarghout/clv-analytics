# CLV Analytics Platform

A production-grade quantitative sports betting analytics system built to detect and exploit Closing Line Value (CLV) in real-time. The platform ingests live odds from multiple sportsbooks, runs an ensemble ML model to predict line movement, surfaces +EV opportunities with systematic bankroll management, monitors live arbitrage between sportsbooks and prediction markets, and operates a paper trading engine that cross-references Kalshi prices against Polymarket and Metaculus forecasts.

---

## What is CLV?

Closing Line Value measures whether you secured better odds than the market's final consensus. Consistently beating the closing line is the single most reliable indicator of long-term profitability in sports betting — equivalent to consistently buying below fair value in financial markets.

**Example:**
- You bet Lakers +200 (3.00 decimal) at 10:00 AM
- Line closes at +180 (2.80 decimal)
- Your CLV: **+2.38%** — you outpriced the market

Replicate this across hundreds of bets with disciplined sizing and the edge compounds.

---

## System Architecture

```
Odds API  ──►  Collector  ──►  PostgreSQL  ──►  FastAPI  ──►  React Dashboard
                                    │
                 ┌──────────────────┼───────────────────────┐
            ML Pipeline         Arb Engine          PM Signal Engine
         (XGBoost + RF)         (Kalshi)        (Kalshi × Polymarket
                │                                   × Metaculus)
       Opportunity Engine
     (CLV + EV + Tracking)
```

**Data flow:**
1. launchd collectors pull opening lines at 10 AM and closing lines before games
2. APScheduler polls Kalshi every 5 minutes for arb spreads
3. APScheduler polls Kalshi every 10 minutes for cross-platform signal generation
4. ML ensemble predicts line movement direction and magnitude
5. CLV engine computes realized edge vs closing odds
6. Best EV+ picks are snapshotted at 11 AM and settled at 3 AM
7. Paper trades are opened on signal, settled when Kalshi markets resolve

---

## Tech Stack

**Backend**
| Component | Technology |
|---|---|
| API | FastAPI + in-memory TTL caching |
| ORM | SQLAlchemy 2.0 — explicit JOINs, bulk-fetched dicts, no N+1 queries |
| Database | PostgreSQL — composite indexes on all hot filter/sort paths |
| Migrations | Alembic |
| ML | XGBoost + scikit-learn ensemble, isotonic calibration, walk-forward CV |
| Scheduling | APScheduler (in-process, polling) + macOS launchd (collection windows) |

**Frontend**
| Component | Technology |
|---|---|
| UI | React 18 + TypeScript (strict) |
| Build | Vite |
| Charts | Recharts |
| Styling | Tailwind CSS |
| Performance | `React.memo` + `useMemo` + `useCallback` throughout; 500ms debounce on sim inputs |

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

# 2. Apply migrations
poetry run alembic upgrade head

# 3. Configure environment
cp .env.example .env
# Set ODDS_API_KEY, KALSHI_API_KEY, DATABASE_URL

# 4. Install dependencies
poetry install

# 5. (Optional) Train ML model — required to enable predictions
poetry run python -m scripts.train_movement_model --walk-forward

# 6. Launch backend + frontend
./start.sh
```

- **Dashboard:** http://localhost:5173
- **API Docs:** http://localhost:8000/docs

---

## Features

### Machine Learning — Line Movement Prediction

Ensemble of XGBoost and Random Forest predicting line movement direction and magnitude before game time. Both classifiers are wrapped in `CalibratedClassifierCV` (isotonic, 5-fold) to produce calibrated probability estimates. Class imbalance is corrected via balanced sample weights so the model doesn't ignore "STAY" signals.

**Current thresholds (do not loosen without walk-forward evidence):**
| Parameter | Value | Rationale |
|---|---|---|
| Minimum confidence | 62% | Calibrated probability floor |
| Minimum EV score | 2.0 | Filters noise before surfacing picks |
| Minimum predicted movement | ±2.5% | Below this is within model uncertainty |
| Minimum time to game | 1 hour | Line has already moved if less |
| Prediction cap (h2h) | ±8% | Prevents over-sized signals |
| Prediction cap (spreads/totals) | ±4% | Tighter due to noisier markets |

**Current validated accuracy (walk-forward, 5 chronological folds):**
- 56.6% ± 0.7% vs 40.1% baseline
- Consistent across all folds — model generalizes across time

**Feature groups:**
- *Temporal:* movement velocity, price volatility, cumulative drift, direction change count
- *Bookmaker:* sharp-book positioning (Pinnacle), steam move signals, consensus distance, relative spread

**Training:**
```bash
poetry run python -m scripts.train_movement_model --walk-forward
```

### Best EV+ Opportunities

Today-only filtered view of the highest-confidence picks from the day's upcoming slate. Cards grouped by sport with Quarter-Kelly stake sizing.

- Filters: today's games only, confidence ≥ 62%, EV score ≥ 2.0, ≥ 1 hour to game
- Deduplicates on `(game_id, market_type, outcome_name)` — multiple bookmaker snapshots collapse to one card keeping the highest EV
- "Save for Tracking" snapshots today's picks to `BestEVPick` with lifecycle tracking
- Settlement runs automatically at 3 AM after scores are ingested

### Best EV+ Lifecycle

```
10:00 AM  — Opening odds collected (launchd)
11:00 AM  — save_daily_picks() snapshots top picks → BestEVPick (result=pending)
  2:00 AM — fetch_game_scores() ingests final scores
  3:00 AM — settle_picks() settles all pending picks → BestEVPick (result=win/loss/push)
```

All three market types settle correctly:
- **h2h** — compares outcome to `BettingOutcome.winner`
- **spreads** — `cover_margin = team_margin + point_line`; positive = win
- **totals** — `BettingOutcome.total_points` vs `point_line`

### Prediction Market Arbitrage — Markets Tab

Live monitoring of spread between traditional sportsbook implied probabilities and Kalshi.

- Polls Kalshi every 5 minutes via APScheduler
- Series-based fetch (KXNBAGAME, KXMLBGAME, KXNHLGAME, KXNFLGAME, KXNBAPTS, KXNBAREB, KXMMA, KXSOCCER) with keyword scan fallback when all series are off-season
- Bid/ask midpoint used for probability estimates (more accurate than bid-only)
- Auto-refreshes every 60 seconds; manual "Refresh Now" triggers a fresh backend poll
- Color-coded: green ≥ 2%, yellow 1–2%, gray < 1%
- 7-day historical trend chart of average and max spreads

```
arb_spread = (Kalshi implied prob − SB implied prob) × 100
Positive spread → Kalshi assigns higher probability than sportsbook
```

### Prediction Market Paper Trading — Pred Markets Tab

Cross-platform signal engine that compares Kalshi prices against Polymarket and Metaculus, opens simulated trades on divergence, and settles them when markets resolve.

**Signal generation (every 10 minutes):**
1. Fetch all active Kalshi markets across sports and non-sports series via `get_all_markets()`
2. For each non-game-level market, query Polymarket and Metaculus for the same event
3. Compute weighted fair value: Polymarket 45%, Metaculus 35%, Kalshi momentum 20%
4. If `max(edge_YES, edge_NO) > 6%` → open a paper trade
5. Skip if an open position already exists for that ticker

**Kalshi series covered:**

| Category | Series |
|---|---|
| Sports | KXNBAGAME, KXMLBGAME, KXNHLGAME, KXNFLGAME, KXNBAPTS, KXNBAREB, KXMMA, KXSOCCER |
| Sports futures | KXNBA, KXMLB, KXNHL, KXNFL |
| Crypto | KXBTC, KXETH |
| Politics | KXPRES, KXFED, KXHOUSE, KXSENATE |
| Economics | KXECON |

Game-level series (individual match outcomes) are excluded from signal generation — they won't have Polymarket equivalents and produce false matches. Non-sports markets are matched against Polymarket's `politics` and `crypto` tag feeds.

**Fair value redistribution:** if a source is unavailable (returns None), its weight is redistributed proportionally to the present sources so the estimate stays calibrated.

**Strategy tags** include the market category: `cross_platform_sports_v1`, `cross_platform_crypto_v1`, `cross_platform_politics_v1`, `cross_platform_economics_v1`. Category is inferred from the Kalshi ticker prefix at signal time.

**Polymarket cache** is refreshed every 30 minutes by a dedicated APScheduler job (`poly_cache_refresh`) that fetches sports, politics, and crypto tag feeds. The 10-minute price collection job reads from this cache without blocking on a network call, falling back to an inline refresh only on first run.

**Position sizing (Quarter-Kelly):**
```
size = (edge / (1 − entry_price)) × 0.25 × $1000
Clamped: $25 minimum, $200 maximum
```

**Settlement (daily 3:15 AM):**
- Fetches Kalshi market status via API
- Settles when `status == "finalized"` and `result` is `"yes"` or `"no"`
- WIN pnl: `(1.0 − entry_price) × (size_usd / entry_price)`
- LOSS pnl: `−size_usd`

**Dashboard (Pred Markets tab):**
- Stats bar: total trades, open positions, win rate, total P&L
- Open positions table: ticker, event, side, entry price, size, strategy
- Trade history table: same columns + resolution, realized P&L, entry/exit dates
- Recent signals table: Kalshi vs Polymarket vs Metaculus with divergence score

No API keys required — Polymarket and Metaculus are public APIs.

### Multi-Sport Collection

| Sport | Key | Score Source |
|---|---|---|
| NBA | `basketball_nba` | NBA.com API (free) |
| MLB | `baseball_mlb` | MLB Stats API — statsapi.mlb.com (free) |

NFL to be added when preseason begins (August). Adding a new sport requires one line in `collect_odds.py` and routing in `fetch_game_scores.py` — no DB migrations, no model changes.

### Bankroll Simulation

Multi-strategy simulator aligned 1:1 with Best EV+ picks.

**Data source toggle:**
| Source | Description |
|---|---|
| Best EV+ Picks Only | Only picks saved via the daily 11 AM snapshot (default) |
| All Tracked Bets | Historical CLV opportunities from all reports |

**Sizing strategies:**
| Strategy | Description |
|---|---|
| Fixed Unit | Flat dollar amount per bet |
| Fractional Kelly | Fixed % of current bankroll |
| Full Kelly | f* = (bp − q) / b |
| Half Kelly | 50% Kelly for reduced variance |
| Confidence-Weighted | Scales stake by ML model confidence |

Risk metrics: Sharpe ratio, Sortino ratio, max drawdown, P&L curve, per-bookmaker and per-market breakdown.

### CLV Calculation Engine

Converts decimal odds to implied probabilities, computes entry vs closing edge, and aggregates across bookmakers and market types.

```python
from src.analyzers.clv_calculator import CLVCalculator

clv = CLVCalculator().calculate_clv(entry_odds=2.1, closing_odds=1.95)
# Returns: +3.66%
```

### Daily Reports

Automated report covering all completed games from the prior day.

- CLV analysis across all bookmakers and market types
- Top opportunities ranked by CLV percentage
- Win/loss/push breakdown with profit calculations
- ROI by bookmaker, ROI by market type

### My Bets Tracker

Manual bet tracking with personal P&L dashboard.

- Add bets with game, bookmaker, market type, odds, and stake
- One-click settle as win / loss / push
- Summary: record, win rate, total profit, ROI

---

## Automated Schedule

```
10:00 AM  — Collect opening odds (NBA + MLB)           launchd
11:00 AM  — Snapshot Best EV+ picks                    launchd
 2:00 AM  — Fetch game scores (NBA + MLB)              launchd
 3:00 AM  — Settle Best EV+ picks                      launchd
 3:15 AM  — Settle paper trades                        APScheduler
 Every 5m — Poll Kalshi for arb opportunities          APScheduler
Every 10m — Collect PM prices + generate signals       APScheduler
Every 30m — Refresh Polymarket market cache            APScheduler
```

---

## Environment Variables

```bash
ODDS_API_KEY=<required>       # The Odds API — odds ingestion
KALSHI_API_KEY=<required>     # Kalshi RSA key ID (KALSHI_KEY_ID)
KALSHI_PRIVATE_KEY=<required> # Kalshi RSA private key PEM
DATABASE_URL=<postgres dsn>   # e.g. postgresql://user:pass@localhost/clv
```

Polymarket and Metaculus are public APIs — no keys needed.

---

## API Reference

### Core

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/stats` | CLV statistics (cached 5 min) |
| `GET` | `/api/games` | Games with odds snapshots |
| `GET` | `/api/bookmakers` | Bookmaker performance (cached 10 min) |
| `GET` | `/api/clv-history` | CLV trend over time (cached 5 min) |
| `GET` | `/api/daily-reports` | Daily CLV reports |
| `GET` | `/api/bankroll-simulation` | P&L simulation (`source=best_ev\|all`) |
| `GET` | `/api/user-bets` | Personal bet list |
| `POST` | `/api/user-bets` | Add a bet |
| `PATCH` | `/api/user-bets/{id}` | Settle a bet |

### ML

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/ml/stats` | Model performance metrics |
| `GET` | `/api/ml/best-opportunities` | Today's top +EV picks (`today_only`, `min_ev_score`, `min_hours_to_game`) |
| `GET` | `/api/ml/best-ev-history` | Historical Best EV picks with outcomes |
| `POST` | `/api/ml/save-daily-picks` | Snapshot today's picks |
| `POST` | `/api/ml/settle-picks` | Settle pending picks |
| `POST` | `/api/ml/retrain` | Retrain model with latest data |
| `GET` | `/api/ml/feature-importance` | Feature importance rankings |
| `GET` | `/api/ml/predictions/{game_id}` | Predicted vs actual line movement |
| `GET` | `/api/ml/opportunities` | Filterable full opportunity list |

### Prediction Markets

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/arb-opportunities` | Active Kalshi arb opps (sorted by spread) |
| `GET` | `/api/arb-history` | Historical arb data |
| `POST` | `/api/arb/refresh` | Trigger fresh Kalshi poll |
| `GET` | `/api/paper-trades` | Paper trade list (`is_open`, `strategy_tag`, `limit`) |
| `GET` | `/api/paper-trades/stats` | Aggregate stats (win rate, P&L, by-strategy, by-month) |
| `GET` | `/api/cross-platform-signals` | Recent signals sorted by divergence (`min_divergence`) |

---

## Database Schema

```
Sport ──── Team ──── Game
                      │
                      ├── OddsSnapshot      (time-series per bookmaker)
                      ├── ClosingLine        (final line per bookmaker)
                      └── BettingOutcome     (win/loss/push result)

DailyCLVReport ──── OpportunityPerformance  (CLV-tracked bet outcomes)

BestEVPick                                  (daily ML picks, lifecycle: pending → settled)
PredictionMarketArb                         (Kalshi vs sportsbook spreads)
KalshiMarketPrice                           (time-series price snapshots per ticker)
CrossPlatformSignal                         (Kalshi × Polymarket × Metaculus divergence)
PaperTrade                                  (simulated trades from signal engine)
UserBet                                     (personal tracking, independent of ML)
```

**Key composite indexes:**
- `Game(completed, commence_time)`, `Game(sport_id, commence_time)`
- `OddsSnapshot(game_id, bookmaker_id, market_type, timestamp)` — unique constraint
- `BestEVPick(result, report_date)`, `BestEVPick(game_id)`
- `PredictionMarketArb(timestamp, is_active)`, `(market_source, arb_spread)`
- `KalshiMarketPrice(market_ticker, timestamp)`
- `CrossPlatformSignal(kalshi_ticker, timestamp)`, `(divergence_score)`
- `PaperTrade(market_ticker, is_open)`, `(strategy_tag, resolution_result)`
- `OpportunityPerformance(report_id, game_id, result)`, `(settled_at, result)`
- `UserBet(result, game_date)`

---

## Project Structure

```
src/
├── api/
│   ├── main.py                  # Core FastAPI app — CLV, arb, paper trades, bankroll
│   ├── ml_endpoints.py          # ML router — predictions, Best EV+ lifecycle
│   └── schemas.py               # Pydantic response schemas
├── analyzers/
│   ├── clv_calculator.py        # CLV % computation (entry vs closing odds)
│   ├── features.py              # Feature engineering — temporal + bookmaker signals
│   ├── movement_predictor.py    # XGBoost+RF ensemble, calibration, caps, walk-forward
│   ├── pm_signal_generator.py   # Cross-platform signal engine (Kalshi × Poly × Meta)
│   ├── bet_settlement.py        # Bet outcome tracking
│   └── bet_sizing.py            # Kelly, fractional, confidence-weighted sizing
├── collectors/
│   ├── odds_api_client.py       # The Odds API client
│   ├── odds_proccessor.py       # Odds → DB pipeline (sport-agnostic)
│   ├── kalshi_client.py         # Kalshi REST API — RSA-PSS auth, series → events → markets
│   ├── polymarket_client.py     # Polymarket Gamma API — public, no auth
│   ├── metaculus_client.py      # Metaculus API — community_prediction median
│   ├── arb_calculator.py        # Arb spread calc + fuzzy event matching
│   ├── nba_scores_client.py     # NBA.com scores fetcher
│   └── mlb_scores_client.py     # MLB Stats API client (statsapi.mlb.com)
└── models/
    └── database.py              # All SQLAlchemy models + composite indexes

frontend/src/
├── Dashboard.tsx                # Orchestrator — tab routing, state, lazy loading
└── components/
    ├── MLStats.tsx              # Model performance + feature importance
    ├── CLVOverview.tsx          # Market breakdown + CLV trend chart
    ├── BookmakerPerformance.tsx # Bookmaker comparison table
    ├── DailyReports.tsx         # Daily reports with expandable opportunity rows
    ├── BankrollSimulator.tsx    # P&L sim with source toggle + debounced inputs
    ├── BestEVOpportunities.tsx  # Today's picks — card layout, sport filter, grouped
    ├── ArbOpportunities.tsx     # Kalshi arb tab — 60s auto-refresh, history chart
    ├── PredictionMarkets.tsx    # Paper trading — positions, history, signals
    ├── MyBets.tsx               # Personal bet tracker
    ├── OpportunitiesExplorer.tsx
    ├── GameDetailsModal.tsx
    ├── GameAnalysis.tsx
    ├── GlassCard.tsx
    ├── TimeRangeSelector.tsx
    └── AnimatedCounter.tsx

scripts/
├── collect_odds.py                  # Multi-sport odds collection (NBA + MLB)
├── fetch_game_scores.py             # Score ingestion — routes by sport key
├── analyze_daily_clv.py             # Daily report generation
├── track_opportunity_performance.py # Bet lifecycle tracking
├── update_report_profit_stats.py    # ROI calculations
├── schedule_game_batches.py         # Dynamic launchd batch scheduler
├── train_movement_model.py          # Walk-forward model training
└── auto_retrain.py                  # Auto-retraining on degradation

migrations/                  # Alembic migration history
models/                      # Trained ML artifacts (git-ignored)
launchd/                     # macOS scheduling plists
```

---

## Performance Notes

### Backend
- N+1 queries eliminated — pre-fetched bulk dicts for all ORM loops, explicit JOINs when related data is always needed, `sqlalchemy.orm.aliased` when joining same table twice
- In-memory TTL cache on `/api/stats` (5 min), `/api/bookmakers` (10 min), `/api/clv-history` (5 min)
- `SELECT COUNT(*)` not `len(query.all())`; `statistics.median()` not manual sort

### Frontend
- All display components wrapped in `React.memo`
- Sort and chart data memoized with `useMemo`
- Bankroll sim inputs debounced 500ms via `useRef + setTimeout`
- Expanded report rows cached in a `Map` ref — no duplicate fetches
- Games tab loads lazily on first visit via `loadedTabs` ref

---

## ML Pipeline Fixes (applied — do not revert)

| Bug | Fix |
|---|---|
| Live prediction features were zeroed out | Call `calculate_temporal_features()` + `calculate_bookmaker_features()` before every prediction |
| Train/test split leaked future into training | Chronological cut at 80th percentile by `snapshot_timestamp` |
| Spread + total picks never settled | Added `point_line` column to `BestEVPick`; full settlement logic for all market types |
| Confidence scores were uncalibrated | Wrapped both classifiers in `CalibratedClassifierCV(cv=5, method='isotonic')` |
| STAY class chronically under-predicted | `class_weight='balanced'` on RF; `compute_sample_weight('balanced')` on XGB |

---

**Stack:** Python · TypeScript · React · PostgreSQL · FastAPI · XGBoost · Docker  
**Concepts:** Closing line value · Line movement prediction · Prediction market arbitrage · Cross-platform signal generation · Quantitative position sizing · Walk-forward validation
