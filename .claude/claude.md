# CLV Analytics — Engineering Directives

You are a senior quantitative software engineer. This is a production sports betting analytics platform built to detect and exploit Closing Line Value (CLV) in real-time. Treat every decision as if it carries real financial risk.

---

## Project Overview

A full-stack CLV analytics system with:
- **Live odds ingestion** from multiple sportsbooks via The Odds API
- **ML ensemble** (XGBoost + Random Forest) predicting line movement direction and magnitude
- **Best EV+ opportunities** — today-only, high-confidence picks with Quarter-Kelly sizing
- **Prediction market arbitrage** — live Kalshi vs. sportsbook spread monitoring (series-based fetch)
- **Bankroll simulation** — aligned 1:1 with Best EV+ picks, multi-strategy sizing
- **Automated scheduling** — opening/closing line collection, daily reports, arb polling

---

## Tech Stack

**Backend:** FastAPI + SQLAlchemy 2.0 (PostgreSQL) + Alembic + APScheduler
**ML:** XGBoost + scikit-learn ensemble, isotonic calibration, walk-forward validation
**Frontend:** React 19 + TypeScript + Vite + Recharts + **Tailwind v4 (CSS-first)**
**Infra:** Docker Compose (DB), Poetry (deps), macOS launchd (scheduling)

---

## Key File Map

| Path | Purpose |
|------|---------|
| `src/api/main.py` | Core FastAPI app — CLV, games, reports, bankroll sim, arb endpoints (~1500 lines) |
| `src/api/ml_endpoints.py` | ML router — predictions, Best EV+ opportunities, BestEVPick tracking |
| `src/models/database.py` | All SQLAlchemy models + composite indexes |
| `src/analyzers/movement_predictor.py` | XGBoost+RF ensemble — train, predict, calibration, caps, walk-forward |
| `src/analyzers/clv_calculator.py` | CLV % computation (entry vs closing odds) |
| `src/analyzers/features.py` | ML feature engineering (temporal + bookmaker signals) |
| `src/collectors/kalshi_client.py` | Kalshi REST API client — RSA-PSS auth, series → events → markets |
| `src/collectors/polymarket_client.py` | Polymarket Gamma API — public, no auth; arb + signal lookups |
| `src/collectors/metaculus_client.py` | Metaculus public API — `get_forecast(keyword)` → community prediction median |
| `src/collectors/arb_calculator.py` | Arb spread calc + fuzzy event matching |
| `src/collectors/mlb_scores_client.py` | MLB Stats API client — free, no auth (statsapi.mlb.com); spans +/-1 day |
| `src/collectors/nba_scores_client.py` | NBA scores via **ESPN** public API — free, no auth; sets NO User-Agent (see below) |
| `src/analyzers/pm_signal_generator.py` | Cross-platform signal engine — `PMSignalGenerator` with fair_value(), generate_signal(), kelly_size() |
| `scripts/collect_odds.py` | Multi-sport odds collection — NBA + MLB, closing lines, snapshots |
| `scripts/fetch_game_scores.py` | Score fetcher — NBA via ESPN, MLB via MLB Stats API; supports --backfill |
| `frontend/src/Dashboard.tsx` | Orchestrator — top-level view routing, lazy tab loads, sport filter |
| `frontend/src/components/BestEVOpportunities.tsx` | Today-only best picks — card layout, deduplication, sport filter + grouped by sport |
| `frontend/src/components/ArbOpportunities.tsx` | Markets tab — Kalshi arb, 60s auto-refresh |
| `frontend/src/components/PredictionMarkets.tsx` | Pred Markets tab — paper trades, open positions, signals; 60s auto-refresh |
| `frontend/src/components/BankrollSimulator.tsx` | P&L sim with Best EV+ / All toggle |
| `frontend/src/components/MLStats.tsx` | Model performance + feature importance |

---

## Database Models

| Model | Table | Purpose |
|-------|-------|---------|
| `Game` | `games` | Game metadata, completion status |
| `OddsSnapshot` | `odds_snapshots` | Time-series odds per bookmaker |
| `ClosingLine` | `closing_lines` | Final line per bookmaker per game |
| `DailyCLVReport` | `daily_clv_reports` | Daily report aggregates |
| `OpportunityPerformance` | `opportunity_performances` | CLV-tracked bets with outcomes |
| `BestEVPick` | `best_ev_picks` | Daily ML-selected picks with lifecycle tracking |
| `PredictionMarketArb` | `prediction_market_arb` | Kalshi vs sportsbook spreads |
| `KalshiMarketPrice` | `kalshi_market_prices` | Time-series price snapshots per Kalshi ticker |
| `CrossPlatformSignal` | `cross_platform_signals` | Divergence events — Kalshi vs Polymarket/Metaculus |
| `PaperTrade` | `paper_trades` | Simulated trades from the signal engine |
| `UserBet` | `user_bets` | Manual personal bet tracking |

**`BestEVPick` has a `point_line` column (nullable float)** — stores the spread or total line at time of pick. Null for h2h. Required for spread/total settlement. Added in migration `ae018c662ca2`.

**Migrations:** Always use `poetry run alembic revision --autogenerate -m "description"` then `poetry run alembic upgrade head`. Never modify migration files manually unless absolutely necessary.

---

## ML Model Thresholds (Current — Do Not Loosen Without Data)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `MIN_MOVEMENT` | `0.025` | Require ≥2.5% predicted delta — filters noise |
| `MIN_CONFIDENCE` | `0.62` | Confidence floor — below this, skip |
| `min_ev_score` default | `2.0` | EV floor on best-opportunities endpoint |
| `min_hours_to_game` default | `1.0` | No late bets — line has already moved |
| h2h prediction cap | `±0.08` | Was ±0.10 — prevents over-sized signals |
| spreads/totals cap | `±0.04` | Was ±0.05 — tighter to reduce noise |

The model was chronically over-signaling. These thresholds were raised after measured performance degradation in the bankroll simulator. Do not revert to old values without walk-forward validation evidence showing improvement.

---

## Arb Polling Architecture

APScheduler runs a `_run_arb_poll()` job every 5 minutes inside FastAPI:
1. Marks all existing `PredictionMarketArb` records inactive
2. Fetches latest `OddsSnapshot` records from DB
3. Calls `KalshiClient.get_sports_markets()` — tries series-based fetch first (KXNBAGAME, KXMLBGAME, KXNHLGAME, KXNFLGAME, KXNBAPTS, KXNBAREB, KXMMA, KXSOCCER), falls back to keyword scan
4. Runs fuzzy event matching (>=2 meaningful words in common between titles)
5. Calculates arb spread: `(kalshi_implied_prob - sb_implied_prob) * 100` using bid/ask midpoint
6. Stores new active records, sorted by spread descending

**Env vars required:** `KALSHI_KEY_ID` + `KALSHI_PRIVATE_KEY` (RSA-PSS signed requests).

---

## Best EV+ Lifecycle

1. **11 AM daily** — `save_daily_picks()` snapshots top picks to `BestEVPick` with `result='pending'` (launchd: `com.clvanalytics.save.dailypicks.plist`, runs after opening odds collection at 10 AM)
2. **3 AM daily** — `settle_picks()` matches `BestEVPick` records to `BettingOutcome` results (launchd: `com.clvanalytics.settle.picks.plist`, runs after score fetching at 2 AM)
3. **Bankroll sim** — defaults to `source=best_ev`, queries settled `BestEVPick` records only
4. **Frontend** — `BestEVOpportunities.tsx` fetches `today_only=true` picks; bankroll sim shows same data class

This 1:1 alignment between what you see in Best EV+ and what is simulated is **intentional and critical**. Do not break it.

### Settlement logic (all three market types now implemented)
- **h2h** — compares `outcome_name` to `BettingOutcome.winner` ("home"/"away"/"push")
- **spreads** — `cover_margin = team_margin + point_line`; positive = win, zero = push, negative = loss. `team_margin` is `point_differential` for home picks, `-point_differential` for away picks.
- **totals** — compares `BettingOutcome.total_points` to `point_line`; `outcome_name == "over"` determines direction.

---

## Performance Rules

### Backend
- No N+1 queries — prefer explicit JOINs when related data is always needed; use pre-fetched bulk `_map` dicts when fetching related objects after the main query; use `IN (...)` for bulk lookups
- When JOINing the same table twice (e.g. home + away team), use `sqlalchemy.orm.aliased` — see `get_game_score` in `main.py`
- In-memory TTL cache on `/api/stats` (5 min), `/api/bookmakers` (10 min), `/api/clv-history` (5 min)
- Composite indexes on all hot filter/sort paths — see `database.py`
- `SELECT COUNT(*)` not `len(query.all())`
- `statistics.median()` not manual sort

### Frontend
- **Tailwind is v4, configured CSS-first in `src/index.css`.** There is no
  `tailwind.config.js` — design tokens live in the `@theme` block. Do not reintroduce v3
  `@tailwind base/components/utilities` directives; v4 silently ignores them and emits a
  stylesheet containing nothing but the reset (this shipped broken for months).
- **All API calls go through `src/api/client.ts`** (`fetchJSON` / `postJSON`), which owns
  `API_BASE`, a 30s timeout and `ApiError`. No raw `fetch()` in components.
- **Poll with `usePolling`** (`src/hooks/usePolling.ts`) — it pauses while the tab is hidden.
- **Use `LoadingState` / `ErrorState` / `EmptyState`** from `components/States.tsx`; card chrome
  is `GlassCard` / `Panel`; chart styling is `src/charts/theme.ts`. Do not hand-roll these.
- Use semantic colour tokens (`text-pos`, `text-neg`, `bg-panel`, `border-line`) rather than
  raw `bg-white/5` / `border-white/10` literals.
- `React.memo` on all display components
- `useMemo` on sort and chart computations
- 500ms debounce on bankroll simulator inputs (useRef + setTimeout pattern)
- `Map` ref cache for expanded opportunity rows — no duplicate fetches
- Lazy tab loading in Dashboard — overview data fetched on mount; Games tab fetched on first visit via `loadedTabs` ref
- `sport_key` propagated from API through `GameWithCLV` and `EVOpportunity` — used for sport badge display and per-sport filtering in Games and BestEV tabs
- `BestEVOpportunities` deduplicates on `(game_id, market_type, outcome_name)` keeping highest EV score — multiple bookmaker snapshots produce duplicates that must be collapsed before display

---

## Coding Standards

- **No backwards-compatibility hacks** — if something is unused, delete it
- **No speculative abstractions** — build for the current task, not hypothetical future ones
- **No docstrings on unchanged functions** — only comment where logic is non-obvious
- **Validate only at system boundaries** — user input, external APIs; trust internal framework
- **Always use `poetry run` prefix** for Python commands (alembic, pytest, scripts)
- **TypeScript strict** — fix type errors in files you touch; do not introduce new `any` casts
- **No emojis** in UI unless explicitly requested

---

## Running the System

```bash
# Start database
docker-compose up -d

# Apply migrations
poetry run alembic upgrade head

# Train / retrain ML model
poetry run python -m scripts.train_movement_model --walk-forward

# Start backend + frontend
./start.sh
# Backend: http://localhost:8000  |  Frontend: http://localhost:5173
```

---

## Environment Variables

```
ODDS_API_KEY=<required>         # The Odds API — odds ingestion
KALSHI_KEY_ID=<required>        # Kalshi RSA key ID
KALSHI_PRIVATE_KEY=<required>   # Kalshi RSA private key PEM
DATABASE_URL=<postgres dsn>     # PostgreSQL connection string
```

Polymarket and Metaculus are public APIs — no keys needed.

---

## ML Pipeline Fixes Applied (2026-04)

These were bugs — do not revert.

### 1. Live prediction features were zeroed out
`get_best_opportunities`, `get_upcoming_opportunities`, and `save_daily_picks` were sending hardcoded zeros for all temporal and bookmaker features at prediction time. The model trained on real calculated values but predicted with blank inputs. Fixed by calling `engineer.calculate_temporal_features()` and `engineer.calculate_bookmaker_features()` before every prediction, matching what `prepare_training_data` does during training.

### 2. Train/test split was random — future leaked into training
`train_test_split_data` in `features.py` was using sklearn's random shuffle. Odds snapshots from later dates could end up in the training set while earlier ones were tested. Fixed by:
- Adding `snapshot_timestamp` to every row in `prepare_training_data`
- Sorting by `snapshot_timestamp` and cutting at the 80th percentile position chronologically
- The split date is logged on every retrain so you can verify it

### 3. Spread and total picks never settled
`settle_picks` only handled h2h. Spreads and totals stayed `pending` forever, making bankroll simulator P&L unreliable. Fixed by:
- Adding `point_line` column to `BestEVPick` (migration `ae018c662ca2`)
- Saving `opening_point` into `point_line` when daily picks are created
- Implementing spread settlement via `cover_margin = team_margin + point_line`
- Implementing totals settlement via `total_points` vs `point_line`

### 4. Confidence scores were uncalibrated
XGBoost/RF `predict_proba` outputs are not inherently well-calibrated — "62% confidence" did not mean 62% of those picks won. Fixed by wrapping both classifiers in `CalibratedClassifierCV(cv=5, method='isotonic')` in `movement_predictor.py`. The calibrated versions (`xgb_calibrated`, `rf_calibrated`) are used in `predict_movement()`; the base classifiers remain for `evaluate_classification()` comparison. Both calibrated objects are saved/loaded with the model.

### 5. STAY class was chronically under-predicted (class imbalance)
Training data has ~20% STAY vs ~40% each for UP/DOWN. The classifier learned to rarely predict STAY, inflating apparent accuracy while missing real "no movement" signals. Fixed by:
- `RandomForestClassifier`: added `class_weight='balanced'` at init
- `XGBClassifier`: passes `sample_weight=compute_sample_weight('balanced', y)` to `fit()`
- Result: STAY accuracy improved from 32.6% → 53.5%; overall accuracy 59.2% → 57.5% (expected tradeoff — precision went up from 58.96% → 59.40%)

### Current validated model performance

**Models are per-sport since 2026-07-25** — `models/line_movement_predictor_{sport_key}.pkl`.
Compare a retrain only against the same sport's own history. Never against a mixed-sport figure.

| Scope | Date | Walk-forward accuracy | Naive baseline |
|---|---|---|---|
| **baseball_mlb** (current) | 2026-09-15 | **48.3% ± 2.0%** (18,802 rows) | 35.6% |
| baseball_mlb | 2026-07-25 cut | 48.8% ± 5.2% (8,656 rows) | — |
| Mixed-sport, pre-split (historical) | 2026-04-21 | 56.6% ± 0.7% | 40.1% |

The 2026-04-21 row is **largely NBA and predates the per-sport split**. It is not a valid
comparator for an MLB retrain — a 2026-09 MLB retrain measuring 48.3% was briefly mistaken for an
8-point regression against it. MLB accuracy has been flat since July; what improved with 2.2x the
data is stability (±5.2% → ±2.0%).

- Retrain command: `poetry run python -m scripts.train_movement_model --sport <key> --walk-forward`
- **`train_sport()` saves unconditionally** — it overwrites the live model before you see the
  validation number. Back up the existing pkl before retraining, and there is deliberately no
  scheduled retrain job until a promotion gate exists. See `ROADMAP_2026_09.md` Phase 5.

---

## Multi-Sport Data Collection (2026-04)

### Sports currently collecting
| Sport | Key | Score Source |
|-------|-----|-------------|
| NBA | `basketball_nba` | **ESPN public API** (free, no auth) — replaced cdn.nba.com 2026-09-15 |
| MLB | `baseball_mlb` | MLB Stats API — `statsapi.mlb.com` (free, no auth) |

NFL (`americanfootball_nfl`) to be added when preseason begins (August).

### How it works
- `scripts/collect_odds.py` loops through `SPORTS = [("basketball_nba", "NBA"), ("baseball_mlb", "MLB")]` on every run — add new sports here
- `scripts/fetch_game_scores.py` routes by sport key: NBA → `NBAScoresClient`, MLB → `MLBScoresClient`
- `--backfill` recovers every past game still missing a score, regardless of age. The nightly
  launchd job runs `--days 3` so one missed night self-heals.

### Score-fetching gotchas (learned the hard way, 2026-09-15)
- **Do not set a browser User-Agent on the ESPN client.** ESPN 403s `Mozilla/5.0` and custom agents
  on that endpoint and allows honest tool agents. The client deliberately sets none, so requests'
  own default gets through.
- **ESPN team names are emitted as nicknames** (`Clippers`, not `LA Clippers`) because the matcher
  compares by substring against full DB names and ESPN abbreviates some locations.
- **Score caches span +/-1 day**, because the UTC date of `commence_time` does not always equal the
  source's official game date. This means the same matchup can appear several times during a
  series — `_rank_candidates()` picks the nearest by date and **refuses to match when two
  equidistant candidates disagree**. Never relax that: writing another night's score into
  `betting_outcomes` silently corrupts settlement and P&L.
- `OddsDataProcessor` is sport-agnostic — creates teams, games, and snapshots for any `sport_key` automatically
- The ML model is sport-agnostic — it uses market structure features, not sport-specific ones. Retrain after ~3 weeks of MLB data to incorporate it.

### Adding a new sport
1. Add `("sport_key", "NAME")` to `SPORTS` in `collect_odds.py`
2. Add a scores client or route it in `fetch_game_scores.py`
3. No DB migrations, no model changes needed

---

## Prediction Market Paper Trading (2026-04)

A cross-platform signal system that compares Kalshi prices to Polymarket and Metaculus forecasts, opens paper trades when divergence exceeds a threshold, and settles them when Kalshi markets resolve.

### New DB Models
| Model | Table | Purpose |
|-------|-------|---------|
| `KalshiMarketPrice` | `kalshi_market_prices` | Time-series price snapshots per market ticker |
| `CrossPlatformSignal` | `cross_platform_signals` | Divergence events — Kalshi vs Polymarket/Metaculus |
| `PaperTrade` | `paper_trades` | Simulated trades generated by signal engine |

### New Files
| Path | Purpose |
|------|---------|
| `src/collectors/metaculus_client.py` | Metaculus public API — `get_forecast(keyword)` returns community_prediction median |
| `src/analyzers/pm_signal_generator.py` | `PMSignalGenerator` — fair_value(), generate_signal(), kelly_size() |

`polymarket_client.py` extended:
- `get_market_price(keyword)` and `get_active_markets(limit)` — existing direct-lookup methods unchanged
- `get_markets_by_tag(tag, max_pages, limit)` — paginated fetch filtered by tag; keyword-filters only when `tag=="sports"`
- `get_all_markets_cached()` — fetches sports/politics/crypto tags in sequence with 1s sleep between each; used by `PMSignalGenerator.refresh_poly_cache()`
- `get_sports_markets(**kwargs)` — thin wrapper around `get_markets_by_tag("sports")`

`kalshi_client.py` extended:
- `NON_SPORTS_SERIES = ["KXBTC", "KXETH", "KXPRES", "KXFED", "KXHOUSE", "KXSENATE", "KXECON"]`
- `get_all_markets(series_list=None)` — iterates all series (SPORTS + CHAMPIONSHIP + NON_SPORTS by default) with 0.5s sleep between each; used by `_run_pm_price_collection`
- `get_sports_markets()` — unchanged; still used by `_run_arb_poll` (sports-only)

### Signal Logic
- `fair_value`: weighted avg of Polymarket (0.45), Metaculus (0.35), Kalshi momentum (0.20). Missing sources have weight redistributed proportionally.
- Edge threshold: 0.06 (6 percentage points)
- Position size: Quarter-Kelly on $1000 bankroll, clamped $25–$200
- Strategy tag: `cross_platform_{category}_v1` where category is inferred from ticker prefix via `_infer_category()`:
  - `KXBTC`, `KXETH` → `crypto`
  - `KXPRES`, `KXFED`, `KXHOUSE`, `KXSENATE` → `politics`
  - `KXECON` → `economics`
  - everything else → `sports`
- `SIMILARITY_THRESHOLD = 0.45` — minimum word-overlap score to accept a Polymarket match

### APScheduler Jobs Added
- `poly_cache_refresh` — every 30 min: calls `_refresh_poly_cache()` which instantiates `PMSignalGenerator` and refreshes the Polymarket cache via `get_all_markets_cached()`
- `pm_price_collection` — every 10 min: fetches all Kalshi markets via `get_all_markets()`, reads from cached Polymarket data (falls back to inline refresh if cache is empty on first run), stores `KalshiMarketPrice`, generates `CrossPlatformSignal`, opens `PaperTrade` (skips if open position already exists for that ticker)
- `settle_paper_trades` — daily 3:15 AM: queries Kalshi for `status == "finalized"`, settles WIN/LOSS via `result == "yes"/"no"`, computes pnl

### Settlement P&L Formula
- WIN: `pnl = (1.0 - entry_price) * (size_usd / entry_price)`
- LOSS: `pnl = -size_usd`

### New API Endpoints
- `GET /api/paper-trades` — list trades, params: is_open, strategy_tag, limit
- `GET /api/paper-trades/stats` — aggregate stats (win_rate, total_pnl, pnl_by_strategy_tag, trades_by_month)
- `GET /api/cross-platform-signals` — recent signals, params: limit, min_divergence

### Frontend
- `frontend/src/components/PredictionMarkets.tsx` — stats bar + open positions table + trade history table + recent signals table; 60s auto-refresh
- `Dashboard.tsx` — "Pred Markets" tab added (view: `pred-markets`)

### No new env vars required — Polymarket and Metaculus are public APIs.

---

## Decision-Making Standard

Every architectural choice, threshold change, and data pipeline modification has direct P&L consequences. When proposing changes:
1. State the problem being solved with evidence (simulator output, accuracy metric, etc.)
2. Propose the minimal change that fixes it
3. Do not introduce complexity that is not immediately necessary
4. If uncertain, ask — do not guess on production parameters

Wrong decisions have real financial consequences.

---

## Prediction Market System — Hardening Roadmap (2026-06)

**STATUS: ALL 5 PHASES COMPLETED 2026-06-20.** The roadmap below is preserved as historical context for the rationale behind each fix. Do not re-implement; do not revert.

### Implementation summary (what was actually shipped)

| Phase | File(s) | Change |
|---|---|---|
| 1 (BUG-1) | `src/api/main.py` — `_settle_paper_trades` (~line 2086) + scheduler (~line 2191) | Switched from `cron(hour=3,minute=15)` → `interval(hours=1)`; added 0.3s `time.sleep` between each `kalshi.get_market()`; per-trade try/except already existed; added `Paper trade settlement starting: N open trades to check` log line at batch start |
| 2 (BUG-2) | `src/analyzers/pm_signal_generator.py` — `_match_polymarket`, `generate_signal` | `_match_polymarket` signature changed from `Optional[float]` → `tuple[Optional[float], float]` (returns `(price, similarity_score)`); `generate_signal` rejects the Polymarket match (sets `poly_price=None`) when `abs(yes_price − poly_price) > 0.50 and poly_sim < 0.70`. `SIMILARITY_THRESHOLD = 0.45` is unchanged. Verified: KXMLB-26-TEX synthetic case now returns `None` |
| 3 (BUG-3) | `src/analyzers/pm_signal_generator.py` — `kelly_size`, `generate_signal` | `kelly_size`: when `price > 0.80`, `max_size` is locally tightened to `$50` (overrides `MAX_SIZE_USD = 200`); `generate_signal`: returns `None` when `entry_price > 0.80 and edge < 0.10` (applied after side selection). All three CLAUDE.md synthetic cases verified |
| 4 (BUG-4) | `src/collectors/kalshi_client.py` | Moved `KXNBAGAME`, `KXNBAPTS`, `KXNBAREB` out of `SPORTS_SERIES` into new constant `NBA_GAME_SERIES` (comment: `# re-add to SPORTS_SERIES when NBA season resumes (~October)`). `_GAME_LEVEL_SERIES` set is unchanged so `is_game_level()` still works when they are re-added. `KXNBA` in `CHAMPIONSHIP_SERIES` is unchanged |
| 5 (BUG-5) | `src/api/main.py` — `_run_pm_price_collection` | Before opening a `PaperTrade`, check `if volume < 500: continue`. `CrossPlatformSignal` and `KalshiMarketPrice` are still stored above this guard. Threshold 500 is a starting point — revisit after observing trade volume reduction |

### Re-enabling NBA when the season resumes (~October 2026)

In `src/collectors/kalshi_client.py`, move the three tickers from `NBA_GAME_SERIES` back into `SPORTS_SERIES`. No other changes required.

### Original bug context (preserved)

The following bugs were identified and fixed in order. Each phase had explicit validation criteria — the next phase was not started until the current one passed.

### Known Bugs (resolved — see implementation summary above)

**BUG-1: Settlement rate-limited into silence**
`_settle_paper_trades()` in `main.py` runs once at 3:15 AM and calls `kalshi.get_market(ticker)` individually for every open trade. With 359 open trades this fires 359 sequential Kalshi API calls in one shot, hits rate limits, and silently drops the rest. Result: 359/383 trades permanently stuck open, P&L unreliable.

**BUG-2: Bad Polymarket match on KXMLB championship markets**
`_match_polymarket()` in `pm_signal_generator.py` matched `KXMLB-26-TEX` (Kalshi Rangers World Series at 2.3%) against Polymarket "Will Texas win the 2026 Pro Baseball Championship?" at 99.95% — almost certainly Texas Longhorns college baseball, not MLB. The similarity threshold (0.45) is too loose for championship futures where team names appear in unrelated markets. This fires every 10 minutes, polluting `cross_platform_signals` with a fake 97.6% divergence and generating bad paper trades.

**BUG-3: Kelly sizing is dangerous on high-priced NO positions**
`kelly_size()` in `pm_signal_generator.py` uses `(1.0 - price)` as the denominator. At entry_price=0.91 with edge=0.09, this produces MAX_SIZE_USD ($200) while the actual win payout is only ~$16. The 6% edge threshold at high prices (>0.80) produces poor risk/reward that Kelly math alone doesn't protect against. Crypto strategy is -$177 on 22 trades, largely from this pattern.

**BUG-4: Off-season NBA series still fetched every 10 minutes**
`_run_pm_price_collection()` calls `kalshi.get_all_markets(series_list=SPORTS_SERIES + CHAMPIONSHIP_SERIES + NON_SPORTS_SERIES)`. `SPORTS_SERIES` includes `KXNBAGAME`, `KXNBAPTS`, `KXNBAREB`. NBA season is over — these return empty but consume Kalshi API quota and add latency to every 10-minute run. `KXNBA` (championship futures) in `CHAMPIONSHIP_SERIES` may still have active markets (preseason futures) and should stay.

**BUG-5: Min-volume guard missing on paper trade entry**
`_run_pm_price_collection()` opens paper trades on any market with edge > threshold regardless of volume. Zero-volume Kalshi markets have wide bid/ask spreads that make the implied probability unreliable. Signals on illiquid markets are noise.

---

### Phase 1 — Fix Settlement (BUG-1)
**Files:** `src/api/main.py` (`_settle_paper_trades`)

**Changes:**
- Change APScheduler job from `cron(hour=3, minute=15)` to `interval(hours=1)` so settlement runs continuously
- Add a 0.3s sleep between each `kalshi.get_market()` call to stay under rate limits
- Add per-trade try/except so a single failed API call doesn't abort the whole batch
- Log total open trades at start and settled count at end for observability

**Do not change:** settlement logic, P&L formula, or any other scheduler job.

**Validation:**
- Restart the API (`./start.sh`)
- Wait up to 1 hour and confirm the log shows `Paper trade settlement: N settled` with N > 0
- Run `GET /api/paper-trades/stats` — `open_trades` count should decrease over subsequent hours
- No other endpoints should be affected

---

### Phase 2 — Fix Bad Polymarket Match (BUG-2)
**Files:** `src/analyzers/pm_signal_generator.py` (`generate_signal`, `_match_polymarket`)

**Changes:**
- In `generate_signal()`, after computing `fv_yes` for non-crypto markets: if `poly_price is not None` and `abs(kalshi_price - poly_price) > 0.50`, require that the similarity score used to accept the match was ≥ 0.70 (not just 0.45). Store the similarity score through from `_match_polymarket()` so `generate_signal()` can inspect it.
- Modify `_match_polymarket()` to return a tuple `(price, similarity_score)` instead of just `price`. Update all callers.
- The cache miss → search API fallback path should also return its similarity score.

**Do not change:** SIMILARITY_THRESHOLD constant (still used for initial acceptance), crypto path, Metaculus path, or Kelly sizing.

**Validation:**
- The `KXMLB-26-TEX` signal (Kalshi 0.023, Polymarket 0.9995, divergence 0.9765) must no longer appear in `GET /api/cross-platform-signals` after the next `_run_pm_price_collection` cycle (~10 min after restart)
- Legitimate signals with <50% price gap should still pass through
- Check logs for `Signal generation failed` — should not increase

---

### Phase 3 — Fix Kelly Sizing on High-Priced Markets (BUG-3)
**Files:** `src/analyzers/pm_signal_generator.py` (`kelly_size`, `generate_signal`)

**Changes:**
- In `kelly_size()`, add a guard: if `price > 0.80`, cap `MAX_SIZE_USD` at `50.0` for that call. The risk/reward at high prices makes large positions unjustifiable even with edge.
- In `generate_signal()`, after selecting side: if `entry_price > 0.80 and edge < 0.10`, return `None` — a 6% edge on a 90-cent contract is not worth taking. Raise the edge bar at high prices.

**Do not change:** EDGE_THRESHOLD (still used for low/mid-priced markets), BANKROLL, KELLY_FRACTION, MIN_SIZE_USD, or the crypto pricing model.

**Validation:**
- Manually call `generate_signal()` with a synthetic market: `yes_price=0.91, no_price=0.09, edge=0.07` — should return `None`
- Synthetic market: `yes_price=0.91, no_price=0.09, edge=0.12` — should return a signal with `size_usd <= 50.0`
- Synthetic market: `yes_price=0.55, no_price=0.45, edge=0.08` — should still return a normal-sized signal (unaffected)
- After next collection cycle, confirm no new paper trades open with `size_usd > 50` where `entry_price > 0.80`

---

### Phase 4 — Remove Off-Season NBA from Kalshi Fetch (BUG-4)
**Files:** `src/collectors/kalshi_client.py`, `src/api/main.py`

**Changes:**
- In `kalshi_client.py`, move `KXNBAGAME`, `KXNBAPTS`, `KXNBAREB` out of `SPORTS_SERIES` into a new constant `NBA_GAME_SERIES`. Add a comment: `# re-add to SPORTS_SERIES when NBA season resumes (~October)`.
- In `main.py` `_run_pm_price_collection()`, the `series_list` argument already uses `SPORTS_SERIES` — no change needed there once the constant is updated.
- `KXNBA` stays in `CHAMPIONSHIP_SERIES` (futures markets may still be active).

**Do not change:** `get_sports_markets()` used by arb poll — it already does its own series-based fetch and is separate.

**Validation:**
- After restart, check API logs: next `_run_pm_price_collection` run should fetch fewer markets (no NBA game-level series)
- Arb poll (`GET /api/arb-opportunities`) must still function — confirm it still returns results or empty list without error
- `GET /api/paper-trades` should continue to work

---

### Phase 5 — Add Min-Volume Guard on Paper Trade Entry (BUG-5)
**Files:** `src/api/main.py` (`_run_pm_price_collection`)

**Changes:**
- In `_run_pm_price_collection()`, before opening a new `PaperTrade`, check `volume >= 500`. If volume is `None` or below 500, log at DEBUG level and skip opening a trade (still store the price snapshot and signal record).
- The 500 threshold is a starting point — it filters clearly illiquid markets without being too restrictive.

**Do not change:** signal generation logic, `CrossPlatformSignal` storage (signals should still be recorded regardless of volume), or price snapshot storage.

**Validation:**
- After restart, new paper trades opened should all have corresponding `KalshiMarketPrice.volume >= 500`
- `GET /api/cross-platform-signals` should still return signals for low-volume markets (signals recorded, just no trade opened)
- Total new trades per day should decrease — that's expected and correct
