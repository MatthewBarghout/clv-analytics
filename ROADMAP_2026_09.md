# CLV Analytics — Hardening & Overhaul Roadmap (2026-09)

**Status: ALL PHASES COMPLETE (2026-09-17).** Created 2026-09-15 after a full
system audit.

Successor to the 2026-06 Prediction Market Hardening Roadmap (all 5 phases shipped 2026-06-20,
preserved in `CLAUDE.md`). Bug numbering continues from BUG-5.

Same working rule as the last roadmap: **phases are done in order, and the next phase does not
start until the current one passes its validation criteria.**

---

## Audit baseline (2026-09-15)

Every number below was measured, not estimated. They are the before-state for validation.

| Metric | Value |
|---|---|
| Best EV+ settled record | 345W / 361L / 20 push — all-time **−$1,845** (flat $100 units) |
| Best EV+ recent trend | Jul −$3,310 → Aug +$1,122 → Sep +$1,056 (Aug+Sep: **+$2,178** on 165 picks) |
| Paper trades | 824 total, +$6,206, 74% win rate |
| Paper P&L by strategy | crypto **+$8,278** (713) / politics **−$2,017** (22) / sports −$55 (6) |
| Picks stuck `pending` | **34** (excluding today's 8 legitimately open) |
| Past games with `completed = false` | **232** |
| Paper trades open since Apr/May | **58** (16 on the retired `cross_platform_v1` tag) |
| `/api/ml/best-opportunities` latency | **23.1s** measured |
| MLB model last trained | 2026-07-25 (**7 weeks stale**) |
| Scheduled retrain job | **none exists** |
| Odds API quota remaining | key1: **2**, key2: 113, key3: 500 |

**Confirmed still holding from the 2026-06 roadmap:** BUG-3's high-price cap. All 175 post-fix
trades at `entry_price > 0.80` have `size_usd` of exactly $50.00, netting +$384. Do not touch it.

---

## Operational finding — the host sleeps, and the in-process scheduler stops with it

**This is the most consequential operational finding of the session, and it is not a code bug.**

While validating the hourly settlement job, scheduled jobs stopped firing entirely: the arb poll
(5 min) ran at 16:08:06 and 17:08:06 and not once in between, despite a live scheduler. `pmset -g log`
explains it:

```
18:07:21  Sleep     Entering Sleep state due to 'Maintenance Sleep' ... Using Batt (Charge:61%)
18:07:32  DarkWake  DarkWake from Deep Idle ...
18:07:51  Sleep     Entering Sleep state due to 'Maintenance Sleep' ...
```

On battery the machine cycles into maintenance sleep every ~30s. APScheduler lives inside the
FastAPI process, so it is suspended too; on wake it emits `Run time of job ... was missed by
0:12:41` and skips the interval. Every one of those "missed by" warnings is a wake-up, not a
blocked event loop.

**Consequences:** `_settle_paper_trades` (hourly), `_run_pm_price_collection` (10 min),
`_run_arb_poll` (5 min) and `_refresh_poly_cache` (30 min) only run while the laptop is awake. This
is very likely a contributing cause of BUG-1's original symptom — a 3:15 AM settlement job on a
sleeping laptop never runs at all, and moving it to hourly does not fix that.

**This is not addressed by any phase in this roadmap.** Options, in rough order of robustness:
- Move the scheduler out of the API process into launchd jobs (launchd runs missed calendar jobs on
  wake; APScheduler does not).
- Run the machine on AC power with `caffeinate -s` while the system is expected to collect.
- Accept it and treat collection as best-effort while the laptop is awake.

Worth deciding deliberately, because the paper-trading and arb records are only as complete as the
hours the machine was awake.

---

## Operational finding — `--reload` resets every scheduler timer

`start.sh` runs uvicorn with `--reload`, and APScheduler is started at app import. **Every code
edit restarts the scheduler and resets all interval timers from zero.** During the Phase 1-5 work
the app reloaded 8+ times, so:

- `arb_poll` (5 min) fires normally — short enough to survive between edits
- `pm_price_collection` (10 min) mostly survives
- `settle_paper_trades` (**1 hour**) effectively **never fires during an active development
  session** — a full uninterrupted hour is required

**Note:** the host-sleep finding above turned out to be the larger cause of missed jobs. Both are
real; sleep matters even when nobody is editing code.

This is not a bug in the settlement logic, which BUG-1 already fixed and which was verified working
on demand. But it means paper trade settlement silently stalls whenever the project is being worked
on, which is precisely when someone would be watching for it. Worth considering: run the scheduler
in a separate process from the reload-enabled API, or disable `--reload` for normal operation.

---

## Deferred — not in this roadmap

**Odds API quota exhaustion.** Key 1 is at 2 requests remaining and returning 401s (the rotation
logic in `collect_odds.py` catches this and falls through to key 2). ~615 requests remain across
all three keys. New keys are being procured; until they land, **do not add sports** — NFL and the
October NBA re-enable both depend on quota headroom. Revisit as a separate phase once keys are in.

**Update 2026-09-15: BUG-13 is fixed, so the NBA prerequisite is cleared.** NBA scores now come
from ESPN and are verified working on historical dates. Re-enabling NBA odds in October is blocked
only on Odds API quota, not on score collection.

Consequence to accept in the meantime: no NFL sportsbook odds are being collected during NFL
season, while Kalshi collection is already pulling 192 `KXNFLGAME` markets every 10 minutes. The
prediction-market side has NFL data with no sportsbook side to compare against.

---

## Phase 1 — Capture API and scheduler logs (BUG-6) — **DONE 2026-09-15**

**Do this first.** It is small, and every later phase's validation depends on being able to read
what the scheduler actually did.

### Outcome
Shipped. `logs/api.log` is live and rotating. The proof point, from the first run after the fix:

```
Pick settlement starting: 42 pending, 0 with completed outcomes
Settled 0 Best EV picks
```

Four months of nightly runs had reported success while settling nothing. The gap is now one
grep away.

**Root cause was more specific than first diagnosed:** `start.sh` redirected uvicorn to
`/tmp/clv-backend.log`, and macOS purges `/tmp` files after ~3 days. The long-running process kept
writing to a deleted inode. Logs now go to `logs/` inside the project, which is not purged.

### The bug (original)
Uvicorn writes to a stdout nobody captures; `logs/api.log` is stale since January. Every
APScheduler job — settlement, PM collection, arb poll — is invisible.

It is worse than "no logs". `settle-picks` and `save-daily-picks` are launchd `curl` calls to
endpoints that return immediately and do the work in a FastAPI background task:

```
/tmp/clv-settle-picks.log:  {"status":"scheduled","message":"Pick settlement running in background"}
```

That acknowledgement is the *entire* log. The background task's output goes to the uncaptured
stdout. **This is why 34 picks sat stuck for four months without anyone noticing** — the job
reported success every single night while settling nothing.

### Changes
- `start.sh`: redirect uvicorn stdout/stderr to `logs/api.log` (append, not truncate).
- Add a `RotatingFileHandler` on the root logger in `src/api/main.py` — 10MB × 5 files, so it
  cannot grow unbounded the way `logs/launchd-opening.log` did (2.0MB and never rotated).
- `_settle_paper_trades`, `_run_pm_price_collection`, `_run_arb_poll`, `settle_picks`,
  `save_daily_picks`: log a one-line summary at start and completion, with counts.
- Add log rotation or a cleanup for the 1,741 files in `logs/`. **Shipped** as a 30-day prune of
  `collection_*.log` at the end of `scripts/run_collection.sh` — collection fires many times daily,
  so the directory self-maintains without a new scheduled job. Note:
  `launchd/cleanup_old_batches.sh` turned out to be a one-off plist migration script, unrelated to
  log files — it is correctly not scheduled.

**Do not change:** any job's schedule, logic, or the endpoints' background-task pattern.

### Validation
- Restart via `./start.sh`; confirm `logs/api.log` grows and has a current timestamp.
- Within one hour, confirm the hourly `Paper trade settlement starting: N open trades` line
  (added in the 2026-06 Phase 1) actually appears in the file.
- Confirm the 10-minute `_run_pm_price_collection` summary appears.
- Trigger `POST /api/ml/settle-picks` by hand and confirm the *outcome* is now visible.

---

## Phase 2 — Size cap on low-priced contracts (BUG-7) — **DONE 2026-09-15**

### Outcome
Shipped as a single symmetric guard in `kelly_size()`:
`max_size = 50.0 if (price > 0.80 or price < 0.20) else MAX_SIZE_USD`.

Validated across price bands — low tail capped at $50, the 0.20 boundary and mid-range untouched
at $200, BUG-3's high-tail cap intact.

**Retrospective check on all 77 settled low-price trades:** actual P&L was **-$1,744.50**; with the
$50 cap it would have been **-$836.62**, an improvement of **+$907.88**, with 55 of 77 trades
affected. Realized win rate was 9W/68L = 11.7% against an 11.4% breakeven — confirming the read
that this bucket's problem is sizing, not edge. The edge filter was correctly not added.

### The bug (original)
BUG-3 fixed Kelly sizing at the top of the price range. The same structural flaw is unaddressed at
the bottom. `kelly_size()` uses `(1.0 - price)` as the denominator, so at a 10¢ price the
denominator is ~0.9 and Kelly runs straight to the `MAX_SIZE_USD` clamp.

Post-fix trades (after 2026-06-21) at `entry_price < 0.20`:

| Side | Trades | Avg price | Avg size | P&L |
|---|---|---|---|---|
| YES | 52 | 0.114 | $124.50 | −$255 |
| NO | 2 | 0.110 | **$200.00** | **−$400** |

**Be honest about what this evidence does and does not show.** The 52 YES trades realized ~11.0%
wins against a breakeven of 11.4% — that book is roughly fair, not obviously broken, and 52 trades
at 11% base rate is far too small a sample to conclude otherwise. The defect is **sizing, not
edge**: two trades at the $200 clamp account for 61% of the bucket's loss. Putting $200 on an 11¢
contract is a variance profile that Kelly math alone does not protect against, exactly as at the
high end.

### Changes
- In `kelly_size()`: if `price < 0.20`, locally tighten `max_size` to `$50.0` for that call —
  mirroring the existing `price > 0.80` rule. One symmetric guard, same constant.

**Do not change:** `EDGE_THRESHOLD`, `BANKROLL`, `KELLY_FRACTION`, `MIN_SIZE_USD`, the `> 0.80`
rule, or the crypto pricing model.

**Explicitly do NOT add** a `entry_price < 0.20 and edge < 0.10 → return None` filter by analogy
with BUG-3. There is no evidence for it — the low-price book is near breakeven, and an edge filter
would cut the bucket entirely on a 54-trade sample. Size first; revisit only with more data.

### Validation
- Synthetic `generate_signal()` with `yes_price=0.11, no_price=0.89, edge=0.12` → signal returned
  with `size_usd <= 50.0`.
- Synthetic `yes_price=0.55, edge=0.08` → normal size, unaffected.
- Synthetic `yes_price=0.91, edge=0.12` → still `size_usd <= 50.0` (BUG-3 path intact).
- After one week: no new trade has `entry_price < 0.20 AND size_usd > 50`.

---

## Phase 3 — Backfill missing scores, unstick settlement (BUG-8) — **DONE 2026-09-15**

### Outcome
Shipped `--backfill` on `fetch_game_scores.py` (walks the whole backlog, fetching one score cache
per distinct game-date rather than per day in the span — the backlog is sparse). Nightly launchd
job widened from `--days 1` to `--days 3` and its log moved out of `/tmp`.

Results:

| Metric | Before | After |
|---|---|---|
| Games missing outcomes | 232 | 171 at the time; **9 after Phases 3b/3c** |
| Picks stuck `pending` | 42 | **16** (8 are today's games, legitimately open) |
| Picks settled | 726 | 730 |
| **All-time Best EV+ P&L** | **-$1,845** | **-$1,366** |

August restated from +$1,122 to **+$1,777**; June from +$209 to +$33. **-$1,366 is now the honest
all-time number.**

The backfill surfaced two further bugs that block the remaining 171 games — logged as BUG-13 and
BUG-14 below. Neither existed in the original audit; both were only visible once the backfill ran.

### The bug (original)
`fetch_game_scores.py` runs nightly with `--days 1`. It only ever looks at yesterday. Any game
missed — API hiccup, machine asleep, MLB Stats API gap — is **never retried**. The result is 232
past games still `completed = false`, and because `settle_picks` matches against `BettingOutcome`,
every pick on those games is stuck `pending` forever and the bankroll simulator silently
under-counts.

The correlation is exact:

| Unscored game cluster | Stuck picks |
|---|---|
| 2026-06-25 | 12 |
| 2026-08-12 (26 games unscored) | 14 |
| 2026-05-05, 07-21, 08-10 | 8 |

Backlog by month: Dec 8, Jan 95, Feb 17, Mar 9, Apr 44, May 3, Jun 25, Jul 5, Aug 26. Nothing is
unscored after 2026-08-13 — the nightly job has been clean since, so this is a **recovery problem,
not an ongoing collection problem**.

### Changes
- Add a `--backfill` mode to `fetch_game_scores.py` that selects games where
  `commence_time < now() - 24h AND completed = false` (regardless of age) rather than a fixed
  day window, and fetches scores for them in batches with rate-limiting between calls.
- Run it once to clear the 232-game backlog.
- Then re-run `POST /api/ml/settle-picks` to settle the 34 freed picks.
- Make the nightly job resilient: widen it from `--days 1` to `--days 3` so a single missed night
  self-heals instead of becoming permanent.

**Do not change:** settlement logic for h2h/spreads/totals, the P&L formula, or the `BettingOutcome`
schema.

### Validation
- `select count(*) from games where commence_time < now() - interval '1 day' and not completed`
  drops from 232 toward 0. Some old NBA games may be unrecoverable if the API no longer serves
  them — record the final floor and the reason.
- Stuck `pending` picks drop from 34 toward 0.
- `GET /api/bankroll-simulation` settled-pick count rises by roughly the number settled.
- Re-measure all-time Best EV+ P&L afterward — **the −$1,845 baseline will move**, and the new
  number is the honest one. Expect the July figure to shift most.

---

## Phase 3a — Blocking `async def` handlers starve the scheduler (BUG-15) — **DONE 2026-09-15**

Found while waiting for Phase 1's settlement validation. This is the mechanism that links the
"slow dashboard" complaint to actual production harm, and it was not in the original audit.

### The bug
Every route handler in `main.py` and `ml_endpoints.py` was declared `async def` — **36 of them —
and not one `await` appears anywhere in either file.** All of them do fully synchronous work: sync
SQLAlchemy queries, pandas, XGBoost inference.

FastAPI runs an `async def` handler *directly on the event loop thread*. A handler that blocks for
25-112 seconds therefore blocks the entire event loop, and APScheduler's dispatcher along with it.

The log had **24 `Run time of job ... was missed by` warnings**:

```
_settle_paper_trades       missed by 0:12:41
_refresh_poly_cache        missed by 0:12:59
_run_pm_price_collection   missed by 0:02:41
_run_save_daily_picks      missed by 0:05:47
```

So: **opening the Best EV+ tab stalled arb polling, PM collection, and paper-trade settlement.**
This is also why Phase 1's hourly-settlement validation kept failing to fire — it was not only the
`--reload` timer reset, the job was being actively starved.

### Change
Dropped `async` from all 36 handlers, so FastAPI dispatches them to its threadpool and leaves the
event loop free. No other change — no logic, no signatures, no `BackgroundTasks` usage (which
works identically in sync handlers).

### Validation — passed
At 17:08:06, immediately after the fix, `_run_arb_poll` **ran and completed successfully while
`prepare_training_data` was executing concurrently** — precisely the combination that produced a
"missed by" warning three minutes earlier at 17:05:47. All 9 smoke-tested endpoints return 200.

Last missed-job warning: 17:05:47, before the fix. None since.

### Note for Phase 6
This does **not** make the endpoints fast — `best-opportunities` is still ~25s warm, and a heavy
request still burns CPU and slows concurrent requests through the GIL. It only stops that work
from starving the scheduler. Phase 6 remains necessary.

---

## Phase 3b — NBA score collection is fully broken (BUG-13) — **DONE 2026-09-15**

Discovered by the Phase 3 backfill. Not in the original audit — the nightly `--days 1` window
masked it completely.

### Outcome
Rewritten against **ESPN's public scoreboard API** (`site.api.espn.com`) — free, no auth, and it
takes a real date parameter. `stats.nba.com` was unreachable, `data.nba.net` is dead, and
`balldontlie.io` now requires a key.

**A trap worth recording:** ESPN 403s browser-spoofing User-Agents on this endpoint and *allows*
honest tool agents. `Mozilla/5.0` → 403; `python-requests/2.31.0` → 200. The client therefore sets
**no** User-Agent override, with a comment saying not to "fix" it by adding one.

Two follow-on defects surfaced only once data started flowing:
1. **Team naming.** ESPN's `displayName` is "LA Clippers" where the DB has "Los Angeles Clippers",
   which defeats the substring matcher. The client now emits the **nickname** (`Clippers`), unique
   across all 30 teams and always a substring of the DB name. That alone recovered 40 more games.
2. **Date bucketing.** Callers bucket by the UTC date of `commence_time`, which does not match the
   source's game day — evening tip-offs roll to the next UTC date. Both clients now span +/- 1 day.

Recovery: **232 unscored games → 9.**

| Stage | Unscored remaining |
|---|---|
| Start of session | 232 |
| After MLB backfill (Phase 3) | 171 |
| After ESPN swap (403 fixed) | 54 |
| After nickname matching | 14 |
| After +/-1 day span + disambiguation | **9** |

### The bug (original)
`NBAScoresClient` has two independent defects:

1. **It always requests today's scoreboard, whatever date you ask for.**
   `get_scoreboard(date)` builds `url = f"{BASE_URL}/todaysScoreboard_00.json"` and ignores `date`
   entirely. Its own comment concedes this — *"we'll use today's endpoint and the caller should
   handle date filtering"* — and the caller does not filter. Historical NBA scores have therefore
   never been retrievable; the nightly job only ever worked for games still on today's board.
2. **The endpoint now returns 403.** `cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json`
   is blocked, with and without a browser User-Agent. So even today's scores fail.

Combined impact: **162 NBA games have no outcome and cannot get one**, and when the season resumes
in ~October, NBA score collection will silently produce nothing — no outcomes, no settled picks,
no training labels, and an NBA-blind bankroll simulator.

### Changes
- Replace the source. `stats.nba.com/stats/scoreboardv2?GameDate=MM/DD/YYYY` takes a real date
  parameter but requires a full browser header set. The free `balldontlie.io` API is a simpler
  alternative with genuine historical date queries. Evaluate both against a known past date before
  committing.
- Whatever the source, `get_scoreboard(date)` must actually honour `date`.
- Re-run `--backfill` afterward to recover the 162 games.

### Validation
- `get_scoreboard()` for a known past NBA date returns that date's games, not today's.
- Backfill recovers a material share of the 162; record the floor and the reason for any remainder.
- **Must be done before the October NBA re-enable.** Re-enabling collection without this produces
  odds with no outcomes — worse than not collecting, because it pollutes training data with
  permanently unlabelled rows.

---

## Phase 3c — MLB team-name matching gap (BUG-14) — **DONE 2026-09-15**

### Outcome — diagnosed as a date problem, not a name problem

The +/-1 day span from Phase 3b immediately exposed a hazard that the old single-day fetch had
hidden: **MLB teams play multi-game series on consecutive days**, so widening the window puts the
same matchup in the cache two or three times with *different scores*. The matcher took the first
hit, which would have written another night's score into `betting_outcomes` — silently corrupting
settlement and P&L. That is far worse than a missing score.

Fix: both clients now carry the source's official `game_date`, and `_rank_candidates()` orders
team-matching entries by distance from the target game's date. **When the two nearest candidates
are equidistant and disagree, nothing is returned** — it refuses to guess.

That refusal is doing real work. Of the 9 remaining games, 5 are MLB matchups where the source has
the same two teams one day either side of our recorded date but nothing on the date itself (e.g.
Game 753, Brewers @ Cardinals: candidates on 05-04 and 05-06, none on 05-05). Most likely
postponed or rescheduled games where the odds were collected against a date the game never
happened on. The remaining 4 NBA games are absent from ESPN entirely.

**9 unrecoverable games out of 232, each with a logged reason.** 15 picks stay `pending`, 8 of
which are today's live games.

### Also changed
The redundant inline team-matching block in `fetch_and_store_score` was removed (it now trusts
`_rank_candidates`) and the winner calculation was lifted out of the triplicated ternary.

### The bug (original)
9 MLB games sit unmatched although the MLB Stats API **does** return data for their dates — the
backfill summary distinguishes the two cases and puts these in the "source had data but could not
be matched by team name" bucket. So this is a name-normalization defect in `fetch_and_store_score`,
not a source problem.

Affected games are spread across 2026-05-05 through 2026-08-13 and include ordinary matchups
(Cardinals/Brewers, Mets/Cubs, Yankees/Pirates), which suggests something situational —
doubleheaders, a suspended game, or a relocated game — rather than a systematic alias failure.

### Changes
- Log the actual near-miss: for each unmatched game, print the game's team names alongside the
  candidate names in the cache for that date. Diagnose from that before changing any matching
  logic.
- Fix whatever it shows. Do not add a fuzzy-matching layer speculatively — 9 games out of ~1,370
  is a narrow, specific failure.

### Validation
- Unmatched MLB count goes to 0, or each remainder has a documented reason.
- No previously-matched game changes its outcome — re-run and diff `betting_outcomes`.

---

## Phase 4 — Retire stale and legacy paper trades (BUG-9) — **DONE 2026-09-15**

### Outcome — **decision reversed from this roadmap's original preference**

The roadmap preferred *backfilling* the `cross_platform_v1` tag via `_infer_category()`, on the
reasoning that those were real signals under the rules of the time. Inspecting the cohort showed
that reasoning was wrong, and all 16 were voided instead.

The cohort is not a mixed bag: it is **16 MLB World Series futures opened in a single batch on
2026-04-26**, 14 of them sub-7¢ YES lottery tickets, from exactly the market family BUG-2
identified as broken. Their payoff profile on a $1,000 bankroll:

| Ticker | Price | Size | Payout if WIN |
|---|---|---|---|
| KXMLB-26-COL | 0.0015 | $46.11 | **$30,692** |
| KXMLB-26-WSH | 0.0015 | $41.34 | **$27,519** |
| KXMLB-26-MIA | 0.0030 | $41.14 | $13,673 |
| KXMLB-26-KC | 0.0065 | $88.59 | $13,540 |
| ... | | | |
| **Total if all won** | | | **$108,952** |

`_infer_category("KXMLB-...")` returns `sports`, so backfilling would have injected these into the
only clean sports cohort (6 trades, -$55). If Colorado wins the World Series, the track record
books a 30x bankroll gain from one trade the pre-fix matcher generated. That is not a track record.

Shipped: all 16 set to `resolution_result='VOID'`, `is_open=false`, `pnl=0` (reversible — nothing
deleted), and `/api/paper-trades/stats` now excludes VOID from `settled` and reports
`voided_trades` separately.

| Metric | Before | After |
|---|---|---|
| Open trades | 83 | **68** |
| `cross_platform_v1` bucket | present | **gone** |
| Win rate / total P&L | 74.0% / $6,206.10 | **74.0% / $6,206.10** (unchanged, as required) |

The legitimate open MLB futures held by the *categorized* strategies were left alone; they resolve
on their own in November.

### The bug (original)
58 paper trades have been open since April/May. Most are **not** a bug: `KXMLB-26-*` championship
futures genuinely do not resolve until the World Series ends in November, and the hourly settlement
job correctly leaves them open.

Two things inside that set are real problems:
- 16 trades carry `cross_platform_v1`, a strategy tag no longer produced by `_infer_category()`.
  They are pre-categorization legacy and will never be attributed to a strategy in
  `pnl_by_strategy_tag`.
- `KXMLB-26-TEX` is still open at a 2.65¢ entry with $37.96 on it — **this is the exact BUG-2 bad
  Polymarket match** (Kalshi Rangers 2.3% vs Polymarket "Texas" 99.95%, almost certainly Texas
  Longhorns college baseball). Phase 2 of the 2026-06 roadmap stopped new ones from opening. It
  never cleaned up the position it had already opened.

### Changes
- Close `KXMLB-26-TEX` as a data-quality void: set `is_open = false`,
  `resolution_result = 'VOID'`, `pnl = 0`. It was opened on a match the system now correctly
  rejects; settling it as a real WIN/LOSS would launder a known-bad signal into the track record.
- Decide the `cross_platform_v1` cohort explicitly — either backfill the tag via `_infer_category()`
  on the ticker, or mark the cohort void. Backfilling is preferred; those were real signals under
  the rules of the time.
- Add a `VOID` branch to the stats aggregation so voids are excluded from win rate rather than
  counted as losses.

**Do not change:** the settlement job's schedule or its WIN/LOSS logic, and do not force-settle the
legitimate open MLB futures — they resolve on their own in November.

### Validation
- `GET /api/paper-trades/stats` shows no `cross_platform_v1` bucket.
- Win rate and `total_pnl` change only by the void exclusions; the crypto/politics/sports splits
  are otherwise unmoved.
- `KXMLB-26-TEX` no longer appears in open positions on the Pred Markets tab.

---

## Phase 5 — Retrain, and schedule retraining — **DONE 2026-09-15 (with a corrected baseline)**

### Outcome

**The validation criterion in this phase was wrong, and it nearly caused a bad rollback.**

This phase said to compare the retrain against "the 2026-04 baseline (56.6% +/- 0.7%)" and keep the
old model if it degraded. The retrain returned **48.34% +/- 2.00%**, which looks like an 8-point
collapse — and the model was rolled back on that basis.

That comparison is invalid. The 56.6% figure in `CLAUDE.md` is a **mixed-sport (largely NBA)
number from 2026-04-21**, measured before the per-sport split shipped on 2026-07-25. MLB alone has
never been near it.

The like-for-like test — same sport, same pipeline, same `n_splits`, data cut at the July model's
own training date:

| Data cut | Rows | Walk-forward accuracy |
|---|---|---|
| Through 2026-07-25 (what the July model saw) | 8,656 | 48.82% +/- **5.22%** |
| Full, through 2026-09-14 | 18,802 | 48.34% +/- **2.00%** |

Accuracy is flat (0.5 points apart, far inside the error bars) and **variance more than halved** on
2.2x the data. The new model is the better one, and it was promoted. Naive baseline on this data is
35.6%, so the model's lift is +12.7 points; class mix is a fairly balanced UP 35.6% / DOWN 35.4% /
STAY 29.0%.

**MLB's real baseline, for all future comparisons: 48.3% +/- 2.0% walk-forward, vs 35.6% naive.**
Do not compare a single-sport retrain against the 56.6% mixed-sport figure again.

### Also shipped
- `analyze_daily_clv.py` converted to per-sport model loading (lazy `_get_model(sport_key)` cache,
  skips games whose sport has no trained model) — it previously loaded one global model and would
  have silently stopped producing EV opportunities once the orphan was deleted.
- Deleted `models/closing_line_predictor.pkl`, `models/line_movement_predictor.pkl`, and
  `scripts/auto_retrain.py`. The script was unreferenced, never scheduled, pointed at the dead
  single-model path, and depended on `models/training_metrics.json` / `retrain_log.json` which
  have **never existed** — so it had never once run successfully.

### Not done — the weekly retrain job needs a gate first

This phase called for a launchd weekly retrain. **It was deliberately not added.** `train_sport()`
calls `predictor.save_model()` unconditionally, so a scheduled retrain overwrites the live model
whatever the validation says. Today's run is the proof: the model was saved before anyone looked at
the number.

Scheduling that weekly would recreate, for models, exactly the class of silent-failure bug this
roadmap exists to remove. **Prerequisite: `train_sport()` must compare the candidate's walk-forward
accuracy against the incumbent's and only save on non-regression, recording both numbers in the
pkl.** Models currently store no metrics at all (`xgb_regression`, `preprocessor`, `feature_names`,
... — no accuracy, no training date), which is why the like-for-like test above had to be
reconstructed by re-running history.

### The problem (original)
There is **no scheduled retrain job**. Models retrain only when run by hand. Current state:

| Artifact | Date | Status |
|---|---|---|
| `line_movement_predictor_baseball_mlb.pkl` | 2026-07-25 | **7 weeks stale**, in active use |
| `line_movement_predictor_basketball_nba.pkl` | 2026-07-25 | NBA data ends April; stale but idle |
| `line_movement_predictor.pkl` | 2026-05-29 | **Orphaned** — serving globs `*_{sport}.pkl`, never loads this |
| `closing_line_predictor.pkl` | 2026-01-04 | **Orphaned** — zero code references |

Worse, `scripts/auto_retrain.py` still hardcodes `MODEL_PATH = "models/line_movement_predictor.pkl"`
— the orphaned legacy path. If it were ever scheduled it would retrain a model nothing serves,
while the real per-sport models kept aging. `scripts/analyze_daily_clv.py:61` has the same stale
path.

~9,400 new MLB snapshots have accumulated since the last training run.

### Changes
- Retrain MLB with walk-forward validation:
  `poetry run python -m scripts.train_movement_model --sport baseball_mlb --walk-forward`
- Compare against the 2026-04 baseline (56.6% ± 0.7% avg accuracy vs 40.1% naive). **Only promote
  if walk-forward accuracy holds or improves.** If it degrades, keep the current model and
  investigate — a 7-week data shift that hurts accuracy is itself a finding.
- Point `auto_retrain.py` and `analyze_daily_clv.py` at the per-sport paths.
- Add a launchd job for weekly retraining (Sunday, after the 2 AM score fetch).
- Delete `closing_line_predictor.pkl` and `line_movement_predictor.pkl` — per the standing
  "no backwards-compatibility hacks" rule, unused means deleted.
- **Do the retrain after Phase 3**, not before. Phase 3 backfills 232 games' worth of outcomes;
  training before that means training on a knowingly incomplete label set.
- Hold NBA retraining until the season resumes and real data accumulates (~3 weeks in).

**Do not change:** `MIN_MOVEMENT` (0.025), `MIN_CONFIDENCE` (0.62), `min_ev_score` (2.0),
prediction caps (h2h ±0.08, spreads/totals ±0.04), calibration, or class balancing. Those were set
against measured degradation and are not in scope here.

### Validation
- Walk-forward output logged, with the chronological split date, and compared to 56.6% baseline.
- `GET /api/ml/model-stats?sport_key=baseball_mlb` reflects the new training date.
- Best EV+ picks still generate the next morning at 11 AM.

---

## Phase 6 — Kill the N+1 in the prediction endpoints (BUG-10) — **DONE 2026-09-15**

**This is the "it's so slow" fix.** Backend half.

### Outcome

Two changes, both applied to all three prediction paths (`get_best_opportunities`,
`_save_picks`, `get_upcoming_opportunities`):

1. **`SnapshotIndex`** (new, in `features.py`) — fetches every snapshot for the relevant games
   once and serves the same filtered, identically-ordered views the four per-outcome feature
   helpers used to query for. Each helper takes an optional `index=`; when absent it runs the
   original query, so there is exactly one implementation of each calculation.
2. **Batched inference** — the loop now collects feature rows and predicts once per sport.

**The second change was the whole win.** Removing the ~2,250 queries alone only moved
23s → 20.9s. Profiling showed why: single-row `predict_movement()` costs **86.9ms** through the
calibrated ensemble, while **450 rows batched cost 110ms in total** — a 355x difference. The
queries were never the bottleneck; inference was.

| Endpoint | Before | After |
|---|---|---|
| `best-opportunities` (today, ev>=2) | 23.0s | **0.21s** |
| `best-opportunities` (today, ev>=0) | 22.7s | **0.16s** |
| `best-opportunities` (all upcoming) | 34.0s | **0.22s** |
| `upcoming-opportunities` | **>259s (timed out)** | **0.17s** |
| `save-daily-picks` | not measured | runs clean, dedupes correctly |

### Output equivalence — verified
Three query variants captured before the change and re-run after: **same row counts, same picks,
same ordering**, zero non-numeric differences. The only deltas are floating-point noise from
batched matrix ops, max **1.78e-15**. One variant was byte-identical.

(An intermediate check appeared to show real differences; that was a filename-hash mix-up in the
comparison script, not a behavioural change.)

### Updated measurements (2026-09-15)
Re-measured after Phases 1-5, with the retrained model:

| Condition | Latency |
|---|---|
| Cold (includes lazy per-sport model load) | **111.9s** |
| Warm | **25.5s / 27.1s** |

The warm number confirms the original 23.1s baseline. The cold number is new and matters: the
per-sport model is loaded lazily on first use, so the **first** Best EV+ visit after any API
restart costs nearly two minutes. With `--reload` active, that is every code edit.

### The bug
`/api/ml/best-opportunities` takes **23.1 seconds** measured. It is the default Best EV+ tab, so
the dashboard's primary view blocks for 23 seconds on every visit.

The cause is a per-outcome N+1. The hot loop runs per snapshot × per outcome and calls four feature
functions, each issuing its own queries:

| Call | Queries each |
|---|---|
| `calculate_consensus_line` | 1 |
| `calculate_line_spread` | 1 |
| `calculate_temporal_features` | 1 |
| `calculate_bookmaker_features` | 2 |

At ~180 snapshots/day × ~2.5 outcomes that is ~450 iterations × 5 queries ≈ **2,250 queries**, plus
~450 single-row `predict_movement()` calls through a calibrated ensemble — single-row inference is
the worst case for XGBoost + isotonic calibration.

The same pattern is in `save_daily_picks` and `get_upcoming_opportunities` in the same file.

### Changes
- Bulk-prefetch consensus lines, line spreads, temporal and bookmaker features for all
  `(game, market_type, outcome)` combinations in the result set **before** the loop, into `_map`
  dicts — the pattern already used for bookmakers, teams and sports at the top of the same function.
- Batch inference: assemble the full feature matrix, call `predict_movement()` **once**, and zip
  results back. Single-row calls in a loop are the dominant cost after the queries.
- Apply the same treatment to `save_daily_picks` and `get_upcoming_opportunities`.
- Warm the per-sport models at app startup instead of lazily on first request — the 112s cold
  path is otherwise paid by whoever opens the dashboard first after a restart.
- Add a short TTL cache (60s) on `best-opportunities` keyed by its query params — consistent with
  the existing cache on `/api/stats`, `/api/bookmakers`, `/api/clv-history`.

**Do not change:** any threshold, the EV score formula, the dedup rule, or which picks qualify.
**This phase must be output-identical.** Capture the current response body before the change and
diff it after — same picks, same order, same EV scores.

### Validation
- Latency under **1.5s** warm, under 3s cold.
- Response body byte-identical to the pre-change capture for the same params.
- `save_daily_picks` at 11 AM still writes the same picks it would have.

---

## Phase 7 — Frontend performance (BUG-11) — **DONE 2026-09-17**

**"It's so slow" — frontend half.** Phase 6 must land first; no amount of frontend work fixes a
23-second endpoint.

### Outcome

**The build was already broken before this phase started** — `npm run build` failed with 7
TypeScript errors (4 unused `React` imports under the new JSX transform, a `useRef<number>()`
missing its argument, and a `style` prop passed to `GlassCard` which does not declare one). The
frontend could not be production-built at all. Fixed as part of this phase.

Shipped:
- Deleted `OpportunitiesExplorer.tsx` (349 lines, imported by nothing) and the dead `npm create
  vite` cluster: `main.ts`, `counter.ts`, `style.css`, `typescript.svg`. `index.html` loads
  `main.tsx`, so `main.ts` and everything it referenced was unreachable.
- **`src/api/client.ts`** — one `API_BASE` (now overridable via `VITE_API_BASE`), a typed
  `fetchJSON`/`postJSON` with a 30s timeout, `AbortController` support and an `ApiError` carrying
  status and path. All **38 call sites across 9 files** migrated; **zero raw `fetch(` calls remain**
  outside the client.
- **`src/hooks/usePolling.ts`** — the Markets and Pred Markets 60s timers are torn down when the
  tab is hidden and fire once immediately on return. A backgrounded dashboard no longer polls.
- **`src/components/States.tsx`** — one `LoadingState` / `ErrorState` / `EmptyState`, replacing
  hand-rolled spinners that had drifted to different colours and sizes per tab.

**Correction to this phase's stated expectation:** the roadmap predicted a bundle-size drop from
deleting dead code. It did not happen, and could not have — unreferenced modules were never
bundled in the first place (659.90 kB → 659.33 kB, noise). The real wins are that the build works,
there is one place to point the app at a backend, and hidden tabs stop making requests.

Build is clean, TypeScript strict passes, and the `any` count is unchanged at 2 — none added.

### Findings
- **Dead code shipping in the bundle.** `src/components/OpportunitiesExplorer.tsx` (349 lines) is
  imported by nothing. `src/counter.ts`, `src/main.ts`, `src/style.css`, `src/typescript.svg` are
  unreferenced Vite scaffolding from `npm create vite`. Delete all of it.
- **No API layer.** `localhost:8000` is hardcoded across **9 files** at **38 fetch sites**. No
  shared client, no timeout, no error convention, and no way to point the app at anything but a
  dev backend.
- **Polling cost.** `ArbOpportunities` and `PredictionMarkets` each auto-refresh on a 60s timer
  with full refetches, regardless of whether the tab is visible.
- Loading state is re-implemented ad hoc in 11 components.

### Changes
- Delete the dead component and the scaffolding files.
- Extract `src/api/client.ts`: one `API_BASE` from `import.meta.env`, a typed `fetchJSON` helper
  with timeout and consistent error shape. Migrate all 38 call sites.
- Gate the 60s polls on document visibility; skip refetch when the tab is hidden.
- Extract one shared `<LoadingState>` / `<ErrorState>` / `<EmptyState>` trio and use them
  everywhere.
- Keep the existing `React.memo` / `useMemo` / debounce / `Map`-ref-cache work — it is correct and
  documented in `CLAUDE.md`.

**Do not change:** tab structure or navigation. Navigation was explicitly confirmed as *not* a pain
point — the complaint is speed and appearance. Leave the 8-tab IA and the `useState` view switching
alone.

### Validation
- `npm run build` clean, TypeScript strict, no new `any` (currently only 2 in the codebase — do not
  add a third).
- Best EV+ tab interactive in under 2s.
- Bundle size drops (measure before/after).
- Switching away from Markets/Pred Markets stops their network traffic.

---

## Phase 8 — Visual overhaul (BUG-12) — **DONE 2026-09-17**

**"It's so visually ugly."** Scope is restyle only — same tabs, same data, same routing.

### Outcome — the dashboard had no CSS at all

**This is the actual answer to "it's so visually ugly", and it was not a design problem.**

`package.json` had Tailwind **v4.1.18** with the v4 PostCSS plugin, but `src/index.css` still used
the v3 `@tailwind base/components/utilities` directives and the theme lived in a v3-style
`tailwind.config.js`. Tailwind v4 silently ignores both. Measured, by building each way:

| | Built CSS | `bg-white/5` | `text-gray-400` | `rounded-lg` | `grid-cols-4` |
|---|---|---|---|---|---|
| Old (v3 directives under v4) | **6,265 B** | absent | absent | absent | absent |
| New (`@import "tailwindcss"`) | **47,890 B** | present | present | present | present |

The old stylesheet was the preflight reset and nothing else. **Every utility class across ~4,000
lines of TSX was a no-op** — no cards, no grid, no spacing, no colour. The dashboard was rendering
as unstyled HTML. This almost certainly broke at the v3 to v4 upgrade and was never caught, because
`npm run build` was *also* broken (see Phase 7) so nobody built it.

A visible consequence: `clv-positive-*` / `clv-negative-*`, used in five places in
`BookmakerPerformance`, resolved to nothing, so those CLV figures had no colour.

### Also shipped
- **Design tokens in `@theme`** — surfaces (`surface` / `panel` / `panel-raised`), two line weights,
  three text weights, and semantic money colours (`pos`, `neg`, `warn`, `info`). Dead config
  (fade-in, slide-up, the gradient backgrounds, `backdrop-blur-xs`) was dropped; only
  `animate-count-up` was actually used. `tailwind.config.js` deleted.
- **110 ad-hoc utility literals collapsed onto tokens.** The codebase mixed six border weights
  (`border-white/5,8,10,15,20,30`) and eight background weights; these now resolve to two line
  tokens and one panel token. Zero ad-hoc card-chrome literals remain.
- **Debug styling removed from `GameDetailsModal`** — the modal backdrop was shipping
  `rgba(255,0,0,0.5)` with a `10px solid yellow` border, wrapping a `5px solid lime` div, wrapping
  a `3px solid cyan` GlassCard, plus a `console.log` on every backdrop click. Inline styles work
  without Tailwind, so unlike everything else **this was fully visible**. Replaced with a normal
  `bg-black/70 backdrop-blur-sm` scrim.
- **One Recharts theme** (`src/charts/theme.ts`). Four components each had their own
  `tooltipStyle` and repeated axis/grid hex values; there are now **zero raw hex colours** in any
  chart component.
- **Tabular numerals** on all tables and on `AnimatedCounter`, so figures stop reflowing as digits
  change.
- **Emojis removed** — 16 occurrences across 6 files, against the standing "no emojis in UI" rule.
  Decorative ones were dropped; `✅`/`❌` were replaced with `text-pos` / `text-neg` styling that
  carries the same meaning.
- **Committed to dark-only.** `color-scheme: light dark` claimed a light theme that never existed;
  it is now `dark`, and the page background comes from the surface token instead of a stray
  `#242424`.

### Findings
- **No design tokens.** Styling is ad-hoc repeated Tailwind strings. The same input styling is
  retyped 15 times, the same card chrome 12 times, `text-sm text-gray-400` 19 times.
- **`GlassCard` is bypassed.** A `GlassCard` component exists, but 12 sites hand-roll
  `bg-white/5 rounded-lg p-4 border border-white/10` — which is what `GlassCard` already does,
  slightly differently each time. Hence the inconsistent look.
- **Tailwind version mismatch.** `package.json` has Tailwind **v4.1.18**, but `src/index.css`
  uses the v3 `@tailwind base/components/utilities` directives instead of v4's
  `@import "tailwindcss"`. Verify which path is actually active before restyling — building on a
  misconfigured setup will waste the whole phase.
- Two stylesheets (`index.css` 28 lines, `style.css` 96 lines) with `style.css` unreferenced.
- Dark theme is hardcoded (`#242424` on `:root`), with `color-scheme: light dark` declared but no
  light palette — a half-implemented theme.

### Changes
1. **Resolve the Tailwind v3/v4 config question first.** Everything else depends on it.
2. Define tokens in one place — surface/border/text ramps, semantic pos/neg/warn colors for P&L,
   spacing and radius scale. Replace the repeated literals.
3. Consolidate on `GlassCard`; delete the 12 hand-rolled duplicates. One card chrome everywhere.
4. Typography scale and tabular numerals for all money and percentage columns — figures currently
   jitter as values change.
5. Restyle tab by tab, in this order: **Best EV+** (the tab that matters daily) → Overview →
   Bankroll → Pred Markets → Markets → the rest.
6. Unify Recharts theming — one shared chart config for axes, grid, tooltip, colors, rather than
   per-component styling.
7. Commit to dark-only and remove the dead `color-scheme: light dark`, or implement light properly.
   Pick one; do not leave it half-done.

**Do not change:** any displayed value, calculation, threshold or dedup rule. This phase moves
pixels only. **No emojis** (standing rule).

### Validation
- Every tab visually reviewed against the previous build; no data regressions.
- One card component, one input style, one chart theme across the app — grep for the old literals
  returns nothing.
- `npm run build` clean.
- Numbers do not shift horizontally as they update.

---

## Phase order and rationale

| # | Phase | Why here |
|---|---|---|
| 1 | Logging | Everything downstream needs it; it is why BUG-8 hid for 4 months |
| 2 | Low-price size cap | Only phase touching live position sizing |
| 3 | Score backfill | Unblocks Phase 5; fixes P&L honesty |
| 3a | **Blocking async handlers (BUG-15)** | **Was starving arb/PM/settlement jobs** |
| 3b | **NBA scores (BUG-13)** | **Urgent — hard blocker on the October NBA re-enable** |
| 3c | MLB matching (BUG-14) | Small, bounded; finishes the backfill |
| 4 | Stale trade cleanup | Data integrity; cheap |
| 5 | Retrain | Must follow Phase 3 — needs complete labels |
| 6 | Backend N+1 | The actual source of "slow" |
| 7 | Frontend perf | Pointless before Phase 6 |
| 8 | Visual overhaul | Last; purely cosmetic, no dependencies |

Phases 1–5 are correctness and money. Phases 6–8 are the dashboard complaint. Nothing in 6–8 may
change a displayed number.
