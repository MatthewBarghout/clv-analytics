# CLV Analytics - Quick Start Guide

## Setup Complete

Your NBA CLV analytics system is ready to use!

## Test Collection Now

```bash
# Test regular collection
./scripts/run_collection.sh

# Test closing-only mode
./scripts/run_collection.sh --closing-only

# View logs
ls -lh logs/
tail -f logs/collection_*.log
```

## Install Scheduled Collection (macOS)

### 1. Copy Launch Agents

```bash
cp launchd/com.clvanalytics.odds.opening.plist ~/Library/LaunchAgents/
cp launchd/com.clvanalytics.odds.closing.plist ~/Library/LaunchAgents/
```

### 2. Load Agents

```bash
launchctl load ~/Library/LaunchAgents/com.clvanalytics.odds.opening.plist
launchctl load ~/Library/LaunchAgents/com.clvanalytics.odds.closing.plist
```

### 3. Verify

```bash
launchctl list | grep clvanalytics
```

### 4. Test Run Now

```bash
launchctl start com.clvanalytics.odds.opening
tail -f logs/launchd-opening.log
```

## Schedule

- **10:00 AM ET**: Opening lines collection
- **6:00 PM ET**: Closing lines collection

## Query Your Data

```bash
# View collected data
poetry run python -m scripts.query_data

# Test CLV calculation
poetry run python -m scripts.test_clv_with_fake_data
```

## Monitor API Usage

```bash
# Check remaining API requests
grep "API requests remaining" logs/collection_*.log | tail -1
```

Free tier: 500 requests/month

## Files Created

- `scripts/collect_odds.py` - Main collection script
- `scripts/run_collection.sh` - Bash wrapper with logging
- `launchd/*.plist` - macOS scheduler configs
- `SCHEDULING_SETUP.md` - Full setup guide
- `logs/` - Collection logs

## Next Steps

1. Let it collect data for a few days
2. Check `closing_lines` table fills up near game times
3. Run CLV calculations on historical bets
4. Build your betting model based on positive CLV opportunities

See `SCHEDULING_SETUP.md` for detailed documentation.
