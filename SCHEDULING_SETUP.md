# NBA Odds Collection Scheduling Setup

This guide explains how to set up automated collection of NBA odds at scheduled times.

## Collection Schedule

- **10:00 AM ET**: Opening lines (regular snapshots)
- **6:00 PM ET**: Closing lines (stores both snapshots and closing lines)

## Quick Start

### Test Manual Collection

Before setting up automation, test the collection manually:

```bash
# Test regular collection
./scripts/run_collection.sh

# Test closing-only mode
./scripts/run_collection.sh --closing-only

# Check the logs
ls -lh logs/
```

## macOS Setup (launchd - Recommended)

launchd is the native macOS task scheduler and is preferred over cron.

### 1. Install the Launch Agents

Copy the plist files to your LaunchAgents directory:

```bash
# Opening lines (10am ET)
cp launchd/com.clvanalytics.odds.opening.plist ~/Library/LaunchAgents/

# Closing lines (6pm ET)
cp launchd/com.clvanalytics.odds.closing.plist ~/Library/LaunchAgents/
```

### 2. Load the Launch Agents

```bash
# Load opening lines collector
launchctl load ~/Library/LaunchAgents/com.clvanalytics.odds.opening.plist

# Load closing lines collector
launchctl load ~/Library/LaunchAgents/com.clvanalytics.odds.closing.plist
```

### 3. Verify Setup

```bash
# Check if agents are loaded
launchctl list | grep clvanalytics

# You should see:
# com.clvanalytics.odds.opening
# com.clvanalytics.odds.closing
```

### 4. Test Run Immediately

```bash
# Trigger opening lines collection now (for testing)
launchctl start com.clvanalytics.odds.opening

# Trigger closing lines collection now (for testing)
launchctl start com.clvanalytics.odds.closing

# Check logs
tail -f logs/launchd-opening.log
tail -f logs/launchd-closing.log
```

### Managing Launch Agents

```bash
# Stop a collector
launchctl stop com.clvanalytics.odds.opening

# Unload (disable) a collector
launchctl unload ~/Library/LaunchAgents/com.clvanalytics.odds.opening.plist

# Reload after making changes to plist
launchctl unload ~/Library/LaunchAgents/com.clvanalytics.odds.opening.plist
launchctl load ~/Library/LaunchAgents/com.clvanalytics.odds.opening.plist
```

## Alternative: Cron Setup

If you prefer cron over launchd:

### 1. Edit Crontab

```bash
crontab -e
```

### 2. Add These Lines

```cron
# NBA Odds Collection
# Opening lines at 10am ET daily
0 10 * * * cd /Users/matthewbarghout/clv-analytics && ./scripts/run_collection.sh

# Closing lines at 6pm ET daily
0 18 * * * cd /Users/matthewbarghout/clv-analytics && ./scripts/run_collection.sh --closing-only
```

### 3. Verify Cron Jobs

```bash
crontab -l
```

### Note on Cron Timezone

Cron uses the system timezone. If your Mac is set to Eastern Time, these will run at the correct times. To verify:

```bash
date
```

## Monitoring

### Check Collection Logs

```bash
# View latest collection log
ls -lt logs/collection_*.log | head -1 | xargs tail -f

# View all logs from today
ls logs/collection_$(date +%Y-%m-%d)*.log

# Check for errors
grep -i error logs/collection_*.log
```

### Check API Usage

```bash
# See remaining API requests in latest log
grep "API requests remaining" logs/collection_*.log | tail -1
```

### Database Queries

```bash
# Run the query script to see collected data
poetry run python -m scripts.query_data

# Count odds snapshots by date
poetry run python -c "
from sqlalchemy import create_engine, func
from src.models.database import OddsSnapshot
from dotenv import load_dotenv
import os

load_dotenv()
engine = create_engine(os.getenv('DATABASE_URL'))

with engine.connect() as conn:
    result = conn.execute('''
        SELECT DATE(timestamp) as date, COUNT(*) as snapshots
        FROM odds_snapshots
        GROUP BY DATE(timestamp)
        ORDER BY date DESC
        LIMIT 7
    ''')
    for row in result:
        print(f'{row[0]}: {row[1]} snapshots')
"
```

## Troubleshooting

### Collection Not Running

1. Check if launchd agents are loaded:
   ```bash
   launchctl list | grep clvanalytics
   ```

2. Check launchd error logs:
   ```bash
   tail logs/launchd-opening-error.log
   tail logs/launchd-closing-error.log
   ```

3. Verify database is running:
   ```bash
   docker-compose ps
   ```

4. Test manual collection:
   ```bash
   ./scripts/run_collection.sh
   ```

### API Key Issues

If you see "Invalid API key" errors:

```bash
# Verify .env file has ODDS_API_KEY
grep ODDS_API_KEY .env

# Test API key manually
curl "https://api.the-odds-api.com/v4/sports?apiKey=YOUR_KEY"
```

### Database Connection Issues

```bash
# Check DATABASE_URL in .env
grep DATABASE_URL .env

# Verify PostgreSQL is running
docker-compose ps postgres

# Test connection
poetry run python -c "
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
load_dotenv()
engine = create_engine(os.getenv('DATABASE_URL'))
print('Connected successfully!' if engine.connect() else 'Failed')
"
```

### Rate Limit Warnings

The Odds API free tier has 500 requests/month. Monitor usage:

```bash
# Check remaining requests from logs
grep "API requests remaining" logs/collection_*.log | tail -5
```

If running low:
- Reduce collection frequency
- Collect fewer bookmakers
- Upgrade to paid tier

## Logs Cleanup

Logs will accumulate over time. Set up automatic cleanup:

```bash
# Add to crontab (runs weekly on Sunday at 2am)
0 2 * * 0 find /Users/matthewbarghout/clv-analytics/logs -name "collection_*.log" -mtime +30 -delete
```

This removes collection logs older than 30 days.

## What Gets Collected

### Opening Lines (10am ET)
- Regular odds snapshots
- All markets: moneyline (h2h), spreads, totals
- All configured bookmakers: Pinnacle, FanDuel, DraftKings, ESPNbet

### Closing Lines (6pm ET)
- Only stores closing lines (no snapshots in `--closing-only` mode)
- Only for games starting within 2 hours
- Same markets and bookmakers

This two-collection strategy:
- Captures opening lines when they're first posted
- Captures closing lines right before games start
- Allows accurate CLV calculation
