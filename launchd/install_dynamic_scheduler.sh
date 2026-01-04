#!/bin/bash

echo "Installing Dynamic Game-Based Scheduling System"
echo "================================================"
echo ""

# Clean up old batches first
echo "Step 1: Cleaning up old fixed-time batches..."
./launchd/cleanup_old_batches.sh
echo ""

# Install the scheduler (runs at 8am daily)
echo "Step 2: Installing 8am daily scheduler..."
launchctl unload ~/Library/LaunchAgents/com.clvanalytics.scheduler.plist 2>/dev/null
cp /Users/matthewbarghout/clv-analytics/launchd/com.clvanalytics.scheduler.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.clvanalytics.scheduler.plist
echo "✓ Scheduler installed"
echo ""

echo "================================================"
echo "New Dynamic Schedule:"
echo "  8am   - Daily scheduler (checks today's games)"
echo "  10am  - Opening lines batch"
echo "  Auto  - Batches 30min before each game starts"
echo ""
echo "How it works:"
echo "  1. At 8am, scheduler checks today's games"
echo "  2. Creates a batch 30min before each game"
echo "  3. Automatically captures all closing lines"
echo ""
echo "Benefits:"
echo "  ✓ Captures closing lines for ALL games"
echo "  ✓ Uses fewer API calls (only when needed)"
echo "  ✓ Adapts to different schedules each day"
echo ""
echo "To test the scheduler manually:"
echo "  python3 -m scripts.schedule_game_batches"
echo ""
echo "Installation complete!"
echo "================================================"
