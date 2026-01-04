#!/bin/bash

echo "Cleaning up old fixed-time batches..."
echo "======================================"
echo ""

# Remove all old batch schedules (keep only opening at 10am)
echo "Unloading old batches..."

launchctl unload ~/Library/LaunchAgents/com.clvanalytics.odds.afternoon.plist 2>/dev/null
rm ~/Library/LaunchAgents/com.clvanalytics.odds.afternoon.plist 2>/dev/null
echo "  ✓ Removed 2pm afternoon batch"

launchctl unload ~/Library/LaunchAgents/com.clvanalytics.odds.pregame.plist 2>/dev/null
rm ~/Library/LaunchAgents/com.clvanalytics.odds.pregame.plist 2>/dev/null
echo "  ✓ Removed 5pm pregame batch"

launchctl unload ~/Library/LaunchAgents/com.clvanalytics.odds.closing.plist 2>/dev/null
rm ~/Library/LaunchAgents/com.clvanalytics.odds.closing.plist 2>/dev/null
echo "  ✓ Removed 6pm closing batch"

launchctl unload ~/Library/LaunchAgents/com.clvanalytics.odds.primetime.plist 2>/dev/null
rm ~/Library/LaunchAgents/com.clvanalytics.odds.primetime.plist 2>/dev/null
echo "  ✓ Removed 7pm primetime batch"

launchctl unload ~/Library/LaunchAgents/com.clvanalytics.odds.late.plist 2>/dev/null
rm ~/Library/LaunchAgents/com.clvanalytics.odds.late.plist 2>/dev/null
echo "  ✓ Removed 10pm late batch"

echo ""
echo "======================================"
echo "Remaining batches:"
echo "  ✓ 10am - Opening lines (kept)"
launchctl list | grep clvanalytics
echo ""
echo "Cleanup complete!"
echo "======================================"
