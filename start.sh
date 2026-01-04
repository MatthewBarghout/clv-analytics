#!/bin/bash

# CLV Analytics Startup Script
# Starts both backend and frontend servers

echo "🚀 Starting CLV Analytics..."
echo ""

# Navigate to project directory
cd "$(dirname "$0")"

# Kill any existing processes on ports 8000 and 5173
echo "Cleaning up existing processes..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:5173 | xargs kill -9 2>/dev/null || true
sleep 1

# Start backend
echo "Starting backend API on http://localhost:8000..."
poetry run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000 > /tmp/clv-backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend started (PID: $BACKEND_PID)"

# Wait for backend to be ready
sleep 3

# Start frontend
echo "Starting frontend on http://localhost:5173..."
cd frontend
npm run dev > /tmp/clv-frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..
echo "Frontend started (PID: $FRONTEND_PID)"

echo ""
echo "✅ Both services are starting..."
echo ""
echo "📊 Dashboard: http://localhost:5173"
echo "🔌 API:       http://localhost:8000"
echo "📄 API Docs:  http://localhost:8000/docs"
echo ""
echo "Logs:"
echo "  Backend:  tail -f /tmp/clv-backend.log"
echo "  Frontend: tail -f /tmp/clv-frontend.log"
echo ""
echo "To stop: pkill -f uvicorn; pkill -f vite"
echo ""

# Show initial logs
sleep 2
echo "=== Backend Status ==="
tail -10 /tmp/clv-backend.log
echo ""
echo "=== Frontend Status ==="
tail -10 /tmp/clv-frontend.log

# Keep script running and tail logs
echo ""
echo "Press Ctrl+C to stop tailing logs (services will keep running)"
echo "==================== LIVE LOGS ===================="
tail -f /tmp/clv-backend.log /tmp/clv-frontend.log
