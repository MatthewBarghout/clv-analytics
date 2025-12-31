#!/bin/bash

# Start CLV Analytics Dashboard (Backend + Frontend)

set -e

echo "Starting CLV Analytics Dashboard..."
echo ""

# Get project directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_DIR"

# Check if database is running
echo "Checking database connection..."
if ! docker-compose ps | grep -q "postgres.*Up"; then
    echo "Starting PostgreSQL database..."
    docker-compose up -d postgres
    sleep 3
fi

echo "Database is running"
echo ""

# Start backend in background
echo "Starting FastAPI backend on http://localhost:8000 ..."
python3 -m src.api.main > logs/api.log 2>&1 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"
echo ""

# Wait for backend to start
echo "Waiting for backend to be ready..."
sleep 3

# Start frontend
echo "Starting React frontend on http://localhost:5173 ..."
echo ""
echo "Dashboard will open in your browser"
echo "Press Ctrl+C to stop both backend and frontend"
echo ""

cd frontend
npm run dev

# Cleanup on exit
trap "echo 'Stopping backend...'; kill $BACKEND_PID 2>/dev/null" EXIT
