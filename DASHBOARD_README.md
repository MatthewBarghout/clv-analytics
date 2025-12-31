# CLV Analytics Dashboard

Full-stack application for analyzing Closing Line Value in sports betting.

## Architecture

- **Backend**: FastAPI (Python)
- **Frontend**: React + TypeScript + Vite
- **Database**: PostgreSQL
- **Charts**: Recharts
- **Styling**: Tailwind CSS

## Prerequisites

- Python 3.11+
- Node.js 18+
- PostgreSQL (via Docker Compose)
- Poetry

## Quick Start

### 1. Start the Database

```bash
docker-compose up -d
```

### 2. Run Database Migrations

```bash
alembic upgrade head
```

### 3. Collect Some Data (Optional but Recommended)

```bash
# Collect NBA odds
python3 -m scripts.collect_odds

# Or run the test collector to get initial data
python3 -m scripts.test_collector
```

### 4. Start the Backend API

In one terminal:

```bash
# Run with uvicorn
python3 -m src.api.main

# Or with poetry
poetry run python -m src.api.main
```

The API will be available at `http://localhost:8000`

API Documentation: `http://localhost:8000/docs`

### 5. Start the Frontend

In another terminal:

```bash
cd frontend
npm run dev
```

The dashboard will be available at `http://localhost:5173`

## API Endpoints

### GET /api/health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-12-30T12:00:00Z"
}
```

### GET /api/stats
Overall CLV statistics.

**Response:**
```json
{
  "mean_clv": 2.5,
  "median_clv": 1.8,
  "total_analyzed": 150,
  "positive_clv_count": 85,
  "positive_clv_percentage": 56.7,
  "by_bookmaker": { ... },
  "by_market_type": { ... }
}
```

### GET /api/games?limit=50
List of recent games with CLV data.

**Query Parameters:**
- `limit` (optional): Number of games to return (default: 50)

### GET /api/clv-history?days=30
CLV trend over time.

**Query Parameters:**
- `days` (optional): Number of days to return (default: 30)

### GET /api/bookmakers
Bookmaker statistics.

## Dashboard Features

### 1. Summary Cards
- **Mean CLV**: Average CLV across all analyzed bets
- **Total Analyzed**: Number of bets with closing line data
- **Positive CLV %**: Percentage of bets that beat the closing line

### 2. CLV Trend Chart
Line chart showing average CLV over time. Helps identify:
- Improving/declining betting performance
- Patterns in line movement
- Best times to place bets

### 3. Bookmaker Comparison
Bar chart comparing average CLV by bookmaker. Shows which books offer the best lines.

### 4. Recent Games Table
Scrollable table of recent games with:
- Game matchup
- Commence time
- Number of odds snapshots collected
- Number of closing lines captured
- Game status (upcoming/completed)

## Development

### Backend Development

```bash
# Install dependencies
poetry install

# Run with auto-reload
uvicorn src.api.main:app --reload

# Run tests
poetry run pytest
```

### Frontend Development

```bash
cd frontend

# Install dependencies
npm install

# Run dev server with hot reload
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Environment Variables

Required in `.env`:

```bash
# Database
DATABASE_URL=postgresql://clv_user:clv_password@localhost:5432/clv_analytics

# API (optional)
ODDS_API_KEY=your_api_key_here
```

## Troubleshooting

### Backend Issues

**"No module named 'fastapi'"**
```bash
poetry add fastapi uvicorn[standard]
```

**"Connection refused" to database**
```bash
# Check if PostgreSQL is running
docker-compose ps

# Restart database
docker-compose restart postgres
```

### Frontend Issues

**Build errors**
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

**API connection failed**
- Ensure backend is running on port 8000
- Check browser console for CORS errors
- Verify `API_BASE` in Dashboard.tsx matches backend URL

### No Data Showing

**If dashboard shows 0 or N/A:**

1. Check if data has been collected:
   ```bash
   python3 -m scripts.query_data
   ```

2. Collect sample data:
   ```bash
   python3 -m scripts.collect_odds
   ```

3. Insert test closing line:
   ```bash
   python3 -m scripts.test_clv_with_fake_data
   ```

## Production Deployment

### Backend

```bash
# Install production dependencies only
poetry install --no-dev

# Run with production server
gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

### Frontend

```bash
cd frontend
npm run build

# Serve with nginx, apache, or any static file server
# Build output is in frontend/dist/
```

## Next Steps

1. Set up automated data collection (see `SCHEDULING_SETUP.md`)
2. Let the system collect data for a few days
3. Analyze which bookmakers consistently offer the best lines
4. Identify optimal bet timing (opening vs closing lines)
5. Build betting models based on positive CLV opportunities

## Tech Stack Details

- **FastAPI**: Modern Python web framework with automatic OpenAPI docs
- **React 18**: Component-based UI framework
- **TypeScript**: Type-safe JavaScript
- **Vite**: Fast build tool and dev server
- **Tailwind CSS**: Utility-first CSS framework
- **Recharts**: React charting library built on D3
- **SQLAlchemy 2.0**: Python ORM with async support
- **Pydantic**: Data validation using Python type hints
