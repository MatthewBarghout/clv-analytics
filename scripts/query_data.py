"""
Query the database to see collected data.
Run: poetry run python -m scripts.query_data
"""
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

# Connect to database
db_url = os.getenv("DATABASE_URL")
engine = create_engine(db_url)

queries = [
    ("GAMES (first 3):", """
        SELECT id, home_team_id, away_team_id, commence_time, completed
        FROM games
        LIMIT 3
    """),

    ("TEAMS (first 10):", """
        SELECT id, name, key
        FROM teams
        LIMIT 10
    """),

    ("ODDS SNAPSHOTS with Bookmaker (first 5):", """
        SELECT
            os.id,
            os.game_id,
            b.name as bookmaker,
            os.market_type,
            os.outcomes,
            os.timestamp
        FROM odds_snapshots os
        JOIN bookmakers b ON os.bookmaker_id = b.id
        LIMIT 5
    """),

    ("COUNT BY MARKET TYPE:", """
        SELECT market_type, COUNT(*) as count
        FROM odds_snapshots
        GROUP BY market_type
    """),

    ("COUNT BY BOOKMAKER:", """
        SELECT b.name, COUNT(*) as snapshot_count
        FROM odds_snapshots os
        JOIN bookmakers b ON os.bookmaker_id = b.id
        GROUP BY b.name
    """),

    ("GAMES WITH TEAM NAMES AND SNAPSHOT COUNTS:", """
        SELECT
            g.id,
            ht.name as home_team,
            at.name as away_team,
            g.commence_time,
            COUNT(os.id) as total_snapshots
        FROM games g
        JOIN teams ht ON g.home_team_id = ht.id
        JOIN teams at ON g.away_team_id = at.id
        LEFT JOIN odds_snapshots os ON os.game_id = g.id
        GROUP BY g.id, ht.name, at.name, g.commence_time
        ORDER BY g.commence_time
        LIMIT 5
    """)
]

def format_row(row):
    """Format a row for display."""
    return " | ".join(str(v) for v in row)

def run_query(title, query):
    """Run a query and display results."""
    print(f"\n{'='*80}")
    print(f"{title}")
    print('='*80)

    try:
        with engine.connect() as conn:
            result = conn.execute(text(query))
            rows = result.fetchall()

            if not rows:
                print("(No data found)")
                return

            # Print column headers
            headers = result.keys()
            print(format_row(headers))
            print('-' * 80)

            # Print rows
            for row in rows:
                print(format_row(row))

            print(f"\nTotal rows: {len(rows)}")

    except Exception as e:
        print(f"Error: {e}")

def main():
    print("\nCLV ANALYTICS DATABASE QUERY RESULTS")
    print("=" * 80)

    for title, query in queries:
        run_query(title, query)

    print(f"\n{'='*80}")
    print("Query complete")
    print('='*80 + "\n")

if __name__ == "__main__":
    main()
