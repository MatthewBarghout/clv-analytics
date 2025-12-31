"""
Test script for odds collection.
Run: poetry run python -m scripts.test_collector
"""
import logging
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.collectors.odds_api_client import OddsAPIClient
from src.collectors.odds_proccessor import OddsDataProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def main():
    # Load environment
    load_dotenv()

    # Initialize API client
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        print("ERROR: ODDS_API_KEY not found in .env")
        return

    client = OddsAPIClient(api_key=api_key)

    # Setup database
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not found in .env")
        return

    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Initialize processor
    processor = OddsDataProcessor(session=session)

    try:
        # Fetch NBA odds
        print("Fetching NBA odds...")
        response = client.get_odds(
            sport="basketball_nba",
            bookmakers=["pinnacle", "fanduel", "draftkings", "espnbet"],
        )

        # Process and store
        print(f"Processing {len(response['data'])} games...")
        snapshots_stored = processor.process_odds_response(
            sport_key="basketball_nba", api_data=response
        )

        print(f"Successfully stored {snapshots_stored} odds snapshots")
        print(f"API requests remaining: {response.get('remaining_requests')}")

    except Exception as e:
        print(f"Error: {e}")
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()
