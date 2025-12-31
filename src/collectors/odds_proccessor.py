import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.models.database import Bookmaker, ClosingLine, Game, OddsSnapshot, Sport, Team

logger = logging.getLogger(__name__)


class OddsDataProcessor:
    """
    Transform API responses into database models.

    Responsibilities:
    - Parse API JSON
    - Create/update Sport, Team, Game, Bookmaker records
    - Store OddsSnapshot records
    - Handle duplicates gracefully
    """

    def __init__(self, session: Session):
        """
        Initialize the processor with a database session.

        Args:
            session: SQLAlchemy session for database operations
        """
        self.session = session

    def process_odds_response(self, sport_key: str, api_data: dict) -> int:
        """
        Process complete API response and store odds snapshots.

        Args:
            sport_key: Sport identifier (e.g., 'basketball_nba')
            api_data: Response from OddsAPIClient.get_odds()

        Returns:
            Number of odds snapshots successfully stored

        Example api_data structure:
            {
                "data": [
                    {
                        "id": "game_id",
                        "sport_key": "basketball_nba",
                        "sport_title": "NBA",
                        "commence_time": "2024-01-15T00:00:00Z",
                        "home_team": "Lakers",
                        "away_team": "Celtics",
                        "bookmakers": [
                            {
                                "key": "fanduel",
                                "title": "FanDuel",
                                "markets": [
                                    {
                                        "key": "h2h",
                                        "outcomes": [...]
                                    }
                                ]
                            }
                        ]
                    }
                ],
                "remaining_requests": 450
            }
        """
        games_data = api_data.get("data", [])
        snapshots_stored = 0

        if not games_data:
            logger.warning(f"No games data found for sport: {sport_key}")
            return 0

        logger.info(f"Processing {len(games_data)} games for sport: {sport_key}")

        for game_data in games_data:
            try:
                # Get or create sport
                sport = self._get_or_create_sport(
                    sport_key=game_data["sport_key"], sport_name=game_data["sport_title"]
                )

                # Get or create teams
                home_team = self._get_or_create_team(
                    team_key=self._normalize_team_key(game_data["home_team"]),
                    team_name=game_data["home_team"],
                    sport_id=sport.id,
                )

                away_team = self._get_or_create_team(
                    team_key=self._normalize_team_key(game_data["away_team"]),
                    team_name=game_data["away_team"],
                    sport_id=sport.id,
                )

                # Get or create game
                game = self._get_or_create_game(
                    sport_id=sport.id,
                    home_team_id=home_team.id,
                    away_team_id=away_team.id,
                    commence_time=datetime.fromisoformat(
                        game_data["commence_time"].replace("Z", "+00:00")
                    ),
                )

                # Process bookmakers and their odds
                for bookmaker_data in game_data.get("bookmakers", []):
                    bookmaker = self._get_or_create_bookmaker(
                        bookmaker_key=bookmaker_data["key"],
                        bookmaker_name=bookmaker_data["title"],
                    )

                    # Process each market (h2h, spreads, totals)
                    for market in bookmaker_data.get("markets", []):
                        snapshot = self._create_odds_snapshot(
                            game_id=game.id,
                            bookmaker_id=bookmaker.id,
                            market_type=market["key"],
                            outcomes=market["outcomes"],
                        )

                        if snapshot:
                            snapshots_stored += 1

                # Commit after each game to avoid losing all data on error
                self.session.commit()

            except Exception as e:
                logger.error(f"Error processing game {game_data.get('id', 'unknown')}: {e}")
                self.session.rollback()
                continue

        logger.info(f"Successfully stored {snapshots_stored} odds snapshots")
        return snapshots_stored

    def _get_or_create_sport(self, sport_key: str, sport_name: str) -> Sport:
        """
        Get existing sport or create new one.

        Args:
            sport_key: Unique sport identifier
            sport_name: Display name for the sport

        Returns:
            Sport model instance
        """
        stmt = select(Sport).where(Sport.key == sport_key)
        sport = self.session.execute(stmt).scalar_one_or_none()

        if not sport:
            sport = Sport(key=sport_key, name=sport_name, active=True)
            self.session.add(sport)
            self.session.flush()
            logger.info(f"Created new sport: {sport_name} ({sport_key})")

        return sport

    def _get_or_create_team(self, team_key: str, team_name: str, sport_id: int) -> Team:
        """
        Get existing team or create new one.

        Args:
            team_key: Unique team identifier (normalized)
            team_name: Display name for the team
            sport_id: ID of the sport this team belongs to

        Returns:
            Team model instance
        """
        stmt = select(Team).where(Team.key == team_key, Team.sport_id == sport_id)
        team = self.session.execute(stmt).scalar_one_or_none()

        if not team:
            team = Team(key=team_key, name=team_name, sport_id=sport_id)
            self.session.add(team)
            self.session.flush()
            logger.debug(f"Created new team: {team_name} ({team_key})")

        return team

    def _get_or_create_game(
        self, sport_id: int, home_team_id: int, away_team_id: int, commence_time: datetime
    ) -> Game:
        """
        Get existing game or create new one.

        Args:
            sport_id: ID of the sport
            home_team_id: ID of the home team
            away_team_id: ID of the away team
            commence_time: Scheduled start time for the game

        Returns:
            Game model instance

        Note:
            Uses a ±12 hour time window to match games, preventing issues with
            back-to-back games (doubleheaders, playoff series).
        """
        # Look for existing game with same teams within time window
        # ±12 hours handles schedule changes while avoiding back-to-back confusion
        time_window = timedelta(hours=12)
        stmt = (
            select(Game)
            .where(
                Game.sport_id == sport_id,
                Game.home_team_id == home_team_id,
                Game.away_team_id == away_team_id,
                Game.commence_time.between(
                    commence_time - time_window, commence_time + time_window
                ),
            )
            .order_by(Game.commence_time.desc())
            .limit(1)
        )
        game = self.session.execute(stmt).scalar_one_or_none()

        # Create new game if none exists or update commence_time if changed
        if not game:
            game = Game(
                sport_id=sport_id,
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                commence_time=commence_time,
                completed=False,
            )
            self.session.add(game)
            self.session.flush()
            logger.debug(
                f"Created new game: team {home_team_id} vs team {away_team_id} at {commence_time}"
            )
        elif game.commence_time != commence_time:
            # Update commence time if schedule changed
            game.commence_time = commence_time
            logger.debug(f"Updated game {game.id} commence time to {commence_time}")

        return game

    def _get_or_create_bookmaker(self, bookmaker_key: str, bookmaker_name: str) -> Bookmaker:
        """
        Get existing bookmaker or create new one.

        Args:
            bookmaker_key: Unique bookmaker identifier
            bookmaker_name: Display name for the bookmaker

        Returns:
            Bookmaker model instance
        """
        stmt = select(Bookmaker).where(Bookmaker.key == bookmaker_key)
        bookmaker = self.session.execute(stmt).scalar_one_or_none()

        if not bookmaker:
            bookmaker = Bookmaker(key=bookmaker_key, name=bookmaker_name, active=True)
            self.session.add(bookmaker)
            self.session.flush()
            logger.info(f"Created new bookmaker: {bookmaker_name} ({bookmaker_key})")

        return bookmaker

    def _create_odds_snapshot(
        self, game_id: int, bookmaker_id: int, market_type: str, outcomes: List[dict]
    ) -> Optional[OddsSnapshot]:
        """
        Create an odds snapshot record.

        Args:
            game_id: ID of the game
            bookmaker_id: ID of the bookmaker
            market_type: Type of market ('h2h', 'spreads', 'totals')
            outcomes: List of outcome dictionaries from API

        Returns:
            OddsSnapshot instance if created, None if duplicate

        Example outcomes for h2h:
            [{"name": "Lakers", "price": -150}, {"name": "Celtics", "price": 130}]

        Example outcomes for spreads:
            [{"name": "Lakers", "point": -3.5, "price": -110}, ...]
        """
        snapshot = OddsSnapshot(
            game_id=game_id,
            bookmaker_id=bookmaker_id,
            market_type=market_type,
            outcomes=outcomes,
        )

        try:
            self.session.add(snapshot)
            self.session.flush()
            return snapshot

        except IntegrityError:
            # Duplicate snapshot (same game/bookmaker/market/timestamp)
            self.session.rollback()
            logger.debug(
                f"Duplicate snapshot skipped: game={game_id}, "
                f"bookmaker={bookmaker_id}, market={market_type}"
            )
            return None

    def _normalize_team_key(self, team_name: str) -> str:
        """
        Normalize team name to create a consistent key.

        Args:
            team_name: Team display name from API

        Returns:
            Normalized team key (lowercase, underscores)

        Example:
            "Los Angeles Lakers" -> "los_angeles_lakers"
        """
        return team_name.lower().replace(" ", "_").replace("-", "_")

    def store_closing_line(
        self, game_id: int, bookmaker_id: int, market_type: str, outcomes: List[dict]
    ) -> Optional[ClosingLine]:
        """
        Store the closing line for a game.

        This should be called when a game is about to commence to capture
        the final odds for CLV calculation.

        Args:
            game_id: ID of the game
            bookmaker_id: ID of the bookmaker
            market_type: Type of market ('h2h', 'spreads', 'totals')
            outcomes: List of outcome dictionaries

        Returns:
            ClosingLine instance if created, None if already exists
        """
        closing_line = ClosingLine(
            game_id=game_id,
            bookmaker_id=bookmaker_id,
            market_type=market_type,
            outcomes=outcomes,
            closed_at=datetime.now(timezone.utc),
        )

        try:
            self.session.add(closing_line)
            self.session.commit()
            logger.info(
                f"Stored closing line: game={game_id}, "
                f"bookmaker={bookmaker_id}, market={market_type}"
            )
            return closing_line

        except IntegrityError:
            # Closing line already exists
            self.session.rollback()
            logger.warning(
                f"Closing line already exists: game={game_id}, "
                f"bookmaker={bookmaker_id}, market={market_type}"
            )
            return None
