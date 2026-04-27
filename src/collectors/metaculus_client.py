"""Metaculus forecast aggregation client.

Uses the public Metaculus API v2 — no auth required.
API: https://www.metaculus.com/api2
"""
import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)

METACULUS_BASE_URL = "https://www.metaculus.com/api2"
REQUEST_TIMEOUT = 10


class MetaculusClient:
    """Client for the Metaculus public API."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def get_forecast(self, question_keyword: str) -> Optional[float]:
        """
        Search for a question by keyword and return the community prediction.

        Returns a float in [0, 1] for the best matching active/resolved question,
        or None if no match is found.

        NOTE: Metaculus api2 now returns 403 for unauthenticated requests.
        Returning None until auth is configured.
        """
        return None
