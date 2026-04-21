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
        """
        try:
            resp = self.session.get(
                f"{METACULUS_BASE_URL}/questions/",
                params={"search": question_keyword, "limit": 5},
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            questions = data.get("results", [])
            for q in questions:
                resolution = q.get("resolution")
                # Skip ambiguous or annulled resolutions
                if resolution in (-1, -2):
                    continue
                community_pred = q.get("community_prediction")
                if not community_pred:
                    continue
                # community_prediction is a dict with "full" sub-dict containing "q2" (median)
                full = community_pred.get("full")
                if not full:
                    continue
                median = full.get("q2")
                if median is not None:
                    return float(median)
        except Exception as e:
            logger.error(f"Metaculus get_forecast failed for '{question_keyword}': {e}")
        return None
