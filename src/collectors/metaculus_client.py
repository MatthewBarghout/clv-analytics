"""Metaculus forecast aggregation client.

Uses the Metaculus API v2 — token auth required (METACULUS_API_TOKEN env var).
API: https://www.metaculus.com/api2

NOTE: The api2 text search param is non-functional. We fetch a page of open
questions and do client-side word-overlap matching. Coverage is best for
macro/political questions (KXFED, KXPRES, KXECON); sparse for sports/crypto.
"""
import logging
import os
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

METACULUS_BASE_URL = "https://www.metaculus.com/api2"
REQUEST_TIMEOUT = 10
SIMILARITY_THRESHOLD = 0.40
_FETCH_LIMIT = 100  # questions per call — keeps latency under ~1s

# Simple stopwords to ignore in similarity matching
_STOPWORDS = {"will", "the", "a", "an", "of", "in", "to", "be", "is", "are",
              "was", "were", "it", "its", "for", "on", "at", "by", "or", "and",
              "before", "after", "than", "that", "this", "with", "have", "has"}


def _word_overlap(a: str, b: str) -> float:
    a_words = set(a.lower().split()) - _STOPWORDS
    b_words = set(b.lower().split()) - _STOPWORDS
    if not a_words or not b_words:
        return 0.0
    return len(a_words & b_words) / min(len(a_words), len(b_words))


class MetaculusClient:
    """Client for the Metaculus API (token auth)."""

    def __init__(self):
        self.session = requests.Session()
        token = os.getenv("METACULUS_API_TOKEN", "")
        if token:
            self.session.headers.update({"Authorization": f"Token {token}"})

    def _get_open_questions(self, offset: int = 0) -> list:
        """Fetch a page of open binary questions with community predictions."""
        try:
            resp = self.session.get(
                f"{METACULUS_BASE_URL}/questions/",
                params={"status": "open", "type": "binary", "limit": _FETCH_LIMIT, "offset": offset},
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json().get("results", [])
        except requests.exceptions.HTTPError as e:
            logger.warning(f"Metaculus HTTP {e.response.status_code if e.response else '?'} fetching questions")
            return []
        except Exception as e:
            logger.error(f"Metaculus fetch failed: {e}")
            return []

    def get_forecast(self, question_keyword: str) -> Optional[float]:
        """Find the best-matching open Metaculus question and return its community prediction.

        Fetches a single page of open binary questions and does client-side
        word-overlap matching. Returns float in [0, 1] or None.
        """
        questions = self._get_open_questions()
        if not questions:
            return None

        best_sim = 0.0
        best_center = None

        for q in questions:
            title = q.get("title", "")
            sim = _word_overlap(question_keyword, title)
            if sim <= best_sim:
                continue

            nested = q.get("question", {}) or {}
            agg = (nested.get("aggregations") or {}).get("recency_weighted", {})
            latest = agg.get("latest") if agg else None
            centers = latest.get("centers") if latest else None
            if not centers:
                continue

            best_sim = sim
            best_center = float(centers[0])

        if best_sim >= SIMILARITY_THRESHOLD and best_center is not None:
            logger.debug(f"Metaculus match (sim={best_sim:.2f}) for '{question_keyword[:40]}': {best_center:.3f}")
            return best_center

        logger.debug(f"Metaculus no match (best_sim={best_sim:.2f}) for '{question_keyword[:40]}'")
        return None
