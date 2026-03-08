"""
TrustLens Phase 2: Fact-Check Verification Layer (Phase 6 async refactor)

Replaces urllib.request with httpx.AsyncClient so the Google Fact Check
API call does not block the event loop during asyncio.gather().

Why httpx over urllib.request:
  urllib.request.urlopen() is fully synchronous — it blocks the OS thread.
  In an asyncio context this stalls the event loop for up to REQUEST_TIMEOUT_SECONDS.
  httpx.AsyncClient.get() releases the event loop during the network wait,
  allowing Phase 3 (domain) and Phase 4 (image) to run concurrently.

calculate_fact_check_score() is CPU-only (pure dict parsing) and intentionally
stays SYNCHRONOUS — wrapping it in async would add overhead with no benefit.
"""

import re
import json
import httpx
import urllib.parse
from typing import Any

from app.config.settings import Config

# Google Fact Check Tools API
FACTCHECK_API_BASE = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
REQUEST_TIMEOUT_SECONDS = 8.0
MAX_CLAIM_LENGTH = 500

# Map textualRating (case-insensitive) to fact-check score 0-100.
RATING_TO_SCORE = {
    "false":         10,
    "mostly false":  25,
    "partly false":  40,
    "partly true":   60,
    "mostly true":   80,
    "true":          90,
}


def _sanitize_claim(claim_text: str) -> str:
    """
    Sanitize claim text before sending to fact-check API.
    Prevents injection and ensures safe query string.
    """
    if not claim_text or not isinstance(claim_text, str):
        return ""
    sanitized = claim_text.strip()
    if len(sanitized) > MAX_CLAIM_LENGTH:
        sanitized = sanitized[:MAX_CLAIM_LENGTH]
    sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", sanitized)
    sanitized = re.sub(r"\s+", " ", sanitized)
    return sanitized


async def search_fact_check(claim_text: str) -> dict[str, Any]:
    """
    Async: Query Google Fact Check Tools API for claims matching the text.

    Returns raw API response or empty structure on failure.
    Never raises — always returns a dict so asyncio.gather() with
    return_exceptions=True has a clean fallback.
    """
    sanitized = _sanitize_claim(claim_text)
    if not sanitized:
        return {"claims": [], "nextPageToken": ""}

    api_key = getattr(Config, "GOOGLE_FACTCHECK_API_KEY", None) or ""
    if not api_key:
        return {"claims": [], "nextPageToken": ""}

    params = urllib.parse.urlencode({"query": sanitized, "key": api_key})
    url = f"{FACTCHECK_API_BASE}?{params}"

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, dict) else {"claims": [], "nextPageToken": ""}

    except httpx.TimeoutException:
        print("⚠️ Fact-check API error: request timed out")
    except httpx.HTTPStatusError as e:
        print(f"⚠️ Fact-check API error: HTTP {e.response.status_code}")
    except httpx.ConnectError as e:
        print(f"⚠️ Fact-check API error: connection failed: {e}")
    except Exception as e:
        print(f"⚠️ Fact-check API error: {e}")

    return {"claims": [], "nextPageToken": ""}


async def search_fact_check_multi(queries: list[str]) -> dict[str, Any]:
    """
    Run search_fact_check concurrently for multiple queries and merge the results.
    Deduplicates claims by their reference URL.
    """
    if not queries:
        return {"claims": [], "nextPageToken": ""}

    unique_queries = list(dict.fromkeys(q.strip() for q in queries if q.strip()))
    if not unique_queries:
        return {"claims": [], "nextPageToken": ""}

    import asyncio
    results = await asyncio.gather(
        *[search_fact_check(q) for q in unique_queries],
        return_exceptions=True
    )

    merged_claims = []
    seen_urls = set()

    for res in results:
        if isinstance(res, dict) and "claims" in res:
            for claim in res["claims"]:
                # Try to get URL to deduplicate
                url = ""
                reviews = claim.get("claimReview", [])
                if reviews and isinstance(reviews, list):
                    url = reviews[0].get("url", "")
                
                # If no URL or not seen, add it
                if not url or url not in seen_urls:
                    if url:
                        seen_urls.add(url)
                    merged_claims.append(claim)

    return {"claims": merged_claims, "nextPageToken": ""}


def calculate_fact_check_score(api_response: dict[str, Any]) -> dict[str, Any]:
    """
    Synchronous: normalize Google Fact Check API response into TrustLens structure.

    Why synchronous: pure dict processing — no I/O. Adding async here would
    add coroutine overhead with zero concurrency benefit.
    """
    default = {
        "factCheckScore":      50,
        "matchFound":          False,
        "verdict":             "No Match",
        "source":              "",
        "referenceURL":        "",
        "confidenceAdjustment": -0.1,
    }

    if not api_response or not isinstance(api_response, dict):
        return default

    claims = api_response.get("claims")
    if not claims or not isinstance(claims, list):
        return default

    first_claim = claims[0]
    reviews = first_claim.get("claimReview")
    if not reviews or not isinstance(reviews, list):
        return default

    review = reviews[0]
    textual_rating = review.get("textualRating") or ""
    rating_lower = textual_rating.lower().strip()

    fact_check_score = RATING_TO_SCORE.get(rating_lower, 50)

    publisher = review.get("publisher") or {}
    publisher_name = publisher.get("name", "") if isinstance(publisher, dict) else ""
    reference_url = review.get("url", "")

    verdict_map = {
        "false":         "False",
        "mostly false":  "Mostly False",
        "partly false":  "Partly False",
        "partly true":   "Partly True",
        "mostly true":   "Mostly True",
        "true":          "True",
    }
    verdict = verdict_map.get(rating_lower, textual_rating or "No Match")

    return {
        "factCheckScore":       max(0, min(100, fact_check_score)),
        "matchFound":           True,
        "verdict":              verdict,
        "source":               str(publisher_name)[:200],
        "referenceURL":         str(reference_url)[:500],
        "confidenceAdjustment": 0.0,
    }
