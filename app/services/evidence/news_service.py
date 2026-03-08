import logging
import httpx
from typing import List, Dict, Any

from app.config.settings import Config

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 8.0

# Whitelist of high-credibility news sources
TRUSTED_DOMAINS = [
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
    "nytimes.com", "theguardian.com", "washingtonpost.com",
    "wsj.com", "ft.com", "economist.com",
    "thehindu.com", "ndtv.com", "indianexpress.com",
    "aljazeera.com", "dw.com", "npr.org", "pbs.org"
]

def search_news_articles(claim: str) -> List[Dict[str, Any]]:
    """
    Queries NewsAPI for recent articles related to the claim.
    Returns up to 15 top structured article results from trusted domains.
    Uses httpx (consistent with the rest of the codebase) for HTTP requests.
    """
    api_key = getattr(Config, "NEWS_API_KEY", None) or ""
    if not api_key:
        print("⚠️ [NewsService] NEWS_API_KEY is missing. Skipping news retrieval.")
        return []

    if not claim:
        return []

    try:
        params = {
            "q": claim,
            "domains": ",".join(TRUSTED_DOMAINS),
            "sortBy": "relevancy",
            "pageSize": 15,  # Fetch more so evidence_ranker has a good pool to rank
            "language": "en",
            "apiKey": api_key
        }

        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            response = client.get("https://newsapi.org/v2/everything", params=params)

        if response.status_code == 200:
            articles = response.json().get("articles", [])
            return [
                {
                    "title": a.get("title"),
                    "source": a.get("source", {}).get("name", "Unknown Publisher"),
                    "url": a.get("url"),
                    "description": a.get("description") or a.get("content", "No description provided."),
                    "publishedAt": a.get("publishedAt")  # Important for recency filtering
                }
                for a in articles
                if a.get("title") != "[Removed]" and a.get("url")
            ]
        else:
            print(f"❌ [NewsService] NewsAPI request failed: {response.status_code}")
            return []

    except Exception as e:
        print(f"❌ [NewsService] Error fetching news evidence for '{claim[:30]}': {e}")
        return []
