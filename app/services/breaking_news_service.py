import logging
import httpx
from typing import List, Dict, Any
from datetime import datetime, timezone, timedelta

from app.config.settings import Config
import re

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 8.0

# 1. Trusted Whitelist for Breaking News
TRUSTED_NEWS_SOURCES = [
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "bbc.co.uk",
    "theguardian.com",
    "aljazeera.com",
    "nytimes.com",
    "washingtonpost.com"
]


async def retrieve_real_time_news(queries: List[str]) -> List[Dict[str, Any]]:
    """
    Retrieve recent articles from trusted news publishers.
    Uses NewsAPI and restricts domains to the whitelist.
    
    Tries queries in order until enough results are found.
    """
    api_key = getattr(Config, "NEWS_API_KEY", None) or ""
    if not api_key:
        logger.warning("[BreakingNewsService] NEWS_API_KEY missing. Skipping.")
        return []

    if not queries:
        return []

    articles_out = []
    seen_urls = set()

    for query in queries:
        try:
            params = {
                "q": query,
                "domains": ",".join(TRUSTED_NEWS_SOURCES),
                "sortBy": "publishedAt",  # Crucial for breaking news: sort by freshest
                "pageSize": 10,
                "language": "en",
                "apiKey": api_key
            }

            # httpx.AsyncClient for true async performance
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                response = await client.get("https://newsapi.org/v2/everything", params=params)

            if response.status_code == 200:
                articles = response.json().get("articles", [])
                for a in articles:
                    url = a.get("url")
                    if url and url not in seen_urls and a.get("title") != "[Removed]":
                        seen_urls.add(url)
                        articles_out.append({
                            "title": a.get("title"),
                            "source": a.get("source", {}).get("name", "Unknown Publisher"),
                            "url": url,
                            "description": a.get("description") or a.get("content", ""),
                            "publishedAt": a.get("publishedAt")
                        })
            
            # If we have a good batch, no need to issue more expensive queries
            if len(articles_out) >= 10:
                break

        except Exception as e:
            logger.error(f"[BreakingNewsService] Error fetching news for '{query}': {e}")
            
    return articles_out


def filter_recent_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Prefer articles from the last 72 hours.
    Strictly discard articles older than 30 days.
    """
    recent_articles = []
    now = datetime.now(timezone.utc)
    
    for article in articles:
        pub_str = article.get("publishedAt")
        if not pub_str:
            continue
            
        try:
            if pub_str.endswith('Z'):
                pub_str = pub_str[:-1] + '+00:00'
            pub_dt = datetime.fromisoformat(pub_str)
            if pub_dt.tzinfo is None:
                pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                
            age = now - pub_dt
            
            # Keep if <= 72 hours (highly preferred) or at least <= 30 days
            if age <= timedelta(days=30):
                # Attach age_hours for downstream logic if needed
                article["age_hours"] = age.total_seconds() / 3600.0
                recent_articles.append(article)
                
        except Exception:
            pass  # Bad date format, discard
            
    # Sort by freshest first
    recent_articles.sort(key=lambda x: x.get("age_hours", 9999))
    return recent_articles


def _score_relevance(claim: str, item: Dict[str, Any]) -> float:
    """
    Computes a keyword-overlap relevance score between the claim and the
    evidence item's title/description. Used for basic source agreement matching
    without relying on the heavy semantic transformer layer.
    """
    if not claim:
        return 0.5

    claim_words = set(re.sub(r"[^\w\s]", "", claim.lower()).split())
    if not claim_words:
        return 0.5

    text = f"{item.get('title', '')} {item.get('description', '')}".lower()
    text_words = set(re.sub(r"[^\w\s]", "", text).split())

    if not text_words:
        return 0.1

    overlap = claim_words.intersection(text_words)
    score = len(overlap) / len(claim_words)
    return round(score, 4)


def calculate_source_agreement(articles: List[Dict[str, Any]], claim: str) -> Dict[str, Any]:
    """
    Check whether multiple independent sources report the same event.
    For breaking news without LLM verifier overhead, we use similarity signals.
    """
    supporting_sources = 0
    contradicting_sources = 0
    supporting_publishers = set()
    
    for article in articles:
        # We reuse the keyword overlap scoring from EvidenceRanker
        similarity = _score_relevance(claim, article)
        
        # Determine strict support based on headline/body overlap
        # Since these are curated whitelist sources, high overlap means reporting the event
        if similarity >= 0.45:
            supporting_sources += 1
            supporting_publishers.add(article.get("source"))
        # (Contradicting is hard to detect via pure keywords without an LLM, 
        # so for this purely algorithmic tier, we focus on presence of support)
        
    return {
        "supporting_sources": supporting_sources,
        "contradicting_sources": contradicting_sources,
        "supporting_publishers": list(supporting_publishers)
    }


def compute_breaking_news_confidence(agreement_result: Dict[str, Any]) -> int:
    """
    Convert source agreement into a 0-100 confidence score.
    """
    supporters = agreement_result.get("supporting_sources", 0)
    
    if supporters >= 3:
        return 90
    elif supporters == 2:
        return 70
    elif supporters == 1:
        return 50
    else:
        return 20
