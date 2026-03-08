import asyncio
import logging
import re
from typing import Dict, Any, List

from app.services.fact_check_service import search_fact_check
from app.services.evidence.wikipedia_service import search_wikipedia
from app.services.evidence.news_service import search_news_articles
from app.services.evidence.evidence_ranker import rank_and_filter

from app.services.event_tuple_extractor import extract_event_tuple
from app.services.query_anchor_service import generate_anchored_queries

logger = logging.getLogger(__name__)

async def _aggregate_single_query(query: str) -> Dict[str, Any]:
    """
    Runs multi-source evidence retrieval concurrently for a SINGLE anchor query.
    Applies preliminary extraction, returning raw datasets. (Phase E ranking happens outside).
    """
    if not query:
        return {"factChecks": [], "wikipedia": None, "newsArticles": [], "ranked_evidence": []}

    print(f"📡 [EvidenceAggregator] Retrieval for query: '{query}'")

    # Execute all 3 sources concurrently
    factcheck_coro = search_fact_check(query)
    wiki_coro      = asyncio.to_thread(search_wikipedia, query)
    news_coro      = asyncio.to_thread(search_news_articles, query)

    results = await asyncio.gather(
        factcheck_coro, wiki_coro, news_coro,
        return_exceptions=True
    )

    fc_raw    = results[0] if not isinstance(results[0], Exception) else {"claims": []}
    wiki_data = results[1] if not isinstance(results[1], Exception) else None
    news_data = results[2] if not isinstance(results[2], Exception) else []

    if isinstance(results[0], Exception):
        print(f"⚠️ [EvidenceAggregator] Fact check retrieval failed: {results[0]}")
    if isinstance(results[1], Exception):
        print(f"⚠️ [EvidenceAggregator] Wikipedia retrieval failed: {results[1]}")
    if isinstance(results[2], Exception):
        print(f"⚠️ [EvidenceAggregator] News retrieval failed: {results[2]}")

    wiki_title = wiki_data["title"] if wiki_data else "None"
    news_count = len(news_data) if news_data else 0
    fc_count   = len(fc_raw.get("claims", []))
    print(f"✅ [EvidenceAggregator] Retrieved: Wikipedia={wiki_title}, News={news_count}, FactChecks={fc_count}")

    # ─── Phase E: Build flat source list for reliability ranking ────────────
    flat_sources = []

    for fc in fc_raw.get("claims", [])[:3]:
        reviews = fc.get("claimReview", [])
        if reviews:
            pub = reviews[0].get("publisher", {})
            flat_sources.append({
                "title":       fc.get("text", "Fact Check"),
                "source":      pub.get("name", "Fact Checker"),
                "url":         reviews[0].get("url", ""),
                "description": reviews[0].get("textualRating", ""),
                "type":        "factcheck",
            })

    if wiki_data:
        flat_sources.append({
            "title":       wiki_data.get("title", "Wikipedia"),
            "source":      "Wikipedia",
            "url":         wiki_data.get("url", "https://www.wikipedia.org"),
            "description": wiki_data.get("summary", ""),
            "type":        "wikipedia",
        })

    for article in (news_data or [])[:5]:
        flat_sources.append({
            "title":       article.get("title", ""),
            "source":      article.get("source", ""),
            "url":         article.get("url", ""),
            "description": article.get("description", ""),
            "type":        "news",
        })

    # Rank by trust score, filter those below 0.60
    # Note: re-ranking over combined results will happen in multi-query aggregator
    ranked_evidence = rank_and_filter(query, flat_sources)

    return {
        "factChecks":      fc_raw.get("claims", [])[:3],
        "wikipedia":       wiki_data,
        "newsArticles":    news_data[:5] if news_data else [],
        "ranked_evidence": ranked_evidence,   # ← trusted sources for LLM
    }


async def aggregate_evidence_multi_query(
    queries: List[str],
) -> Dict[str, Any]:
    """
    Run evidence retrieval for multiple normalized query strings CONCURRENTLY,
    then deduplicate and re-rank the merged pool.

    Used by the claim decomposition pipeline: each claim produces 2-3 query
    variants; this function fans them out and returns a single merged evidence
    dict that covers all query angles.

    Falls back to building and checking a single query if only one supplied.
    """
    if not queries:
        return {"factChecks": [], "wikipedia": None, "newsArticles": [], "ranked_evidence": []}

    unique_queries = list(dict.fromkeys(q.strip() for q in queries if q.strip()))

    if not unique_queries:
        return {"factChecks": [], "wikipedia": None, "newsArticles": [], "ranked_evidence": []}

    logger.info(
        "[MultiQuery] Fanning out evidence retrieval across %d queries: %s",
        len(unique_queries), unique_queries,
    )

    results = await asyncio.gather(
        *[_aggregate_single_query(q) for q in unique_queries],
        return_exceptions=True,
    )

    # --- Merge results, deduplicating by URL ---
    merged_fc:   list = []
    merged_wiki: dict | None = None
    merged_news: list = []
    merged_flat: list = []

    seen_urls: set[str] = set()

    def _dedup_url(item: dict) -> bool:
        url = item.get("url", "")
        if url in seen_urls:
            return False
        if url:
            seen_urls.add(url)
        return True

    for i, res in enumerate(results):
        if isinstance(res, Exception):
            logger.warning(
                "[MultiQuery] Evidence retrieval failed for query '%s': %s",
                unique_queries[i], res,
            )
            continue
        for fc in res.get("factChecks", []):
            url = (fc.get("claimReview") or [{}])[0].get("url", "")
            if url not in seen_urls:
                seen_urls.add(url)
                merged_fc.append(fc)
        if merged_wiki is None and res.get("wikipedia"):
            merged_wiki = res["wikipedia"]
        for article in res.get("newsArticles", []):
            if _dedup_url(article):
                merged_news.append(article)
        for src in res.get("ranked_evidence", []):
            url = src.get("url", "")
            if url not in seen_urls:
                seen_urls.add(url)
                merged_flat.append(src)

    # Re-rank the combined flat pool
    re_ranked = rank_and_filter(unique_queries[0], merged_flat)

    logger.info(
        "[MultiQuery] Merged: factChecks=%d wikipedia=%s news=%d ranked=%d",
        len(merged_fc), bool(merged_wiki), len(merged_news), len(re_ranked),
    )

    return {
        "factChecks":      merged_fc[:5],
        "wikipedia":       merged_wiki,
        "newsArticles":    merged_news[:10], # Keep a slightly larger pool from multi-queries
        "ranked_evidence": re_ranked,
    }


async def aggregate_evidence(claim: str) -> Dict[str, Any]:
    """
    Aggregate Evidence via Claim-Centered Query Anchoring.
    Extracts the core event tuple, generates strictly entity-anchored queries,
    and fans out evidence retrieval across all permutations, merging the results.
    """
    if not claim:
         return {"factChecks": [], "wikipedia": None, "newsArticles": [], "ranked_evidence": []}
         
    event_tuple = await extract_event_tuple(claim)
    anchored_queries = generate_anchored_queries(event_tuple)
    
    if not anchored_queries:
        # Absolute fallback if extraction completely fails
        anchored_queries = [claim]
        
    return await aggregate_evidence_multi_query(anchored_queries)
