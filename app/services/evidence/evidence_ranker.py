"""
TrustLens Evidence Ranker (Phase E)

Ranks and filters evidence sources by their reliability score before
sending them to the LLM verifier.

Rules:
  - Minimum trust score to pass: 0.60
  - Maximum sources passed to LLM: 5
  - Sort descending by trust score
"""

from typing import List, Dict, Any
from datetime import datetime, timezone, timedelta
import re

from app.services.evidence.source_reliability import (
    calculate_source_trust,
    calculate_source_trust_by_name,
    extract_domain,
)
from app.services.semantic_ranker import rank_articles_by_semantic_similarity

# Thresholds
MIN_TRUST_THRESHOLD = 0.60
MIN_SEMANTIC_SIMILARITY_THRESHOLD = 0.65
MAX_SOURCES_TO_LLM  = 5


def _score_evidence_item(item: Dict[str, Any]) -> float:
    """
    Returns a composite trust score for a single evidence item.
    Prefers URL-based scoring; falls back to source-name scoring.
    """
    url = item.get("url") or ""
    source_name = item.get("source") or item.get("publisher") or ""

    if url:
        url_score = calculate_source_trust(url)
        # If URL score is low (unknown domain), also try name-based
        if url_score < 0.45 and source_name:
            name_score = calculate_source_trust_by_name(source_name)
            return max(url_score, name_score)
        return url_score

    # No URL available — use name only
    return calculate_source_trust_by_name(source_name)


def _score_recency(published_at: str) -> float:
    """
    Computes a recency score based on the ISO publication date.
    """
    if not published_at:
        return 0.5  # Neutral if unknown

    try:
        # Handle "Z" suffix for UTC
        if published_at.endswith('Z'):
            published_at = published_at[:-1] + '+00:00'
        pub_dt = datetime.fromisoformat(published_at)
        if pub_dt.tzinfo is None:
            pub_dt = pub_dt.replace(tzinfo=timezone.utc)
            
        now = datetime.now(timezone.utc)
        age = now - pub_dt
        
        if age <= timedelta(hours=72):
            return 1.0  # highly recent
        elif age <= timedelta(days=7):
            return 0.8  # recent
        elif age <= timedelta(days=30):
            return 0.5  # standard
        else:
            return 0.2  # older
    except Exception:
        return 0.5


def rank_evidence_sources(claim: str, evidence_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Attaches a composite score (trust, semantic, recency) to each evidence item
    and returns the list sorted descending by the composite score.
    """
    # 1. Apply deep semantic embedding ranking
    semantically_augmented = rank_articles_by_semantic_similarity(claim, evidence_list)

    for item in semantically_augmented:
        trust_score = _score_evidence_item(item)
        semantic_score = item.get("semantic_similarity", 0.5)
        regen_score = _score_recency(item.get("publishedAt"))
        
        # Composite formula: 50% semantic, 30% trust, 20% recency
        composite = (0.50 * semantic_score) + (0.30 * trust_score) + (0.20 * regen_score)
        
        item["trustScore"] = round(trust_score, 4)
        item["semanticScore"] = round(semantic_score, 4)
        item["recencyScore"] = regen_score
        item["compositeScore"] = round(composite, 4)
        item["domain"] = extract_domain(item.get("url") or "") or item.get("source", "")

    return sorted(evidence_list, key=lambda x: x.get("compositeScore", 0), reverse=True)


def filter_evidence_sources(
    ranked_evidence: List[Dict[str, Any]],
    min_trust: float = MIN_TRUST_THRESHOLD,
    max_sources: int = MAX_SOURCES_TO_LLM,
) -> List[Dict[str, Any]]:
    """
    Filters ranked evidence to only include trusted sources and those within 30 days.
    """
    # Filter 1: Minimum trust
    trusted = [e for e in ranked_evidence if e.get("trustScore", 0) >= min_trust]
    
    # Filter 2: Max age 30 days (recency score >= 0.5)
    # Exceptions: Fact checks are allowed to be older (they are direct validations)
    age_filtered = []
    for e in trusted:
        if e.get("type") in ("factcheck", "wikipedia") or e.get("recencyScore", 0.5) >= 0.5:
            age_filtered.append(e)

    if not age_filtered:
        if trusted:
            # If all are too old, just take the best trusted one
            age_filtered = [trusted[0]]
        elif ranked_evidence:
            best = ranked_evidence[0]
            print(f"⚠️ [EvidenceRanker] No sources exceeded {min_trust} — using best available: "
                  f"{best.get('domain')} ({best.get('trustScore', 0)})")
            return [best]
        else:
            return []

    # Filter 3: Semantic Similarity Threshold (>= 0.65)
    # Applied to all sources to prevent contextual drift
    semantically_valid = []
    for e in age_filtered:
        if e.get("semantic_similarity", 0.0) >= MIN_SEMANTIC_SIMILARITY_THRESHOLD:
            semantically_valid.append(e)

    return semantically_valid[:max_sources]


def rank_and_filter(
    claim: str,
    evidence_list: List[Dict[str, Any]],
    min_trust: float = MIN_TRUST_THRESHOLD,
    max_sources: int = MAX_SOURCES_TO_LLM,
) -> List[Dict[str, Any]]:
    """
    Convenience wrapper: rank then filter in one call.
    """
    ranked = rank_evidence_sources(claim, evidence_list)
    filtered = filter_evidence_sources(ranked, min_trust=min_trust, max_sources=max_sources)

    print(f"📊 [EvidenceRanker] {len(evidence_list)} sources → "
          f"{len(ranked)} ranked → {len(filtered)} passed filter (≥{min_trust} trust, ≤30 days)")

    for item in filtered:
        print(f"   ✓ {item.get('domain'):30s} composite={item.get('compositeScore', 0):.2f} "
              f"(trust={item.get('trustScore', 0):.2f}, semantic={item.get('semanticScore', 0):.2f})")

    discarded = len(ranked) - len(filtered)
    if discarded > 0:
        print(f"   ✗ {discarded} source(s) discarded (below threshold or too old)")

    return filtered
