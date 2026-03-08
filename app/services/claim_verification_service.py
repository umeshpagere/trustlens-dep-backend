"""
TrustLens Claim Verification Service

Verifies every extracted text claim against the multi-source evidence pipeline
(Fact Check, Wikipedia, News) using the existing evidence_aggregator and
evidence_verifier services. All claims are verified CONCURRENTLY.

Pipeline per claim
------------------
  claim (str)
    → aggregate_evidence()          # Fact Check + Wikipedia + News, concurrent
    → verify_claim_with_evidence()  # LLM judges evidence vs claim
    → VerifiedClaim dict

Aggregation
-----------
  knowledgeSupportScore = mean(claim["knowledgeSupportScore"])  ∈ [0.0, 1.0]
  If no claims: 0.0 (neutral; caller may substitute NEUTRAL_SCORES value)
"""

import asyncio
import logging
from typing import Any

from app.services.evidence.evidence_aggregator import (
    aggregate_evidence,
    aggregate_evidence_multi_query,
)
from app.services.evidence.evidence_verifier import verify_claim_with_evidence
from app.services.breaking_news_detector import detect_breaking_news
from app.services.breaking_news_service import (
    retrieve_real_time_news,
    filter_recent_articles,
    calculate_source_agreement,
    compute_breaking_news_confidence
)
from app.services.fact_check_service import search_fact_check_multi

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal: verify a single claim end-to-end
# ---------------------------------------------------------------------------

async def _verify_single_claim(claim: str) -> dict[str, Any]:
    """
    Retrieve evidence for *claim* and run LLM verification against it.

    Returns a VerifiedClaim dict:
      {
        "claim":               str,
        "verdict":             "supported" | "contradicted" | "uncertain",
        "knowledgeSupportScore": float,    # 0.0–1.0
        "reasoning":           str,
        "trustedSourcesUsed":  list[str],
        "evidenceSources":     list[str],
      }
    """
    logger.info("Verifying claim: %s", claim[:120])
    try:
        evidence = await aggregate_evidence(claim)
        logger.info(
            "Evidence sources retrieved for claim '%s...': factChecks=%d, wiki=%s, news=%d",
            claim[:60],
            len(evidence.get("factChecks", [])),
            bool(evidence.get("wikipedia")),
            len(evidence.get("newsArticles", [])),
        )

        result = await verify_claim_with_evidence(claim, evidence)
        logger.info(
            "Verification result for claim '%s...': verdict=%s score=%.3f",
            claim[:60],
            result.get("verdict"),
            result.get("knowledgeSupportScore", 0.5),
        )
        return {
            "claim":                 claim,
            "verdict":               result.get("verdict", "uncertain"),
            "knowledgeSupportScore": float(result.get("knowledgeSupportScore", 0.5)),
            "reasoning":             result.get("reasoning", ""),
            "trustedSourcesUsed":    result.get("trustedSourcesUsed", []),
            "evidenceSources":       result.get("evidenceSources", []),
        }

    except Exception as exc:
        logger.warning("Claim verification failed for '%s...': %s", claim[:60], exc)
        return {
            "claim":                 claim,
            "verdict":               "uncertain",
            "knowledgeSupportScore": 0.5,
            "reasoning":             "Verification failed due to an internal error.",
            "trustedSourcesUsed":    [],
            "evidenceSources":       [],
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def verify_all_claims(
    claims: list[str],
) -> tuple[list[dict[str, Any]], float]:
    """
    Verify every claim in *claims* concurrently using the multi-source
    evidence pipeline, then aggregate a single knowledgeSupportScore.

    Parameters
    ----------
    claims : list[str]
        Pre-validated, normalised claim strings (from claim_validator).

    Returns
    -------
    (verified_claims, knowledge_support_score)
        verified_claims          — list of VerifiedClaim dicts (one per input claim)
        knowledge_support_score  — mean knowledgeSupportScore in [0.0, 1.0]
                                   0.0 if *claims* is empty.

    Design notes
    ------------
    - asyncio.gather(return_exceptions=True) so a single failing claim does
      not abort the others.
    - Deduplication is done before calling this function (in the route).
    """
    if not claims:
        logger.info("verify_all_claims: no claims to verify — returning empty result")
        return [], 0.0

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_claims: list[str] = []
    for c in claims:
        key = c.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique_claims.append(c)

    logger.info("verify_all_claims: verifying %d unique claim(s) concurrently", len(unique_claims))

    tasks = [_verify_single_claim(claim) for claim in unique_claims]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    verified_claims: list[dict[str, Any]] = []
    scores: list[float] = []

    for claim, result in zip(unique_claims, raw_results):
        if isinstance(result, Exception):
            logger.warning("Unexpected exception during claim verification: %s", result)
            # Treat as uncertain rather than crashing
            result = {
                "claim":                 claim,
                "verdict":               "uncertain",
                "knowledgeSupportScore": 0.5,
                "reasoning":             "An unexpected error occurred during verification.",
                "trustedSourcesUsed":    [],
                "evidenceSources":       [],
            }
        verified_claims.append(result)
        scores.append(result["knowledgeSupportScore"])

    knowledge_support_score = round(sum(scores) / len(scores), 4) if scores else 0.0

    logger.info(
        "verify_all_claims complete: %d claims verified, "
        "knowledgeSupportScore=%.4f  verdicts=%s",
        len(verified_claims),
        knowledge_support_score,
        [v["verdict"] for v in verified_claims],
    )

    return verified_claims, knowledge_support_score


# ---------------------------------------------------------------------------
# Decomposed-claim path (uses normalized queries from decomposition pipeline)
# ---------------------------------------------------------------------------

async def _verify_single_claim_decomposed(decomposed: dict[str, Any]) -> dict[str, Any]:
    """
    Verify a single DecomposedClaim dict produced by claim_decomposition_service.

    Uses `normalizedQueries` for evidence retrieval (multi-query fan-out),
    giving broader, more relevant evidence than searching the raw claim text.
    """
    claim_text = decomposed.get("claim", "")
    queries    = decomposed.get("normalizedQueries") or [claim_text]

    logger.info(
        "Verifying decomposed claim '%s...' with %d normalized queries",
        claim_text[:60], len(queries),
    )

    try:
        # Pre-flight: Fact Check only to check for breaking news condition
        fc_results = await search_fact_check_multi(queries)
        fc_claims = fc_results.get("claims", [])

        is_breaking = detect_breaking_news(claim_text, fc_claims)
        
        if is_breaking:
            logger.info("⚡ [BreakingNews Fast-track] Activated for '%s...'", claim_text[:60])
            # Retrieve from whitelist ONLY, sorted by publishedAt
            news = await retrieve_real_time_news(queries)
            # Apply strictly <= 72h / 30day age filters
            filtered_news = filter_recent_articles(news)
            # Evaluate using purely NLP heuristic agreement scaling
            agreement = calculate_source_agreement(filtered_news, claim_text)
            conf_score = compute_breaking_news_confidence(agreement)
            
            logger.info(
                "⚡ [BreakingNews] %d fresh articles found, %d support claim, conf=%d",
                len(filtered_news), agreement["supporting_sources"], conf_score
            )
            
            verdict = "supported" if conf_score >= 50 else "uncertain"
            reasoning = "Multiple trusted news sources confirm the reported event." if verdict == "supported" else "Could not find sufficient real-time reporting from trusted sources."
            
            return {
                "claim":                 claim_text,
                "verdict":               verdict,
                "knowledgeSupportScore": float(conf_score / 100.0),
                "reasoning":             f"[Breaking News Detection] {reasoning}",
                "trustedSourcesUsed":    agreement["supporting_publishers"],
                "evidenceSources":       ["NewsAPI (Real-time Fast Track)"],
                "subject":               decomposed.get("subject",  ""),
                "action":                decomposed.get("action",   ""),
                "object":                decomposed.get("object",   ""),
                "context":               decomposed.get("context",  ""),
                "normalizedQueries":     queries,
                "isBreakingNews":        True
            }

        # Otherwise: standard pipeline.
        # aggregate_evidence_multi_query internally runs factCheck, wiki, and standard news
        evidence = await aggregate_evidence_multi_query(queries)
        logger.info(
            "Multi-query evidence for '%s...': factChecks=%d wiki=%s news=%d ranked=%d",
            claim_text[:60],
            len(evidence.get("factChecks", [])),
            bool(evidence.get("wikipedia")),
            len(evidence.get("newsArticles", [])),
            len(evidence.get("ranked_evidence", [])),
        )

        result = await verify_claim_with_evidence(claim_text, evidence)
        logger.info(
            "Decomposed claim result '%s...': verdict=%s score=%.3f",
            claim_text[:60],
            result.get("verdict"),
            result.get("knowledgeSupportScore", 0.5),
        )

        return {
            # Core fields
            "claim":                 claim_text,
            "verdict":               result.get("verdict", "uncertain"),
            "knowledgeSupportScore": float(result.get("knowledgeSupportScore", 0.5)),
            "reasoning":             result.get("reasoning", ""),
            "trustedSourcesUsed":    result.get("trustedSourcesUsed", []),
            "evidenceSources":       result.get("evidenceSources", []),
            # Structured decomposition fields (for rich API response)
            "subject":               decomposed.get("subject",  ""),
            "action":                decomposed.get("action",   ""),
            "object":                decomposed.get("object",   ""),
            "context":               decomposed.get("context",  ""),
            "normalizedQueries":     queries,
            "isBreakingNews":        False,
        }

    except Exception as exc:
        logger.warning(
            "Decomposed claim verification failed for '%s...': %s",
            claim_text[:60], exc,
        )
        return {
            "claim":                 claim_text,
            "verdict":               "uncertain",
            "knowledgeSupportScore": 0.5,
            "reasoning":             "Verification failed due to an internal error.",
            "trustedSourcesUsed":    [],
            "evidenceSources":       [],
            "subject":               decomposed.get("subject",  ""),
            "action":                decomposed.get("action",   ""),
            "object":                decomposed.get("object",   ""),
            "context":               decomposed.get("context",  ""),
            "normalizedQueries":     queries,
            "isBreakingNews":        False,
        }


async def verify_all_claims_decomposed(
    decomposed_claims: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float]:
    """
    Verify every DecomposedClaim concurrently using the multi-query evidence
    pipeline, then aggregate a single knowledgeSupportScore.

    Parameters
    ----------
    decomposed_claims : list[DecomposedClaim dicts]
        Output from claim_decomposition_service.decompose_claims().

    Returns
    -------
    (verified_claims, knowledge_support_score)
        verified_claims         — VerifiedClaim dicts enriched with decomposition fields
        knowledge_support_score — mean knowledgeSupportScore ∈ [0.0, 1.0]
    """
    if not decomposed_claims:
        logger.info("verify_all_claims_decomposed: no claims — returning empty")
        return [], 0.0

    logger.info(
        "verify_all_claims_decomposed: verifying %d claim(s) concurrently",
        len(decomposed_claims),
    )

    tasks = [_verify_single_claim_decomposed(dc) for dc in decomposed_claims]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    verified_claims: list[dict[str, Any]] = []
    scores: list[float] = []

    for dc, result in zip(decomposed_claims, raw_results):
        if isinstance(result, Exception):
            logger.warning(
                "Unexpected exception verifying decomposed claim '%s': %s",
                dc.get("claim", "")[:60], result,
            )
            result = {
                "claim":                 dc.get("claim", ""),
                "verdict":               "uncertain",
                "knowledgeSupportScore": 0.5,
                "reasoning":             "An unexpected error occurred during verification.",
                "trustedSourcesUsed":    [],
                "evidenceSources":       [],
                "subject":               dc.get("subject",  ""),
                "action":                dc.get("action",   ""),
                "object":                dc.get("object",   ""),
                "context":               dc.get("context",  ""),
                "normalizedQueries":     dc.get("normalizedQueries", []),
            }
        verified_claims.append(result)
        scores.append(result["knowledgeSupportScore"])

    knowledge_support_score = round(sum(scores) / len(scores), 4) if scores else 0.0

    logger.info(
        "verify_all_claims_decomposed complete: %d claims, "
        "knowledgeSupportScore=%.4f verdicts=%s",
        len(verified_claims),
        knowledge_support_score,
        [v["verdict"] for v in verified_claims],
    )

    return verified_claims, knowledge_support_score
