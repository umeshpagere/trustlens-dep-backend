"""
TrustLens Evidence-Based Credibility Engine

SCORING PHILOSOPHY (v2 — evidence-driven):
  The final credibility score is driven entirely by verifiable evidence signals.
  Heuristic proxies (source reputation, domain trust) have been removed from the
  scoring formula. Domain trust is still used downstream for evidence FILTERING
  (evidence_ranker.py) but must NOT influence credibilityScore.

NEW WEIGHTED FORMULA:
  factCheckScore        × 0.35  — primary: external fact-check result
  knowledgeSupportScore × 0.25  — Wikipedia / news evidence verification
  videoEvidenceScore    × 0.25  — video authenticity / transcript analysis
  semanticScore         × 0.10  — claim clarity / verifiability
  imageAuthenticityScore× 0.05  — media authenticity

Weights sum to 1.00.  Final score is clamped to [0, 95].

ASYNC ARCHITECTURE (Phase 6 — unchanged):
  Phase 1: LLM text analysis (must complete first — primaryClaim needed)
  Phase 2+3+4: fact-check, domain (filtering only), image — CONCURRENT
  Phase 5: synchronous scoring + confidence (pure CPU)

  return_exceptions=True means one failing service does not abort others.
"""

import asyncio
import logging
from typing import Any

from app.services.fact_check_service import search_fact_check, calculate_fact_check_score
from app.services.domain_reputation_service import evaluate_domain
from app.services.image_authenticity_service import evaluate_image
from app.services.confidence_service import calculate_confidence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Evidence-based weights (v2) — heuristic signals removed
# ---------------------------------------------------------------------------
WEIGHTS = {
    "factCheckScore":         0.35,  # Primary: external fact-check result
    "knowledgeSupportScore":  0.25,  # WikipediaI / news evidence
    "videoEvidenceScore":     0.25,  # Video authenticity / transcript analysis
    "semanticScore":          0.10,  # Claim clarity; NOT sentiment
    "imageAuthenticityScore": 0.05,  # Media authenticity
}

# Neutral baselines — represent "absence of a signal" rather than "presence of risk".
NEUTRAL_SCORES = {
    "factCheckScore":         65,   # No fact-check match found
    "knowledgeSupportScore":  60,   # No evidence retrieved
    "videoEvidenceScore":     65,   # No video to analyse
    "semanticScore":          60,   # Claim present but not assessed
    "imageAuthenticityScore": 75,   # No image to assess
}


# ---------------------------------------------------------------------------
# Neutral fallbacks — used when an async phase fails during gather
# ---------------------------------------------------------------------------

def _neutral_fact_check() -> dict[str, Any]:
    return {
        "factCheckScore": 65, "matchFound": False, "verdict": "No Match",
        "source": "", "referenceURL": "", "confidenceAdjustment": -0.1,
    }


def _neutral_domain() -> dict[str, Any]:
    """Domain still run for evidence filtering; never feeds credibilityScore."""
    return {
        "domainTrustScore": 65, "domain": None, "domainAgeDays": None,
        "httpsSecure": False, "isTrustedSource": False, "isBlacklisted": False,
        "riskFactors": ["Domain check unavailable"],
    }


def _neutral_image() -> dict[str, Any]:
    return {
        "imageAuthenticityScore": 75, "hashMatched": False, "matchedContext": None,
        "matchedEventDate": None, "contextMismatch": False, "aiGeneratedLikelihood": 0.0,
        "metadataPresent": False, "cameraMake": None, "cameraModel": None,
        "editingSoftwareDetected": False, "riskFactors": [],
    }


# ---------------------------------------------------------------------------
# Synchronous scoring (Phase 5 — CPU only, no I/O)
# ---------------------------------------------------------------------------

def calculate_credibility_score(scores: dict) -> float:
    """
    Pure function: compute the evidence-based credibility score.

    Parameters
    ----------
    scores : dict
        Expects keys: factCheckScore, knowledgeSupportScore, videoEvidenceScore,
        semanticScore, imageAuthenticityScore (all 0–100 floats).

    Returns
    -------
    float — final score in [0.0, 95.0], rounded to 2 dp.
    """
    fact_check    = max(0.0, min(100.0, float(scores.get("factCheckScore",        NEUTRAL_SCORES["factCheckScore"]))))
    knowledge     = max(0.0, min(100.0, float(scores.get("knowledgeSupportScore", NEUTRAL_SCORES["knowledgeSupportScore"]))))
    video         = max(0.0, min(100.0, float(scores.get("videoEvidenceScore",    NEUTRAL_SCORES["videoEvidenceScore"]))))
    semantic      = max(0.0, min(100.0, float(scores.get("semanticScore",         NEUTRAL_SCORES["semanticScore"]))))
    image_auth    = max(0.0, min(100.0, float(scores.get("imageAuthenticityScore",NEUTRAL_SCORES["imageAuthenticityScore"]))))

    final = (
        fact_check * WEIGHTS["factCheckScore"]
        + knowledge  * WEIGHTS["knowledgeSupportScore"]
        + video      * WEIGHTS["videoEvidenceScore"]
        + semantic   * WEIGHTS["semanticScore"]
        + image_auth * WEIGHTS["imageAuthenticityScore"]
    )

    logger.info(
        "Score composition — factCheck=%.1f knowledge=%.1f video=%.1f "
        "semantic=%.1f imageAuth=%.1f → final=%.2f",
        fact_check, knowledge, video, semantic, image_auth, final,
    )

    return round(max(0.0, min(95.0, final)), 2)


def compute_weighted_final_result(
    *,
    semantic_score: int | float | None = None,
    fact_check_details: dict[str, Any] | None = None,
    image_authenticity_score: int | float | None = None,
    domain_result: dict[str, Any] | None = None,
    image_auth_result: dict[str, Any] | None = None,
    manipulation_indicators: list | None = None,
    ai_video_probability: float | None = None,
    context_reuse_detected: bool = False,
    knowledge_support_score: float | None = None,
    video_evidence_score: float | None = None,
    breaking_news_detected: bool = False,
    breaking_news_confidence: float | None = None,
) -> dict[str, Any]:
    """
    Synchronous: compute final credibility score and confidence breakdown.

    Why synchronous: pure arithmetic — no I/O whatsoever.
    Called after all async phases have resolved.

    NOTE: sourceReputationScore and domainTrustScore are NO LONGER PARAMETERS.
    Domain metadata (domain_result) is still accepted so it can be passed
    through to evidence_flags (for confidence modelling) and domainReputation
    in the API response, but it does not influence credibilityScore.
    """

    def _safe(val: int | float | None, key: str) -> float:
        if val is None:
            return float(NEUTRAL_SCORES.get(key, 60))
        try:
            return max(0.0, min(100.0, float(val)))
        except (TypeError, ValueError):
            return float(NEUTRAL_SCORES.get(key, 60))

    fc = fact_check_details or {}
    # Use the actual fact-check score only when a match was found; otherwise neutral.
    fact_check_score = fc.get("factCheckScore") if fc.get("matchFound") else None
    if fact_check_score is None:
        fact_check_score = NEUTRAL_SCORES["factCheckScore"]

    component_scores = {
        "factCheckScore":         _safe(fact_check_score,        "factCheckScore"),
        "knowledgeSupportScore":  _safe(knowledge_support_score, "knowledgeSupportScore"),
        "videoEvidenceScore":     _safe(video_evidence_score,    "videoEvidenceScore"),
        "semanticScore":          _safe(semantic_score,          "semanticScore"),
        "imageAuthenticityScore": _safe(image_authenticity_score,"imageAuthenticityScore"),
    }

    if breaking_news_detected and breaking_news_confidence is not None:
        component_scores["breakingNewsConfidence"] = _safe(breaking_news_confidence, "knowledgeSupportScore")
        
        # Adjust weights: Fact checks are irrelevant for breaking news
        # Move factCheck weight (0.35) into breakingNewsConfidence
        bn_weight = WEIGHTS["factCheckScore"] + WEIGHTS["knowledgeSupportScore"]
        
        fact_check    = 0.0 # Ignored
        knowledge     = 0.0 # Ignored, replaced by breaking news
        bn_conf       = component_scores["breakingNewsConfidence"]
        video         = component_scores["videoEvidenceScore"]
        semantic      = component_scores["semanticScore"]
        image_auth    = component_scores["imageAuthenticityScore"]
        
        final = (
            bn_conf    * bn_weight
            + video      * WEIGHTS["videoEvidenceScore"]
            + semantic   * WEIGHTS["semanticScore"]
            + image_auth * WEIGHTS["imageAuthenticityScore"]
        )
        base_score = int(round(max(0.0, min(95.0, final))))
    else:
        base_score = int(round(calculate_credibility_score(component_scores)))

    # -----------------------------------------------------------------------
    # Penalty Layer — hard negative evidence (unchanged)
    # -----------------------------------------------------------------------
    _dr = domain_result or {}
    _ir = image_auth_result or {}

    if ai_video_probability is not None and ai_video_probability > 0.7:
        logger.warning("Phase B Penalty applied: AI video probability = %.2f", ai_video_probability)
        base_score = max(0, base_score - 40)

    if context_reuse_detected:
        logger.warning("Phase C Penalty applied: reused historical video context")
        base_score = max(0, base_score - 25)

    # -----------------------------------------------------------------------
    # Positive Boost Layer — reward verified evidence (domain removed)
    # -----------------------------------------------------------------------
    boost_applied = 0
    boost_reasons: list[str] = []

    eligible_for_boost = all(s >= 30 for s in component_scores.values())

    if eligible_for_boost:
        # 1. Verified fact-check TRUE
        if component_scores["factCheckScore"] >= 85 and fc.get("matchFound"):
            boost_applied += 10
            boost_reasons.append("Verified fact-check TRUE")
        # 2. Strong evidence support
        if component_scores["knowledgeSupportScore"] >= 75:
            boost_applied += 5
            boost_reasons.append("Evidence supports claim")
        # 3. Image authenticity confirmed
        if component_scores["imageAuthenticityScore"] >= 85:
            boost_applied += 5
            boost_reasons.append("Image authenticity confirmed")

    boost_applied = min(15, boost_applied)
    final_score = int(min(95, max(0, base_score + boost_applied)))

    # -----------------------------------------------------------------------
    # Phase 5: principled confidence (synchronous CPU-only)
    # Domain is kept in evidence_flags so confidence still benefits from it.
    # -----------------------------------------------------------------------
    evidence_flags = {
        "factCheckMatch":  bool(fc.get("matchFound")),
        "contextMismatch": bool(_ir.get("contextMismatch")),
        "imageReuseFound": bool(_ir.get("hashMatched")),
        "trustedDomain":   bool(_dr.get("isTrustedSource")),  # influences confidence only
    }
    confidence_result = calculate_confidence(component_scores, evidence_flags)

    def classify_score(score: int) -> tuple[str, str]:
        if score >= 85:   return "Minimal",    "Highly Reliable"
        elif score >= 70: return "Low",         "Reliable"
        elif score >= 50: return "Low-Medium",  "Likely Reliable"
        elif score >= 30: return "Medium",      "Questionable"
        else:             return "High",        "High Risk"

    risk_level, final_verdict = classify_score(final_score)

    logger.info(
        "Final credibility — score=%d verdict=%s riskLevel=%s boost=%d boostReasons=%s",
        final_score, final_verdict, risk_level, boost_applied, boost_reasons,
    )

    result_dict = {
        "componentScores":      component_scores,
        "factCheckDetails":     fc,
        "baseWeightedScore":    base_score,
        "positiveBoostApplied": boost_applied,
        "boostReasons":         boost_reasons,
        "finalScore":           final_score,
        "finalVerdict":         final_verdict,
        "riskLevel":            risk_level,
        "confidence":           confidence_result["confidenceScore"],
        "confidenceLevel":      confidence_result["confidenceLevel"],
        "confidenceBreakdown":  confidence_result,
    }
    
    if breaking_news_detected:
        result_dict["breakingNewsDetected"] = True
        
    return result_dict


# ---------------------------------------------------------------------------
# Async orchestrator — Phase 6 parallel execution
# ---------------------------------------------------------------------------

async def compute_full_credibility(
    text_analysis: dict[str, Any] | None,
    image_analysis: dict[str, Any] | None,
    video_analysis: dict[str, Any] | None = None,
    source_url: str | None = None,
    image_bytes: bytes | None = None,
) -> dict[str, Any]:
    """
    Async: Run the full Phase 6 credibility pipeline.

    Execution order:
      Phase 1: LLM text analysis (awaited in ROUTE before this is called)
      Phase 2+3+4: fact-check, domain (filtering only), image — CONCURRENT
      Phase 5: synchronous scoring + confidence

    Domain Phase (3) still runs for evidence filtering downstream but its
    score is NOT included in the credibility formula.
    """
    primary_claim = ""
    semantic_score: float | None = None
    manipulation_indicators: list | None = None
    knowledge_support_score: float | None = None
    breaking_news_detected = False
    breaking_news_confidence: float | None = None

    if text_analysis and text_analysis.get("status") != "skipped":
        semantic = text_analysis.get("semantic") or {}
        primary_claim = (semantic.get("primaryClaim") or "").strip()
        semantic_score = text_analysis.get("credibilityScore") or semantic.get("semanticScore")
        manipulation_indicators = semantic.get("manipulationIndicators", [])
        
        kv = text_analysis.get("knowledgeVerification", {})
        knowledge_support_score = kv.get("knowledgeSupportScore")
        
        breaking_news_detected = kv.get("breakingNewsDetected", False)
        if breaking_news_detected:
            # We stored breaking news confidence in knowledgeSupportScore as a 0-1 float
            # Let's extract it as a 0-100 float
            if knowledge_support_score is not None:
                breaking_news_confidence = knowledge_support_score * 100
    # ---- Extract video evidence score (new: dedicated signal, not blended into semantic) ----
    video_evidence_score: float | None = None
    ai_video_probability: float | None = None
    context_reuse_detected = False

    if video_analysis and video_analysis.get("status") != "skipped":
        raw_video_score = video_analysis.get("credibilityScore")
        if raw_video_score is not None:
            # Video score feeds videoEvidenceScore directly (25 % weight)
            video_evidence_score = float(raw_video_score)
            # Apply an internal pre-penalty for very-low scores to prevent
            # video result from being silently neutral.
            if video_evidence_score < 40:
                video_evidence_score = max(0.0, video_evidence_score - 10)

        ai_detection = video_analysis.get("aiDetection", {})
        ai_video_probability = ai_detection.get("aiGeneratedProbability")
        context_detection = video_analysis.get("contextDetection", {})
        context_reuse_detected = context_detection.get("contextReuseDetected", False)

        # Video-based knowledge support overrides text-based if present
        v_knowledge = video_analysis.get("knowledgeVerification", {}).get("knowledgeSupportScore")
        if v_knowledge is not None:
            knowledge_support_score = v_knowledge

    # ---- Extract image signals ----
    image_auth_score_override: float | None = None
    if image_analysis and image_analysis.get("status") != "skipped":
        img_score = image_analysis.get("credibilityScore")
        if img_score is not None:
            image_auth_score_override = float(img_score)
            # Blend semantic with image when image is the primary content
            if semantic_score is not None:
                semantic_score = (semantic_score * 0.2) + (image_auth_score_override * 0.8)
                if image_auth_score_override < 40:
                    semantic_score -= 10
            else:
                semantic_score = image_auth_score_override
                if image_auth_score_override < 40:
                    semantic_score -= 10

    # ---- Phase 2 + 3 + 4: concurrent I/O ----
    fc_raw, domain_result, image_auth_result = await asyncio.gather(
        search_fact_check(primary_claim) if primary_claim else _coro_empty_fact(),
        evaluate_domain(source_url),
        asyncio.to_thread(evaluate_image, image_bytes, primary_claim or None),
        return_exceptions=True,
    )

    if isinstance(fc_raw, Exception):
        logger.warning("Fact-check phase failed: %s", fc_raw)
        fc_raw = {"claims": []}
    if isinstance(domain_result, Exception):
        logger.warning("Domain phase failed: %s", domain_result)
        domain_result = _neutral_domain()
    if isinstance(image_auth_result, Exception):
        logger.warning("Image phase failed: %s", image_auth_result)
        image_auth_result = _neutral_image()

    fact_check_details     = calculate_fact_check_score(fc_raw)
    image_authenticity_score = image_auth_result["imageAuthenticityScore"]

    # Override image auth score from LLM image analysis if available
    if image_auth_score_override is not None:
        image_authenticity_score = image_auth_score_override

    # Normalise knowledge support (0-1 → 0-100)
    ks_normalised = (
        knowledge_support_score * 100
        if knowledge_support_score is not None and knowledge_support_score <= 1.0
        else knowledge_support_score
    )

    # ---- Phase 5: synchronous scoring + confidence ----
    weighted_result = compute_weighted_final_result(
        semantic_score=semantic_score,
        fact_check_details=fact_check_details,
        image_authenticity_score=image_authenticity_score,
        domain_result=domain_result,
        image_auth_result=image_auth_result,
        manipulation_indicators=manipulation_indicators,
        ai_video_probability=ai_video_probability,
        context_reuse_detected=context_reuse_detected,
        knowledge_support_score=ks_normalised,
        video_evidence_score=video_evidence_score,
        breaking_news_detected=breaking_news_detected,
        breaking_news_confidence=breaking_news_confidence,
    )

    # Domain reputation is returned for informational / UI purposes only.
    # It DOES NOT influence credibilityScore.
    weighted_result["domainReputation"]  = domain_result
    weighted_result["imageAuthenticity"] = image_auth_result
    return weighted_result


async def _coro_empty_fact() -> dict:
    """Async no-op yielding an empty fact-check response (no API call made)."""
    return {"claims": [], "nextPageToken": ""}
