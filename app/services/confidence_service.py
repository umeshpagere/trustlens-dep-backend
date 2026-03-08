"""
TrustLens Phase 5: Confidence Modeling Service

Calculates HOW CERTAIN the system is about its credibility assessment.

Critical distinction (must never be conflated):
  Credibility score : how trustworthy the content appears  (0–100 int)
  Confidence score  : how certain TrustLens is about that verdict  (0.0–0.95 float)

A piece of content can have HIGH credibility BUT LOW confidence (e.g., we only
received an image with no URL, no fact-check match, and short text — we think
it looks fine, but we barely have any evidence either way).

Three-factor model
-------------------
Confidence is computed from three independent, mathematically grounded factors:

  1. COVERAGE (weight 0.40)
     How many of the possible analysis signals actually contributed real data?
     More signals = more information = more certain.
     Why 0.40 weight: coverage is the most fundamental driver of certainty.
     You cannot be confident about something you barely measured.

  2. AGREEMENT (weight 0.30)
     Do the signals agree with each other?
     Low standard deviation across component scores = signals point the same way.
     High variance = conflicting evidence = lower confidence.
     Why 0.30 weight: agreement is necessary but secondary. All signals could
     agree and still only cover 25% of the pipeline.

  3. EVIDENCE STRENGTH (weight 0.30)
     Did any verified external databases return confirmed matches?
     Fact-check database matches, known image hash hits, trusted domain
     confirmations are DETERMINISTIC signals — not estimates. They add certainty.
     Why 0.30 weight: strong evidence is highly meaningful but can only be
     present when specific conditions are met. It must not dominate when absent.

Why confidence is capped at 0.95:
  No automated system has perfect information. Epistemic humility is a design
  requirement: consumers must never treat TrustLens as an oracle. Capping at
  0.95 signals that there is always 5% irreducible uncertainty.

Why we never use raw LLM confidence directly:
  Azure OpenAI returns a `confidenceScore` in its JSON response. That value
  reflects the *model's internal certainty*, not the *pipeline's certainty*.
  A model can be highly confident about a hallucination. The three-factor model
  is grounded in signal coverage and external database confirmations — independent
  of any single model's self-reported certainty.

Parallelism readiness (Phase 6):
  calculate_confidence() is pure CPU math — no I/O, no blocking, no external calls.
  It runs in microseconds. It is safe to call after asyncio.gather() returns all
  component results. No changes to this function are required for async conversion.

No global state. No randomness. Deterministic for any given input set.
"""

import math
from typing import Any

# ---------------------------------------------------------------------------
# Score thresholds for signal "presence"
# A signal is considered PRESENT if its score differs from its neutral default
# by more than this margin. Neutral-at-default scores contribute to coverage
# denominator but not numerator (they add signal slots without real evidence).
# ---------------------------------------------------------------------------
_NEUTRAL_DEFAULTS: dict[str, float] = {
    "factCheckScore":         50.0,   # neutral when no fact-check match
    "knowledgeSupportScore":  50.0,   # neutral when no evidence retrieved
    "videoEvidenceScore":     50.0,   # neutral when no video present
    "semanticScore":          50.0,   # LLM always runs → always present
    "imageAuthenticityScore": 70.0,   # neutral when no image bytes
}

# A signal is "active" (carrying real data) if its value differs from neutral
# by more than this threshold. Small jitter is treated as neutral noise.
_PRESENCE_MARGIN: float = 3.0

# Evidence bonuses
_EVIDENCE_BONUS_FACT_CHECK_MATCH: float = 0.15
_EVIDENCE_BONUS_CONTEXT_MISMATCH: float = 0.15
_EVIDENCE_BONUS_IMAGE_REUSE:      float = 0.10
_EVIDENCE_BONUS_TRUSTED_DOMAIN:   float = 0.10
_EVIDENCE_CAP:                    float = 0.50   # caps total evidence_strength

# Formula weights (must sum to 1.0)
_WEIGHT_COVERAGE:  float = 0.40
_WEIGHT_AGREEMENT: float = 0.30
_WEIGHT_EVIDENCE:  float = 0.30

# Output cap — never claim certainty. 0.95 signals epistemic humility.
_CONFIDENCE_MAX: float = 0.95
_CONFIDENCE_MIN: float = 0.00


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi] inclusive. Handles NaN safely."""
    if not isinstance(value, (int, float)) or math.isnan(value):
        return lo
    return max(lo, min(hi, float(value)))


def _safe_score(val: Any) -> float | None:
    """Convert to float in [0, 100] or return None for invalid input."""
    if val is None:
        return None
    try:
        v = float(val)
        if math.isnan(v) or math.isinf(v):
            return None
        return _clamp(v, 0.0, 100.0)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Factor 1 — Signal Coverage
# ---------------------------------------------------------------------------

def _calculate_coverage(component_scores: dict[str, Any]) -> float:
    """
    Compute the fraction of signals that contributed real (non-neutral) data.

    Why coverage drives confidence:
      If only semantic analysis ran (e.g., no image, no URL, no fact-check
      match) we have a 1/5 coverage ratio. Even if the semantic analysis is
      highly confident, the system has barely examined the claim.

    Why semantic is always counted as present:
      The LLM semantic analysis always runs and always produces a score that
      differs from neutral (50). It is the guaranteed baseline signal.

    Returns a float in [0, 1].
    """
    if not component_scores:
        return 0.0
    # Always derived from the canonical 5-signal set
    total_signals = len(_NEUTRAL_DEFAULTS)
    if total_signals == 0:
        return 0.0

    present = 0
    for key, neutral in _NEUTRAL_DEFAULTS.items():
        raw = component_scores.get(key)
        score = _safe_score(raw)
        if score is not None and abs(score - neutral) > _PRESENCE_MARGIN:
            present += 1

    return _clamp(present / total_signals, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Factor 2 — Signal Agreement
# ---------------------------------------------------------------------------

def _calculate_agreement(component_scores: dict[str, Any]) -> float:
    """
    Compute agreement between signals using normalised standard deviation.

    Formula:
        std_dev  = stdev(all_valid_scores)
        agreement = 1 - (std_dev / 50)
        agreement = clamp(agreement, 0, 1)

    Why low variance == higher agreement:
      std_dev = 0 means every signal returned the same score → perfect agreement.
      std_dev = 50 (the theoretical max for scores in [0,100]) means signals
      are completely polarised (e.g. 0 and 100) → no agreement.

    Why normalise by 50:
      For values in [0, 100], the maximum possible population standard deviation
      of a binary distribution (all values at 0 or 100) is exactly 50.
      Dividing by 50 maps [0, 50] → [0, 1] cleanly.

    Why conflicting signals reduce confidence:
      If semantic says 85 (very credible) and factCheck says 10 (known false),
      the system genuinely doesn't know which to believe more. This uncertainty
      about uncertainty deserves a lower confidence score.

    Edge cases:
      - 0 or 1 scores: std_dev is undefined → return 1.0 (perfect agreement
        assumed — not enough data to disagree).
    """
    scores: list[float] = []
    for val in component_scores.values():
        s = _safe_score(val)
        if s is not None:
            scores.append(s)

    if len(scores) < 2:
        # Cannot compute standard deviation — assume perfect agreement
        return 1.0

    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    std_dev = math.sqrt(variance)

    agreement = 1.0 - (std_dev / 50.0)
    return _clamp(agreement, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Factor 3 — Evidence Strength
# ---------------------------------------------------------------------------

def _calculate_evidence_strength(evidence_flags: dict[str, Any]) -> float:
    """
    Compute evidence strength from verified, deterministic signals.

    Why verified database matches add certainty (over probabilistic signals):
      A fact-check database confirmation is an objective, human-verified record
      — not a model estimate. Similarly, a pHash match against a known image
      registry is a deterministic lookup. These are as close to ground truth
      as an automated system can get.

    Why context mismatch earns the largest bonus:
      It's the combination of two confirmations: the image IS the same
      (pHash match) AND the context has changed. This composite evidence is
      the clearest automated signal of intentional misinformation.

    Why the total is capped at 0.50:
      Evidence is supplementary — it shouldn't dominate. Even if all four
      flags fire simultaneously (max = 0.50 before cap), the other two
      factors still contribute 70% of the total confidence.

    Returns a float in [0, _EVIDENCE_CAP].
    """
    flags = evidence_flags or {}

    strength = 0.0

    if flags.get("factCheckMatch"):
        # A fact-check database returned a match for this specific claim.
        # This is the strongest single signal: a human-curated verdict exists.
        strength += _EVIDENCE_BONUS_FACT_CHECK_MATCH

    if flags.get("contextMismatch"):
        # Image reuse detected + different event context. Classic misinformation.
        strength += _EVIDENCE_BONUS_CONTEXT_MISMATCH

    if flags.get("imageReuseFound"):
        # pHash matched a known image. Reuse is confirmed (mismatch may or may not be).
        strength += _EVIDENCE_BONUS_IMAGE_REUSE

    if flags.get("trustedDomain"):
        # The source domain is on a manually-curated trusted list.
        # Lends credibility to the source, not the content — secondary signal.
        strength += _EVIDENCE_BONUS_TRUSTED_DOMAIN

    return _clamp(strength, 0.0, _EVIDENCE_CAP)


# ---------------------------------------------------------------------------
# Confidence label
# ---------------------------------------------------------------------------

def _confidence_level(score: float) -> str:
    """
    Map a numeric confidence score to a human-readable label.

    Why labels matter for UX:
      A confidence of 0.74 is meaningless to most users.
      "Moderate confidence" is immediately actionable.
      Both are returned: numeric for machine consumers, label for UI.
    """
    if score >= 0.80:
        return "High"
    elif score >= 0.60:
        return "Moderate"
    elif score >= 0.40:
        return "Low"
    else:
        return "Very Low"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_confidence(
    component_scores: dict[str, Any] | None,
    evidence_flags: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Calculate system confidence in the credibility assessment.

    Parameters
    ----------
    component_scores : dict | None
        Dictionary of analysis component scores, keyed as:
          factCheckScore, knowledgeSupportScore, videoEvidenceScore,
          semanticScore, imageAuthenticityScore
        Missing keys are treated as neutral (not present).
        NOTE: sourceReputationScore and domainTrustScore are no longer
        included in scoring — domain is used only for evidence filtering.
    evidence_flags : dict | None
        Binary confirmed-evidence flags:
          factCheckMatch, contextMismatch, imageReuseFound, trustedDomain
        Missing/None treated as all-False.

    Returns
    -------
    dict with keys:
        confidenceScore      : float  — 0.0–0.95 (never exactly 1.0)
        confidenceLevel      : str    — "High" | "Moderate" | "Low" | "Very Low"
        coverageScore        : float  — fraction of signals with real data
        agreementScore       : float  — normalised signal coherence
        evidenceStrengthScore: float  — bonus from verified database hits

    Graceful failure:
        Any unexpected error returns a safe minimum result (0.40 / "Low")
        rather than propagating exceptions.
    """
    safe_default: dict[str, Any] = {
        "confidenceScore":       0.40,
        "confidenceLevel":       "Low",
        "coverageScore":         0.0,
        "agreementScore":        0.0,
        "evidenceStrengthScore": 0.0,
    }

    try:
        scores = component_scores or {}
        flags  = evidence_flags  or {}

        coverage  = _calculate_coverage(scores)
        agreement = _calculate_agreement(scores)
        evidence  = _calculate_evidence_strength(flags)

        raw_confidence = (
            coverage  * _WEIGHT_COVERAGE  +
            agreement * _WEIGHT_AGREEMENT +
            evidence  * _WEIGHT_EVIDENCE
        )
        confidence = _clamp(raw_confidence, _CONFIDENCE_MIN, _CONFIDENCE_MAX)

        return {
            "confidenceScore":        round(confidence, 4),
            "confidenceLevel":        _confidence_level(confidence),
            "coverageScore":          round(coverage,   4),
            "agreementScore":         round(agreement,  4),
            "evidenceStrengthScore":  round(evidence,   4),
        }

    except Exception as exc:
        print(f"⚠️ Confidence model error: {exc}")
        return safe_default
