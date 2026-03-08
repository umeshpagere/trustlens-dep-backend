"""
TrustLens Credibility Engine — Unit Tests

Tests the new evidence-based scoring formula introduced in v2:
  factCheckScore        × 0.35
  knowledgeSupportScore × 0.25
  videoEvidenceScore    × 0.25
  semanticScore         × 0.10
  imageAuthenticityScore× 0.05

All tests are pure CPU — no network, no mocking required.

Run:
    cd /Users/umeshpagere/Downloads/trustlens-2-main/backend
    python3 -m pytest tests/test_credibility_engine.py -v
"""

import unittest

from app.services.credibility_engine import (
    calculate_credibility_score,
    compute_weighted_final_result,
    WEIGHTS,
    NEUTRAL_SCORES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _neutral_scores() -> dict:
    """Neutral baseline — all signals at their 'no-data' defaults."""
    return dict(NEUTRAL_SCORES)


def _strong_evidence_scores() -> dict:
    return {
        "factCheckScore":         90.0,
        "knowledgeSupportScore":  85.0,
        "videoEvidenceScore":     75.0,
        "semanticScore":          80.0,
        "imageAuthenticityScore": 90.0,
    }


def _contradiction_scores() -> dict:
    """Simulates a fact-check contradiction with strong negative signals."""
    return {
        "factCheckScore":         10.0,   # Known false
        "knowledgeSupportScore":  30.0,   # Contradicted by evidence
        "videoEvidenceScore":     65.0,   # Neutral (no video)
        "semanticScore":          50.0,
        "imageAuthenticityScore": 75.0,
    }


def _media_evidence_contradiction() -> dict:
    """Simulates strong media evidence contradicting the claim."""
    return {
        "factCheckScore":         65.0,   # No fact-check match
        "knowledgeSupportScore":  65.0,   # Neutral
        "videoEvidenceScore":     5.0,    # AI-generated / deeply suspicious
        "semanticScore":          50.0,
        "imageAuthenticityScore": 30.0,   # Image authenticity failure
    }


# ---------------------------------------------------------------------------
# calculate_credibility_score — pure function tests
# ---------------------------------------------------------------------------

class TestCalculateCredibilityScore(unittest.TestCase):

    def test_weights_sum_to_one(self):
        """All weights must sum to exactly 1.0 (within float tolerance)."""
        self.assertAlmostEqual(sum(WEIGHTS.values()), 1.0, places=10)

    def test_all_zero_scores_gives_zero(self):
        scores = {k: 0.0 for k in WEIGHTS}
        result = calculate_credibility_score(scores)
        self.assertEqual(result, 0.0)

    def test_all_100_scores_gives_95(self):
        """Perfect scores produce the maximum allowed output (95)."""
        scores = {k: 100.0 for k in WEIGHTS}
        result = calculate_credibility_score(scores)
        self.assertEqual(result, 95.0)

    def test_neutral_scores_produce_reasonable_baseline(self):
        """Neutral-at-default inputs should produce a mid-range score."""
        result = calculate_credibility_score(_neutral_scores())
        self.assertGreater(result, 40.0)
        self.assertLess(result, 80.0)

    def test_result_is_float(self):
        self.assertIsInstance(calculate_credibility_score(_neutral_scores()), float)

    def test_score_clamped_above_zero(self):
        scores = {k: -999.0 for k in WEIGHTS}
        self.assertGreaterEqual(calculate_credibility_score(scores), 0.0)

    def test_missing_keys_use_neutral_defaults(self):
        """Missing keys should fall back to NEUTRAL_SCORES, not crash."""
        result = calculate_credibility_score({})
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result, 0.0)

    def test_formula_weights_correct(self):
        """Manual calculation must match the function for known inputs."""
        scores = {
            "factCheckScore":         80.0,
            "knowledgeSupportScore":  60.0,
            "videoEvidenceScore":     50.0,
            "semanticScore":          70.0,
            "imageAuthenticityScore": 90.0,
        }
        expected = round(
            80.0 * 0.35 + 60.0 * 0.25 + 50.0 * 0.25 + 70.0 * 0.10 + 90.0 * 0.05,
            2,
        )
        self.assertAlmostEqual(calculate_credibility_score(scores), expected, places=2)

    # ---- Deprecated signals must be ignored ----

    def test_source_reputation_score_is_ignored(self):
        """sourceReputationScore must not influence the result."""
        base = calculate_credibility_score(_neutral_scores())
        with_rep = _neutral_scores()
        with_rep["sourceReputationScore"] = 0.0   # worst possible — should not matter
        self.assertAlmostEqual(base, calculate_credibility_score(with_rep), places=2)

    def test_domain_trust_score_is_ignored(self):
        """domainTrustScore must not influence the result."""
        base = calculate_credibility_score(_neutral_scores())
        with_domain = _neutral_scores()
        with_domain["domainTrustScore"] = 0.0   # worst possible — should not matter
        self.assertAlmostEqual(base, calculate_credibility_score(with_domain), places=2)


# ---------------------------------------------------------------------------
# Scenario tests (realistic end-to-end via compute_weighted_final_result)
# ---------------------------------------------------------------------------

class TestScoringScenarios(unittest.TestCase):
    """
    The three canonical test scenarios from the implementation plan.
    """

    def _run(self, component_scores: dict, **kwargs) -> dict:
        """Helper: build a minimal fact_check_details and run scoring."""
        # Inject component scores via the matching keyword arguments
        return compute_weighted_final_result(
            semantic_score=component_scores.get("semanticScore"),
            fact_check_details={
                "factCheckScore": component_scores.get("factCheckScore", 65),
                "matchFound": component_scores.get("factCheckScore", 65) != 65,
                "verdict": "False" if component_scores.get("factCheckScore", 65) < 40 else "No Match",
            },
            knowledge_support_score=component_scores.get("knowledgeSupportScore"),
            image_authenticity_score=component_scores.get("imageAuthenticityScore"),
            video_evidence_score=component_scores.get("videoEvidenceScore"),
            **kwargs,
        )

    # ---- Case 1: Strong fact-check contradiction -------------------------

    def test_case1_strong_factcheck_contradiction_gives_low_score(self):
        """
        Case 1: factCheckScore = 10 (known false), low evidence support.
        Expected: finalScore <= 40 (Questionable or High Risk zone).
        """
        result = self._run(_contradiction_scores())
        score = result["finalScore"]
        self.assertLessEqual(score, 40,
            msg=f"Strong fact-check contradiction should give score <= 40, got {score}")

    def test_case1_verdict_is_high_risk_or_questionable(self):
        result = self._run(_contradiction_scores())
        self.assertIn(result["finalVerdict"], ["High Risk", "Questionable"])

    # ---- Case 2: Strong media evidence contradiction ----------------------

    def test_case2_media_evidence_contradiction_lowers_score(self):
        """
        Case 2: videoEvidenceScore = 5 (near-zero), imageAuthenticityScore = 30.
        Expected: finalScore <= 50 (clearly Questionable, FactCheck at neutral).
        """
        result = self._run(_media_evidence_contradiction())
        score = result["finalScore"]
        self.assertLessEqual(score, 50,
            msg=f"Strong media contradiction should give score <= 50, got {score}")

    def test_case2_ai_video_penalty_applied(self):
        """AI video probability > 0.7 → flat -40 penalty."""
        result = compute_weighted_final_result(
            semantic_score=60,
            fact_check_details={"matchFound": False},
            video_evidence_score=65,
            ai_video_probability=0.85,  # above threshold
        )
        self.assertLess(result["finalScore"], 40)

    def test_case2_context_reuse_penalty_applied(self):
        """Context reuse detected → -25 penalty."""
        result_no_reuse = compute_weighted_final_result(
            semantic_score=60,
            fact_check_details={"matchFound": False},
            context_reuse_detected=False,
        )
        result_reuse = compute_weighted_final_result(
            semantic_score=60,
            fact_check_details={"matchFound": False},
            context_reuse_detected=True,
        )
        self.assertLess(result_reuse["finalScore"], result_no_reuse["finalScore"])

    # ---- Case 3: Verified claim with strong evidence ----------------------

    def test_case3_verified_claim_strong_evidence_gives_high_score(self):
        """
        Case 3: factCheckScore = 90, knowledgeSupportScore = 85 (verified).
        Expected: finalScore > 70.
        """
        result = self._run(_strong_evidence_scores())
        score = result["finalScore"]
        self.assertGreater(score, 70,
            msg=f"Verified claim with strong evidence should give score > 70, got {score}")

    def test_case3_verdict_positive_range(self):
        result = self._run(_strong_evidence_scores())
        self.assertIn(result["finalVerdict"],
                      ["Highly Reliable", "Reliable", "Likely Reliable"])

    def test_case3_boost_applied_for_verified_factcheck(self):
        """Verified fact-check match at ≥ 85 should trigger the boost."""
        result = compute_weighted_final_result(
            semantic_score=80,
            fact_check_details={"factCheckScore": 90, "matchFound": True, "verdict": "True"},
            knowledge_support_score=85,
            image_authenticity_score=90,
            video_evidence_score=75,
        )
        self.assertGreater(result["positiveBoostApplied"], 0)

    # ---- Deprecated fields must not appear in componentScores ------------

    def test_no_source_reputation_in_component_scores(self):
        result = self._run(_neutral_scores())
        self.assertNotIn("sourceReputationScore", result["componentScores"])

    def test_no_domain_trust_in_component_scores(self):
        result = self._run(_neutral_scores())
        self.assertNotIn("domainTrustScore", result["componentScores"])

    # ---- All five expected signals must be present -----------------------

    def test_all_five_evidence_signals_present(self):
        result = self._run(_strong_evidence_scores())
        for key in ["factCheckScore", "knowledgeSupportScore", "videoEvidenceScore",
                    "semanticScore", "imageAuthenticityScore"]:
            self.assertIn(key, result["componentScores"],
                          msg=f"Expected signal '{key}' missing from componentScores")

    # ---- Result structure ------------------------------------------------

    def test_result_has_required_keys(self):
        result = self._run(_neutral_scores())
        for key in ["finalScore", "finalVerdict", "riskLevel", "componentScores",
                    "confidence", "confidenceLevel", "baseWeightedScore"]:
            self.assertIn(key, result)

    def test_final_score_in_valid_range(self):
        for scores in [_neutral_scores(), _contradiction_scores(),
                       _strong_evidence_scores(), _media_evidence_contradiction()]:
            result = self._run(scores)
            self.assertGreaterEqual(result["finalScore"], 0)
            self.assertLessEqual(result["finalScore"], 95)


if __name__ == "__main__":
    unittest.main()
