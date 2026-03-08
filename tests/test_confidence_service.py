"""
TrustLens Phase 5: Confidence Modeling Service Tests

All tests are fully offline — pure math, no network, no mocking required.

Run:
    python3 -m unittest tests.test_confidence_service -v
"""

import unittest

from app.services.confidence_service import (
    calculate_confidence,
    _calculate_coverage,
    _calculate_agreement,
    _calculate_evidence_strength,
    _confidence_level,
    _clamp,
    _NEUTRAL_DEFAULTS,
    _CONFIDENCE_MAX,
    _EVIDENCE_CAP,
    _WEIGHT_COVERAGE,
    _WEIGHT_AGREEMENT,
    _WEIGHT_EVIDENCE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_neutral_scores() -> dict:
    """Component scores exactly at their neutral defaults — no real data."""
    return dict(_NEUTRAL_DEFAULTS)


def _all_active_scores(value: float = 80.0) -> dict:
    """All signals clearly deviating from neutral."""
    return {
        "factCheckScore":         value,
        "knowledgeSupportScore":  value,
        "videoEvidenceScore":     value,
        "semanticScore":          value,
        "imageAuthenticityScore": value,
    }


def _no_flags() -> dict:
    return {"factCheckMatch": False, "contextMismatch": False,
            "imageReuseFound": False, "trustedDomain": False}


def _all_flags() -> dict:
    return {"factCheckMatch": True, "contextMismatch": True,
            "imageReuseFound": True, "trustedDomain": True}


# ---------------------------------------------------------------------------
# _clamp
# ---------------------------------------------------------------------------

class TestClamp(unittest.TestCase):

    def test_within_range_unchanged(self):
        self.assertEqual(_clamp(0.5, 0.0, 1.0), 0.5)

    def test_below_lo_clamped(self):
        self.assertEqual(_clamp(-1.0, 0.0, 1.0), 0.0)

    def test_above_hi_clamped(self):
        self.assertEqual(_clamp(2.0, 0.0, 1.0), 1.0)

    def test_nan_returns_lo(self):
        import math
        self.assertEqual(_clamp(math.nan, 0.0, 1.0), 0.0)

    def test_non_numeric_returns_lo(self):
        self.assertEqual(_clamp("bad", 0.0, 1.0), 0.0)


# ---------------------------------------------------------------------------
# Factor 1 — Coverage
# ---------------------------------------------------------------------------

class TestCalculateCoverage(unittest.TestCase):

    def test_all_neutral_scores_give_zero_coverage(self):
        """Neutral-at-default scores = no real signal present."""
        coverage = _calculate_coverage(_all_neutral_scores())
        self.assertEqual(coverage, 0.0)

    def test_all_active_scores_give_full_coverage(self):
        """All scores clearly off neutral."""
        coverage = _calculate_coverage(_all_active_scores(80.0))
        self.assertAlmostEqual(coverage, 1.0)

    def test_empty_scores_give_zero(self):
        self.assertEqual(_calculate_coverage({}), 0.0)

    def test_none_scores_give_zero(self):
        self.assertEqual(_calculate_coverage(None), 0.0)

    def test_partial_coverage(self):
        """Two out of five signals active → coverage = 0.4."""
        scores = dict(_NEUTRAL_DEFAULTS)
        scores["semanticScore"] = 80.0      # clearly off neutral
        scores["factCheckScore"] = 10.0     # clearly off neutral
        coverage = _calculate_coverage(scores)
        self.assertAlmostEqual(coverage, 2 / 5)

    def test_coverage_clamped_to_one(self):
        coverage = _calculate_coverage(_all_active_scores(100.0))
        self.assertLessEqual(coverage, 1.0)

    def test_coverage_non_negative(self):
        self.assertGreaterEqual(_calculate_coverage({}), 0.0)


# ---------------------------------------------------------------------------
# Factor 2 — Agreement
# ---------------------------------------------------------------------------

class TestCalculateAgreement(unittest.TestCase):

    def test_identical_scores_perfect_agreement(self):
        """Std dev = 0 → agreement = 1.0."""
        scores = _all_active_scores(60.0)
        agreement = _calculate_agreement(scores)
        self.assertAlmostEqual(agreement, 1.0)

    def test_empty_scores_assume_perfect_agreement(self):
        """No scores → cannot disagree → 1.0."""
        self.assertAlmostEqual(_calculate_agreement({}), 1.0)

    def test_single_score_perfect_agreement(self):
        self.assertAlmostEqual(_calculate_agreement({"semanticScore": 75}), 1.0)

    def test_high_variance_low_agreement(self):
        """Polarised scores → high std dev → low agreement."""
        scores = {
            "factCheckScore":         10.0,
            "knowledgeSupportScore":  90.0,
            "videoEvidenceScore":     5.0,
            "semanticScore":          100.0,
            "imageAuthenticityScore": 95.0,
        }
        agreement = _calculate_agreement(scores)
        self.assertLess(agreement, 0.5)

    def test_agreement_clamped_between_0_and_1(self):
        scores = {"a": 0.0, "b": 100.0}
        agreement = _calculate_agreement(scores)
        self.assertGreaterEqual(agreement, 0.0)
        self.assertLessEqual(agreement, 1.0)

    def test_moderate_variance_moderate_agreement(self):
        scores = {
            "semanticScore":         60.0,
            "factCheckScore":        70.0,
            "knowledgeSupportScore": 65.0,
        }
        agreement = _calculate_agreement(scores)
        self.assertGreater(agreement, 0.7)


# ---------------------------------------------------------------------------
# Factor 3 — Evidence Strength
# ---------------------------------------------------------------------------

class TestCalculateEvidenceStrength(unittest.TestCase):

    def test_no_flags_zero_strength(self):
        self.assertEqual(_calculate_evidence_strength(_no_flags()), 0.0)

    def test_all_flags_capped(self):
        """Even with all flags, strength is capped at _EVIDENCE_CAP."""
        strength = _calculate_evidence_strength(_all_flags())
        self.assertLessEqual(strength, _EVIDENCE_CAP)

    def test_fact_check_match_adds_bonus(self):
        flags = _no_flags()
        flags["factCheckMatch"] = True
        strength = _calculate_evidence_strength(flags)
        self.assertGreater(strength, 0.0)
        self.assertAlmostEqual(strength, 0.15)

    def test_context_mismatch_adds_bonus(self):
        flags = _no_flags()
        flags["contextMismatch"] = True
        strength = _calculate_evidence_strength(flags)
        self.assertAlmostEqual(strength, 0.15)

    def test_image_reuse_adds_bonus(self):
        flags = _no_flags()
        flags["imageReuseFound"] = True
        strength = _calculate_evidence_strength(flags)
        self.assertAlmostEqual(strength, 0.10)

    def test_trusted_domain_adds_bonus(self):
        flags = _no_flags()
        flags["trustedDomain"] = True
        strength = _calculate_evidence_strength(flags)
        self.assertAlmostEqual(strength, 0.10)

    def test_none_flags_treated_as_empty(self):
        self.assertEqual(_calculate_evidence_strength(None), 0.0)

    def test_evidence_non_negative(self):
        self.assertGreaterEqual(_calculate_evidence_strength(_no_flags()), 0.0)


# ---------------------------------------------------------------------------
# Confidence level labels
# ---------------------------------------------------------------------------

class TestConfidenceLevel(unittest.TestCase):

    def test_high_label(self):
        self.assertEqual(_confidence_level(0.80), "High")
        self.assertEqual(_confidence_level(0.95), "High")
        self.assertEqual(_confidence_level(0.85), "High")

    def test_moderate_label(self):
        self.assertEqual(_confidence_level(0.60), "Moderate")
        self.assertEqual(_confidence_level(0.79), "Moderate")

    def test_low_label(self):
        self.assertEqual(_confidence_level(0.40), "Low")
        self.assertEqual(_confidence_level(0.59), "Low")

    def test_very_low_label(self):
        self.assertEqual(_confidence_level(0.39), "Very Low")
        self.assertEqual(_confidence_level(0.0), "Very Low")


# ---------------------------------------------------------------------------
# calculate_confidence — integration
# ---------------------------------------------------------------------------

class TestCalculateConfidence(unittest.TestCase):

    def test_output_has_all_required_keys(self):
        result = calculate_confidence(_all_active_scores(), _no_flags())
        for key in ["confidenceScore", "confidenceLevel", "coverageScore",
                    "agreementScore", "evidenceStrengthScore"]:
            self.assertIn(key, result)

    def test_confidence_never_equals_one(self):
        """Epistemic humility cap — even on ideal input."""
        result = calculate_confidence(_all_active_scores(70.0), _all_flags())
        self.assertLess(result["confidenceScore"], 1.0)

    def test_confidence_capped_at_0_95(self):
        result = calculate_confidence(_all_active_scores(70.0), _all_flags())
        self.assertLessEqual(result["confidenceScore"], _CONFIDENCE_MAX)

    def test_confidence_non_negative(self):
        result = calculate_confidence({}, {})
        self.assertGreaterEqual(result["confidenceScore"], 0.0)

    def test_none_inputs_return_safe_default(self):
        result = calculate_confidence(None, None)
        self.assertIn(result["confidenceLevel"], ["High", "Moderate", "Low", "Very Low"])
        self.assertGreaterEqual(result["confidenceScore"], 0.0)

    def test_all_neutral_low_coverage_lowers_confidence(self):
        """With no real signal deviation, coverage is 0 → lower confidence."""
        result_neutral = calculate_confidence(_all_neutral_scores(), _no_flags())
        result_active  = calculate_confidence(_all_active_scores(75.0), _no_flags())
        self.assertLessEqual(result_neutral["confidenceScore"],
                             result_active["confidenceScore"])

    def test_evidence_flags_increase_confidence(self):
        """Same scores, more evidence flags → higher confidence."""
        scores = _all_active_scores(60.0)
        without = calculate_confidence(scores, _no_flags())
        with_ev  = calculate_confidence(scores, _all_flags())
        self.assertGreaterEqual(with_ev["confidenceScore"],
                                without["confidenceScore"])

    def test_score_is_float(self):
        result = calculate_confidence(_all_active_scores(), _no_flags())
        self.assertIsInstance(result["confidenceScore"], float)

    def test_formula_weights_sum_contribution(self):
        """
        Verify that a known input produces an expected output range
        based on the formula: coverage*0.4 + agreement*0.3 + evidence*0.3.
        All scores identical → agreement = 1.0; all active → coverage = 1.0;
        no evidence → evidence = 0. Expected ≈ 0.4 + 0.3 = 0.70.
        """
        scores = _all_active_scores(75.0)  # all same → agreement 1.0, coverage 1.0
        result = calculate_confidence(scores, _no_flags())
        expected_approx = _WEIGHT_COVERAGE * 1.0 + _WEIGHT_AGREEMENT * 1.0 + _WEIGHT_EVIDENCE * 0.0
        self.assertAlmostEqual(result["confidenceScore"], expected_approx, places=2)

    def test_single_signal_no_crash(self):
        """Only one score provided — std dev safe fallback."""
        result = calculate_confidence({"semanticScore": 65}, _no_flags())
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result["confidenceScore"], 0.0)

    def test_invalid_score_values_handled(self):
        """Non-numeric values must not crash the service."""
        scores = {
            "semanticScore":   "invalid",
            "factCheckScore":  None,
            "videoEvidenceScore": float("inf"),
        }
        result = calculate_confidence(scores, _no_flags())
        self.assertIn("confidenceScore", result)

    def test_level_matches_score(self):
        """confidenceLevel must be consistent with confidenceScore."""
        result = calculate_confidence(_all_active_scores(70.0), _all_flags())
        score = result["confidenceScore"]
        level = result["confidenceLevel"]
        if score >= 0.80:
            self.assertEqual(level, "High")
        elif score >= 0.60:
            self.assertEqual(level, "Moderate")
        elif score >= 0.40:
            self.assertEqual(level, "Low")
        else:
            self.assertEqual(level, "Very Low")


if __name__ == "__main__":
    unittest.main()
