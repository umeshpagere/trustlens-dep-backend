"""
Tests for app.services.claim_verification_service

All tests are fully offline — evidence retrieval and LLM calls are mocked.
No network connections are required.

Run:
    cd /Users/umeshpagere/Downloads/trustlens-2-main/backend
    python3 -m pytest tests/test_claim_verification_service.py -v
"""

import asyncio
import sys
import types
import unittest
from unittest.mock import AsyncMock, patch

# ---------------------------------------------------------------------------
# Stub out optional runtime packages that aren't installed in the test env.
# Use MagicMock so any attribute access / callable on the stub succeeds.
# ---------------------------------------------------------------------------
from unittest.mock import MagicMock
import sys
import types

for _pkg in ("googlesearch", "googlesearch.search", "sightengine", "sightengine.client"):
    if _pkg not in sys.modules:
        sys.modules[_pkg] = types.ModuleType(_pkg)

# 'wikipedia' needs to be a MagicMock so wikipedia_service.py's module-level
# call to wikipedia.set_user_agent() doesn't raise AttributeError.
if "wikipedia" not in sys.modules:
    sys.modules["wikipedia"] = MagicMock()

from app.services.claim_verification_service import verify_all_claims, _verify_single_claim


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    """Run an async coroutine in the test loop."""
    return asyncio.run(coro)


def _mock_evidence():
    """Minimal evidence dict returned by aggregate_evidence."""
    return {
        "factChecks":      [],
        "wikipedia":       {"title": "Test", "summary": "Some summary."},
        "newsArticles":    [],
        "ranked_evidence": [
            {"source": "Wikipedia", "title": "Test", "description": "Some summary.",
             "trustScore": 0.80, "domain": "wikipedia.org", "type": "wikipedia"},
        ],
    }


def _mock_verification(score: float, verdict: str):
    """Build a fake verify_claim_with_evidence return value."""
    return {
        "knowledgeSupportScore": score,
        "verdict":               verdict,
        "reasoning":             f"Mocked reasoning for verdict={verdict}",
        "trustedSourcesUsed":    ["wikipedia.org"],
        "evidenceSources":       ["Wikipedia"],
    }


# ---------------------------------------------------------------------------
# verify_all_claims — aggregation logic
# ---------------------------------------------------------------------------

class TestVerifyAllClaims(unittest.TestCase):

    # ---- Empty input -------------------------------------------------------

    def test_empty_claims_returns_empty_list_and_zero_score(self):
        result_claims, score = _run(verify_all_claims([]))
        self.assertEqual(result_claims, [])
        self.assertEqual(score, 0.0)

    def test_single_empty_string_treated_as_empty(self):
        """A list with only blank strings should be treated as empty."""
        result_claims, score = _run(verify_all_claims(["", "  "]))
        self.assertEqual(result_claims, [])
        self.assertEqual(score, 0.0)

    # ---- Score aggregation ------------------------------------------------

    @patch("app.services.claim_verification_service.aggregate_evidence", new_callable=AsyncMock)
    @patch("app.services.claim_verification_service.verify_claim_with_evidence", new_callable=AsyncMock)
    def test_single_claim_score_passed_through(self, mock_verify, mock_agg):
        mock_agg.return_value = _mock_evidence()
        mock_verify.return_value = _mock_verification(0.8, "supported")

        claims, score = _run(verify_all_claims(["COVID-19 was declared a pandemic in 2020."]))
        self.assertEqual(len(claims), 1)
        self.assertAlmostEqual(score, 0.8, places=4)
        self.assertEqual(claims[0]["verdict"], "supported")

    @patch("app.services.claim_verification_service.aggregate_evidence", new_callable=AsyncMock)
    @patch("app.services.claim_verification_service.verify_claim_with_evidence", new_callable=AsyncMock)
    def test_three_claims_mean_score(self, mock_verify, mock_agg):
        """Mean of 0.2, 0.4, 0.6 = 0.4."""
        mock_agg.return_value = _mock_evidence()
        mock_verify.side_effect = [
            _mock_verification(0.2, "contradicted"),
            _mock_verification(0.4, "uncertain"),
            _mock_verification(0.6, "supported"),
        ]
        claims = [
            "Vaccines contain microchips.",
            "The sky is green.",
            "WHO declared COVID pandemic in 2020.",
        ]
        _, score = _run(verify_all_claims(claims))
        self.assertAlmostEqual(score, 0.4, places=4)

    @patch("app.services.claim_verification_service.aggregate_evidence", new_callable=AsyncMock)
    @patch("app.services.claim_verification_service.verify_claim_with_evidence", new_callable=AsyncMock)
    def test_all_high_scores_gives_high_aggregate(self, mock_verify, mock_agg):
        mock_agg.return_value = _mock_evidence()
        mock_verify.return_value = _mock_verification(1.0, "supported")

        _, score = _run(verify_all_claims(["Claim A", "Claim B", "Claim C"]))
        self.assertAlmostEqual(score, 1.0, places=4)

    @patch("app.services.claim_verification_service.aggregate_evidence", new_callable=AsyncMock)
    @patch("app.services.claim_verification_service.verify_claim_with_evidence", new_callable=AsyncMock)
    def test_all_low_scores_gives_low_aggregate(self, mock_verify, mock_agg):
        mock_agg.return_value = _mock_evidence()
        mock_verify.return_value = _mock_verification(0.0, "contradicted")

        _, score = _run(verify_all_claims(["False claim A", "False claim B"]))
        self.assertAlmostEqual(score, 0.0, places=4)

    @patch("app.services.claim_verification_service.aggregate_evidence", new_callable=AsyncMock)
    @patch("app.services.claim_verification_service.verify_claim_with_evidence", new_callable=AsyncMock)
    def test_mixed_verdicts_score_in_middle_range(self, mock_verify, mock_agg):
        """supported + contradicted + uncertain → score between 0.25 and 0.75."""
        mock_agg.return_value = _mock_evidence()
        mock_verify.side_effect = [
            _mock_verification(0.9, "supported"),
            _mock_verification(0.1, "contradicted"),
            _mock_verification(0.5, "uncertain"),
        ]
        _, score = _run(verify_all_claims(["Claim 1", "Claim 2", "Claim 3"]))
        self.assertGreater(score, 0.25)
        self.assertLess(score, 0.75)

    # ---- Per-claim output structure ---------------------------------------

    @patch("app.services.claim_verification_service.aggregate_evidence", new_callable=AsyncMock)
    @patch("app.services.claim_verification_service.verify_claim_with_evidence", new_callable=AsyncMock)
    def test_verified_claim_has_required_keys(self, mock_verify, mock_agg):
        mock_agg.return_value = _mock_evidence()
        mock_verify.return_value = _mock_verification(0.7, "supported")

        claims, _ = _run(verify_all_claims(["WHO declared pandemic in 2020."]))
        self.assertEqual(len(claims), 1)
        claim = claims[0]
        for key in ["claim", "verdict", "knowledgeSupportScore", "reasoning",
                    "trustedSourcesUsed", "evidenceSources"]:
            self.assertIn(key, claim, msg=f"Missing key: {key}")

    @patch("app.services.claim_verification_service.aggregate_evidence", new_callable=AsyncMock)
    @patch("app.services.claim_verification_service.verify_claim_with_evidence", new_callable=AsyncMock)
    def test_original_claim_text_preserved(self, mock_verify, mock_agg):
        mock_agg.return_value = _mock_evidence()
        mock_verify.return_value = _mock_verification(0.6, "supported")

        claim_text = "The government banned bank withdrawals in 2020."
        result_claims, _ = _run(verify_all_claims([claim_text]))
        self.assertEqual(result_claims[0]["claim"], claim_text)

    # ---- Deduplication ----------------------------------------------------

    @patch("app.services.claim_verification_service.aggregate_evidence", new_callable=AsyncMock)
    @patch("app.services.claim_verification_service.verify_claim_with_evidence", new_callable=AsyncMock)
    def test_duplicate_claims_verified_once(self, mock_verify, mock_agg):
        """Identical claims (case-insensitive) should be deduplicated."""
        mock_agg.return_value = _mock_evidence()
        mock_verify.return_value = _mock_verification(0.7, "supported")

        result_claims, _ = _run(verify_all_claims(["Claim A", "Claim A", "  claim a  "]))
        # Should only verify once
        self.assertEqual(len(result_claims), 1)
        self.assertEqual(mock_verify.call_count, 1)

    # ---- Resilience -------------------------------------------------------

    @patch("app.services.claim_verification_service.aggregate_evidence", new_callable=AsyncMock)
    @patch("app.services.claim_verification_service.verify_claim_with_evidence", new_callable=AsyncMock)
    def test_single_failing_claim_does_not_crash_others(self, mock_verify, mock_agg):
        """If one claim's LLM call throws, others still succeed."""
        mock_agg.return_value = _mock_evidence()
        mock_verify.side_effect = [
            Exception("Azure timeout"),
            _mock_verification(0.8, "supported"),
        ]
        result_claims, score = _run(verify_all_claims(["Bad claim", "Good claim"]))
        self.assertEqual(len(result_claims), 2)
        # Failed claim defaults to uncertain / 0.5
        self.assertEqual(result_claims[0]["verdict"], "uncertain")
        self.assertAlmostEqual(result_claims[0]["knowledgeSupportScore"], 0.5, places=4)
        # Successful claim preserved
        self.assertEqual(result_claims[1]["verdict"], "supported")
        # Score is mean(0.5, 0.8) = 0.65
        self.assertAlmostEqual(score, 0.65, places=4)

    @patch("app.services.claim_verification_service.aggregate_evidence", new_callable=AsyncMock)
    def test_evidence_retrieval_failure_gives_uncertain(self, mock_agg):
        """If aggregate_evidence throws, the claim resolves to uncertain/0.5."""
        mock_agg.side_effect = Exception("Network failure")

        result_claims, score = _run(verify_all_claims(["Any claim here."]))
        self.assertEqual(len(result_claims), 1)
        self.assertEqual(result_claims[0]["verdict"], "uncertain")
        self.assertAlmostEqual(score, 0.5, places=4)

    # ---- Scenario: Case 1 — verified false claim --------------------------

    @patch("app.services.claim_verification_service.aggregate_evidence", new_callable=AsyncMock)
    @patch("app.services.claim_verification_service.verify_claim_with_evidence", new_callable=AsyncMock)
    def test_case1_false_claim_contradicted_low_score(self, mock_verify, mock_agg):
        """Vaccines contain microchips → contradicted → knowledgeSupportScore very low."""
        mock_agg.return_value = _mock_evidence()
        mock_verify.return_value = _mock_verification(0.05, "contradicted")

        claims_result, score = _run(verify_all_claims(["Vaccines contain microchips."]))
        self.assertEqual(claims_result[0]["verdict"], "contradicted")
        self.assertLess(score, 0.3)

    # ---- Scenario: Case 2 — verified true claim ---------------------------

    @patch("app.services.claim_verification_service.aggregate_evidence", new_callable=AsyncMock)
    @patch("app.services.claim_verification_service.verify_claim_with_evidence", new_callable=AsyncMock)
    def test_case2_true_claim_supported_high_score(self, mock_verify, mock_agg):
        """WHO declared COVID-19 a pandemic → supported → knowledgeSupportScore high."""
        mock_agg.return_value = _mock_evidence()
        mock_verify.return_value = _mock_verification(0.95, "supported")

        claims_result, score = _run(
            verify_all_claims(["WHO declared COVID-19 a pandemic in 2020."])
        )
        self.assertEqual(claims_result[0]["verdict"], "supported")
        self.assertGreater(score, 0.7)

    # ---- Scenario: Case 3 — unknown claim ---------------------------------

    @patch("app.services.claim_verification_service.aggregate_evidence", new_callable=AsyncMock)
    @patch("app.services.claim_verification_service.verify_claim_with_evidence", new_callable=AsyncMock)
    def test_case3_unknown_claim_uncertain(self, mock_verify, mock_agg):
        """Secret government operation → insufficient evidence → uncertain."""
        mock_agg.return_value = {
            "factChecks": [], "wikipedia": None, "newsArticles": [],
            "ranked_evidence": [],
        }
        mock_verify.return_value = _mock_verification(0.5, "uncertain")

        claims_result, score = _run(
            verify_all_claims(["Secret government operation happening globally."])
        )
        self.assertEqual(claims_result[0]["verdict"], "uncertain")
        self.assertAlmostEqual(score, 0.5, places=2)


if __name__ == "__main__":
    unittest.main()
