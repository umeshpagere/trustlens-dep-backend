"""
TrustLens Claim Validator — Unit Tests

Tests is_valid_claim, normalize_claim, and filter_and_normalize_claims from
app.services.claim_validator.

Run:
    cd /Users/umeshpagere/Downloads/trustlens-2-main/backend
    python -m pytest tests/test_claim_validator.py -v
"""

import unittest
from app.services.claim_validator import (
    is_valid_claim,
    normalize_claim,
    filter_and_normalize_claims,
)


class TestIsValidClaim(unittest.TestCase):
    """Validate that is_valid_claim correctly accepts and rejects claims."""

    # ------------------------------------------------------------------ #
    # VALID claims – all should return True
    # ------------------------------------------------------------------ #
    def test_valid_who_pandemic(self):
        self.assertTrue(is_valid_claim("WHO declared a global health emergency."))

    def test_valid_police_arrest(self):
        self.assertTrue(is_valid_claim("Police arrested protesters in Paris."))

    def test_valid_government_ban(self):
        self.assertTrue(is_valid_claim("Government banned bank withdrawals."))

    def test_valid_nasa_mars(self):
        self.assertTrue(is_valid_claim("NASA confirmed water was found on Mars."))

    def test_valid_with_statistic(self):
        self.assertTrue(is_valid_claim("The unemployment rate rose to 8.4% in July 2020."))

    def test_valid_election_result(self):
        self.assertTrue(is_valid_claim("Congress passed the infrastructure bill."))

    def test_valid_scientist_discovery(self):
        self.assertTrue(is_valid_claim("Scientists discovered a new species of bacteria in Antarctica."))

    def test_valid_company_action(self):
        self.assertTrue(is_valid_claim("Apple released a new iPhone model in September."))

    # ------------------------------------------------------------------ #
    # INVALID claims – all should return False
    # ------------------------------------------------------------------ #
    def test_invalid_vague_suffering(self):
        self.assertFalse(is_valid_claim("People are suffering."))

    def test_invalid_vague_situation(self):
        self.assertFalse(is_valid_claim("The situation is getting worse."))

    def test_invalid_vague_something_happening(self):
        self.assertFalse(is_valid_claim("Something big is happening."))

    def test_invalid_opinion(self):
        self.assertFalse(is_valid_claim("This is a terrible situation."))

    def test_invalid_too_short(self):
        self.assertFalse(is_valid_claim("Bad"))

    def test_invalid_two_words(self):
        self.assertFalse(is_valid_claim("Very bad"))

    def test_invalid_wake_up(self):
        self.assertFalse(is_valid_claim("Wake up and see the truth."))

    def test_invalid_empty(self):
        self.assertFalse(is_valid_claim(""))

    def test_invalid_none(self):
        self.assertFalse(is_valid_claim(None))

    def test_invalid_you_wont_believe(self):
        self.assertFalse(is_valid_claim("You won't believe what happened next."))


class TestNormalizeClaim(unittest.TestCase):
    """Validate that normalize_claim produces clean, API-friendly strings."""

    def test_strips_breaking_prefix(self):
        result = normalize_claim("BREAKING: Government banned bank withdrawals!")
        self.assertNotIn("BREAKING", result)
        self.assertIn("Government", result)
        self.assertIn("bank withdrawals", result)

    def test_removes_hashtags(self):
        result = normalize_claim("#COVID WHO declared a pandemic #health")
        self.assertNotIn("#COVID", result)
        self.assertNotIn("#health", result)
        self.assertIn("WHO", result)
        self.assertIn("pandemic", result)

    def test_removes_intensifiers(self):
        result = normalize_claim("Government completely banned all bank withdrawals.")
        self.assertNotIn("completely", result)
        self.assertIn("Government", result)

    def test_strips_exclamation(self):
        result = normalize_claim("NASA confirmed water on Mars!!!")
        self.assertNotIn("!", result)

    def test_strips_ellipsis(self):
        result = normalize_claim("Scientists found evidence... of life on Mars.")
        self.assertNotIn("...", result)

    def test_empty_string_returns_empty(self):
        self.assertEqual(normalize_claim(""), "")

    def test_none_returns_empty(self):
        self.assertEqual(normalize_claim(None), "")

    def test_preserves_named_entity(self):
        """Proper nouns like WHO, NASA, Paris should be preserved."""
        result = normalize_claim("WHO declared an emergency in Paris.")
        self.assertIn("WHO", result)
        self.assertIn("Paris", result)

    def test_alert_prefix_removed(self):
        result = normalize_claim("ALERT: Police arrested 10 protesters.")
        self.assertNotIn("ALERT:", result)
        self.assertIn("Police", result)


class TestFilterAndNormalizeClaims(unittest.TestCase):
    """Validate the end-to-end filtering pipeline."""

    def test_filters_vague_keeps_valid(self):
        raw = [
            "People are suffering.",
            "Government banned bank withdrawals.",
        ]
        result = filter_and_normalize_claims(raw)
        self.assertEqual(len(result), 1)
        self.assertIn("Government", result[0])
        self.assertIn("bank withdrawals", result[0])

    def test_all_vague_returns_empty(self):
        raw = [
            "People are suffering.",
            "Things are getting worse.",
            "The situation is bad.",
        ]
        result = filter_and_normalize_claims(raw)
        self.assertEqual(result, [])

    def test_all_valid_passes_through(self):
        raw = [
            "WHO declared a global health emergency.",
            "Police arrested protesters in Paris.",
            "Government banned bank withdrawals.",
        ]
        result = filter_and_normalize_claims(raw)
        self.assertEqual(len(result), 3)

    def test_caps_at_5(self):
        raw = [
            "WHO declared a global health emergency.",
            "Police arrested protesters in Paris.",
            "Government banned bank withdrawals.",
            "NASA confirmed water was found on Mars.",
            "Scientists discovered a new species in Antarctica.",
            "Apple released a new iPhone model.",  # 6th — should be dropped
        ]
        result = filter_and_normalize_claims(raw)
        self.assertEqual(len(result), 5)

    def test_normalizes_before_returning(self):
        raw = ["BREAKING: NASA confirmed water on Mars!"]
        result = filter_and_normalize_claims(raw)
        self.assertEqual(len(result), 1)
        self.assertNotIn("BREAKING:", result[0])
        self.assertNotIn("!", result[0])

    def test_empty_input_returns_empty(self):
        self.assertEqual(filter_and_normalize_claims([]), [])

    def test_none_input_returns_empty(self):
        self.assertEqual(filter_and_normalize_claims(None), [])

    def test_deduplication_not_required_but_order_preserved(self):
        """Claims should come out in the same order they came in."""
        raw = [
            "WHO declared a global health emergency.",
            "Police arrested protesters in Paris.",
        ]
        result = filter_and_normalize_claims(raw)
        self.assertIn("WHO", result[0])
        self.assertIn("Police", result[1])


if __name__ == "__main__":
    unittest.main()
