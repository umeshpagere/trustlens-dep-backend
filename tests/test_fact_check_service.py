"""
TrustLens Phase 2: Fact-Check Service Test Cases

Tests score calculation logic with mocked API responses.
Does not require GOOGLE_FACTCHECK_API_KEY or network access.

Run: python -m unittest tests.test_fact_check_service
"""

import unittest
from app.services.fact_check_service import (
    calculate_fact_check_score,
    _sanitize_claim,
)


class TestCalculateFactCheckScore(unittest.TestCase):
    """Test calculate_fact_check_score normalization logic."""

    def test_known_false_claim_returns_low_score(self):
        """Example: Known false claim from fact-check API returns score 10."""
        api_response = {
            "claims": [
                {
                    "text": "Crime has doubled in the last 2 years.",
                    "claimant": "Politician X",
                    "claimReview": [
                        {
                            "publisher": {"name": "Snopes", "site": "snopes.com"},
                            "url": "https://www.snopes.com/fact-check/crime-doubled/",
                            "title": "Has Crime Doubled?",
                            "textualRating": "False",
                        }
                    ],
                }
            ],
        }
        result = calculate_fact_check_score(api_response)

        assert result["factCheckScore"] == 10
        assert result["matchFound"] is True
        assert result["verdict"] == "False"
        assert result["source"] == "Snopes"
        assert "snopes.com" in result["referenceURL"] or result["referenceURL"]
        assert result["confidenceAdjustment"] == 0.0
        assert all(
            key in result
            for key in ["factCheckScore", "matchFound", "verdict", "source", "referenceURL", "confidenceAdjustment"]
        )

    def test_no_match_returns_neutral_score(self):
        """Example: No fact-check match returns score 50 and confidenceAdjustment -0.1."""
        api_response = {"claims": [], "nextPageToken": ""}
        result = calculate_fact_check_score(api_response)

        assert result["factCheckScore"] == 50
        assert result["matchFound"] is False
        assert result["verdict"] == "No Match"
        assert result["source"] == ""
        assert result["referenceURL"] == ""
        assert result["confidenceAdjustment"] == -0.1

    def test_mostly_false_returns_25(self):
        """Mostly False rating maps to score 25."""
        api_response = {
            "claims": [
                {
                    "text": "Example claim",
                    "claimReview": [
                        {"publisher": {"name": "AFP"}, "url": "https://example.com", "textualRating": "Mostly false"}
                    ],
                }
            ],
        }
        result = calculate_fact_check_score(api_response)
        assert result["factCheckScore"] == 25
        assert result["verdict"] == "Mostly False"

    def test_true_returns_90(self):
        """True rating maps to score 90."""
        api_response = {
            "claims": [
                {
                    "text": "Example claim",
                    "claimReview": [
                        {"publisher": {"name": "Reuters"}, "url": "https://example.com", "textualRating": "True"}
                    ],
                }
            ],
        }
        result = calculate_fact_check_score(api_response)
        assert result["factCheckScore"] == 90
        assert result["verdict"] == "True"

    def test_empty_response_returns_neutral(self):
        """Empty or invalid response returns neutral default."""
        result = calculate_fact_check_score({})
        assert result["factCheckScore"] == 50
        assert result["matchFound"] is False
        assert result["verdict"] == "No Match"


class TestSanitizeClaim(unittest.TestCase):
    """Test claim sanitization before API call."""

    def test_truncates_long_claim(self):
        long_claim = "x" * 1000
        result = _sanitize_claim(long_claim)
        self.assertLessEqual(len(result), 500)

    def test_strips_whitespace(self):
        assert _sanitize_claim("  claim  ") == "claim"

    def test_empty_returns_empty(self):
        assert _sanitize_claim("") == ""
        assert _sanitize_claim("   ") == ""
