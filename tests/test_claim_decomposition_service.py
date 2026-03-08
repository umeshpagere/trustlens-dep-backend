"""
Tests for app.services.claim_decomposition_service

All tests are fully offline — Azure OpenAI is mocked.
No network connections are required.

Run:
    cd /Users/umeshpagere/Downloads/trustlens-2-main/backend
    python3 -m pytest tests/test_claim_decomposition_service.py -v
"""

import asyncio
import sys
import types
import unittest
from unittest.mock import AsyncMock, patch, MagicMock

# ---------------------------------------------------------------------------
# Stub out optional runtime packages not available in the test env
# ---------------------------------------------------------------------------
for _pkg in ("googlesearch", "googlesearch.search", "sightengine", "sightengine.client"):
    if _pkg not in sys.modules:
        sys.modules[_pkg] = types.ModuleType(_pkg)
if "wikipedia" not in sys.modules:
    sys.modules["wikipedia"] = MagicMock()

from app.services.claim_decomposition_service import (
    decompose_claims,
    _extract_array_from_response,
    _validate_structured_claim,
    _fallback_decomposed,
    _primary_claim_fallback,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    return asyncio.run(coro)


def _llm_response(items: list) -> str:
    import json
    return json.dumps(items)


def _good_item(
    claim="US fighter jet shot down Iranian fighter jet",
    queries=None,
):
    return {
        "claim":             claim,
        "subject":           "US fighter jet",
        "action":            "shot down",
        "object":            "Iranian fighter jet",
        "context":           "Iran-US military conflict",
        "normalizedQueries": queries or [
            "Iran fighter jet shot down",
            "US shoots down Iranian aircraft",
            "Iranian aircraft shootdown",
        ],
    }


# ---------------------------------------------------------------------------
# _extract_array_from_response
# ---------------------------------------------------------------------------

class TestExtractArrayFromResponse(unittest.TestCase):

    def test_direct_array(self):
        items = [{"claim": "Test."}]
        raw = _llm_response(items)
        result = _extract_array_from_response(raw)
        self.assertEqual(result, items)

    def test_wrapped_in_dict(self):
        import json
        raw = json.dumps({"decomposed": [{"claim": "Test."}]})
        result = _extract_array_from_response(raw)
        self.assertIsInstance(result, list)
        self.assertEqual(result[0]["claim"], "Test.")

    def test_markdown_fenced(self):
        import json
        raw = "```json\n" + json.dumps([{"claim": "Fenced"}]) + "\n```"
        result = _extract_array_from_response(raw)
        self.assertEqual(result[0]["claim"], "Fenced")

    def test_invalid_returns_empty_list(self):
        result = _extract_array_from_response("not json at all @@##")
        self.assertEqual(result, [])

    def test_empty_string_returns_empty(self):
        self.assertEqual(_extract_array_from_response(""), [])


# ---------------------------------------------------------------------------
# _validate_structured_claim
# ---------------------------------------------------------------------------

class TestValidateStructuredClaim(unittest.TestCase):

    def test_valid_item_passes(self):
        result = _validate_structured_claim(_good_item())
        self.assertIsNotNone(result)
        self.assertEqual(result["claim"], "US fighter jet shot down Iranian fighter jet")

    def test_missing_claim_returns_none(self):
        self.assertIsNone(_validate_structured_claim({"subject": "X"}))

    def test_empty_claim_returns_none(self):
        self.assertIsNone(_validate_structured_claim({"claim": "   "}))

    def test_non_dict_returns_none(self):
        self.assertIsNone(_validate_structured_claim("a string"))
        self.assertIsNone(_validate_structured_claim(None))

    def test_missing_queries_defaults_to_claim_text(self):
        item = {"claim": "Flood in Mumbai", "normalizedQueries": []}
        result = _validate_structured_claim(item)
        self.assertEqual(result["normalizedQueries"], ["Flood in Mumbai"])

    def test_queries_capped_at_3(self):
        item = {**_good_item(), "normalizedQueries": ["q1", "q2", "q3", "q4", "q5"]}
        result = _validate_structured_claim(item)
        self.assertEqual(len(result["normalizedQueries"]), 3)


# ---------------------------------------------------------------------------
# _fallback_decomposed
# ---------------------------------------------------------------------------

class TestFallbackDecomposed(unittest.TestCase):

    def test_returns_one_item_per_claim(self):
        result = _fallback_decomposed(["Claim A", "Claim B"])
        self.assertEqual(len(result), 2)

    def test_fallback_uses_claim_as_query(self):
        result = _fallback_decomposed(["Government banned withdrawals"])
        self.assertEqual(result[0]["normalizedQueries"], ["Government banned withdrawals"])

    def test_empty_input_returns_empty(self):
        self.assertEqual(_fallback_decomposed([]), [])

    def test_caps_at_5(self):
        result = _fallback_decomposed([f"Claim {i}" for i in range(10)])
        self.assertEqual(len(result), 5)

    def test_blank_strings_skipped(self):
        result = _fallback_decomposed(["  ", "", "Real claim"])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["claim"], "Real claim")


# ---------------------------------------------------------------------------
# _primary_claim_fallback
# ---------------------------------------------------------------------------

class TestPrimaryClaimFallback(unittest.TestCase):

    def test_empty_string_returns_empty(self):
        self.assertEqual(_primary_claim_fallback(""), [])
        self.assertEqual(_primary_claim_fallback("   "), [])

    def test_valid_claim_returns_structured(self):
        result = _primary_claim_fallback("Amitabh Bachchan purchased land in Ayodhya")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["claim"], "Amitabh Bachchan purchased land in Ayodhya")
        self.assertGreaterEqual(len(result[0]["normalizedQueries"]), 2)
        
    def test_truncates_long_claims(self):
        long_claim = "word " * 25
        result = _primary_claim_fallback(long_claim)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]["claim"].split()), 20)


# ---------------------------------------------------------------------------
# decompose_claims — main public function
# ---------------------------------------------------------------------------

class TestDecomposeClaims(unittest.TestCase):

    # ---- Empty/blank input ------------------------------------------------

    def test_empty_list_returns_empty(self):
        result = _run(decompose_claims([]))
        self.assertEqual(result, [])

    def test_blank_strings_treated_as_empty(self):
        result = _run(decompose_claims(["  ", ""]))
        self.assertEqual(result, [])

    # ---- Successful LLM response ------------------------------------------

    @patch("app.services.claim_decomposition_service.get_azure_client")
    def test_valid_llm_response_returns_structured_claims(self, mock_client_fn):
        mock_client = MagicMock()
        mock_client_fn.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=_llm_response([_good_item()])))]
        )

        result = _run(decompose_claims(["US shot down Iranian jet"]))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["claim"], "US fighter jet shot down Iranian fighter jet")
        self.assertIn("normalizedQueries", result[0])
        self.assertEqual(result[0]["subject"], "US fighter jet")

    @patch("app.services.claim_decomposition_service.get_azure_client")
    def test_multiple_claims_all_returned(self, mock_client_fn):
        mock_client = MagicMock()
        mock_client_fn.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=_llm_response([
                _good_item("Flood in Mumbai"),
                _good_item("Government banned bank withdrawals"),
            ])))]
        )

        result = _run(decompose_claims(["Flood Mumbai", "Bank withdrawal ban"]))
        self.assertEqual(len(result), 2)

    # ---- Topic summary rejection (LLM returns []) -------------------------

    @patch("app.services.claim_decomposition_service.get_azure_client")
    def test_llm_returns_empty_array_falls_back_to_original(self, mock_client_fn):
        mock_client = MagicMock()
        mock_client_fn.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="[]"))]
        )

        result = _run(decompose_claims(["Iran USA war today"]))
        # fallback: original claim returned as-is
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["claim"], "Iran USA war today")

    @patch("app.services.claim_decomposition_service.get_azure_client")
    def test_llm_returns_empty_array_falls_back_to_primary_claim(self, mock_client_fn):
        mock_client = MagicMock()
        mock_client_fn.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="[]"))]
        )

        result = _run(decompose_claims(["Iran USA war today"], primary_claim="Primary Claim Text"))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["claim"], "Primary Claim Text")
        self.assertEqual(result[0]["normalizedQueries"], ["Primary Claim Text"])

    # ---- Deduplication ----------------------------------------------------

    @patch("app.services.claim_decomposition_service.get_azure_client")
    def test_duplicate_claims_sent_once_to_llm(self, mock_client_fn):
        mock_client = MagicMock()
        mock_client_fn.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=_llm_response([_good_item()])))]
        )

        _run(decompose_claims(["Claim A", "Claim A", "  claim a  "]))
        # LLM should only be called once — with deduplicated input
        self.assertEqual(mock_client.chat.completions.create.call_count, 1)

    # ---- LLM failure fallback ---------------------------------------------

    @patch("app.services.claim_decomposition_service.get_azure_client")
    def test_llm_failure_returns_fallback_claims(self, mock_client_fn):
        mock_client_fn.side_effect = Exception("Azure timeout")

        result = _run(decompose_claims(["US shot down Iranian jet"]))
        # Should gracefully fallback — not crash
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["claim"], "US shot down Iranian jet")
        self.assertEqual(result[0]["normalizedQueries"], ["US shot down Iranian jet"])

    @patch("app.services.claim_decomposition_service.get_azure_client")
    def test_llm_failure_returns_primary_claim_fallback(self, mock_client_fn):
        mock_client_fn.side_effect = Exception("Azure timeout")

        result = _run(decompose_claims(["US shot down Iranian jet"], primary_claim="My Fallback Primary Text"))
        # Should gracefully fallback — not crash
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["claim"], "My Fallback Primary Text")
        self.assertIn("My Fallback Primary Text", result[0]["normalizedQueries"])

    @patch("app.services.claim_decomposition_service.get_azure_client")
    def test_invalid_json_from_llm_returns_fallback(self, mock_client_fn):
        mock_client = MagicMock()
        mock_client_fn.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="INVALID JSON @@@"))]
        )

        result = _run(decompose_claims(["Claim A"]))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["claim"], "Claim A")

    # ---- Canonical scenario tests -----------------------------------------

    @patch("app.services.claim_decomposition_service.get_azure_client")
    def test_case1_iran_us_jet_produces_event_claim(self, mock_client_fn):
        """Topic 'Iran USA war' → event 'US shot down Iranian jet'."""
        mock_client = MagicMock()
        mock_client_fn.return_value = mock_client
        event_claim = _good_item("US fighter jet shot down Iranian fighter jet")
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=_llm_response([event_claim])))]
        )

        result = _run(decompose_claims(["Iran and USA war today"]))
        self.assertEqual(len(result), 1)
        self.assertIn("shot down", result[0]["claim"].lower())
        self.assertGreaterEqual(len(result[0]["normalizedQueries"]), 1)

    @patch("app.services.claim_decomposition_service.get_azure_client")
    def test_case2_flood_mumbai_structured_correctly(self, mock_client_fn):
        mock_client = MagicMock()
        mock_client_fn.return_value = mock_client
        flood_item = {
            "claim":             "A severe flood occurred in Mumbai on March 6, 2026.",
            "subject":           "Mumbai",
            "action":            "flooded",
            "object":            "",
            "context":           "India natural disaster",
            "normalizedQueries": ["Mumbai flood 2026", "flood Mumbai India"],
        }
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=_llm_response([flood_item])))]
        )

        result = _run(decompose_claims(["Flood in Mumbai today"]))
        self.assertEqual(result[0]["subject"], "Mumbai")
        self.assertIn("flood", result[0]["claim"].lower())

    @patch("app.services.claim_decomposition_service.get_azure_client")
    def test_case3_bank_ban_passthrough(self, mock_client_fn):
        mock_client = MagicMock()
        mock_client_fn.return_value = mock_client
        bank_item = _good_item(
            "Government banned bank withdrawals.",
            queries=["government ban bank withdrawals", "bank withdrawal ban official"],
        )
        bank_item["subject"] = "Government"
        bank_item["action"]  = "banned"
        bank_item["object"]  = "bank withdrawals"
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=_llm_response([bank_item])))]
        )

        result = _run(decompose_claims(["Government banned bank withdrawals"]))
        self.assertEqual(result[0]["subject"], "Government")
        self.assertEqual(result[0]["action"],  "banned")


if __name__ == "__main__":
    unittest.main()
