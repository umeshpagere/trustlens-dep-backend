"""
TrustLens Claim Decomposition Service

Implements a 3-stage claim hardening pipeline in a SINGLE LLM call:

  Stage 1 — Event Detection:    reject topic-summaries, extract event statements
  Stage 2 — Claim Structuring:  decompose each event into subject/action/object
  Stage 3 — Query Normalization: generate 2-3 search-optimised query variants

A single LLM call handles all three stages, returning structured JSON so that
downstream evidence retrieval (Fact Check, Wikipedia, News) receives focused,
event-level queries rather than broad topic strings.

Example transformation:
  IN:  "Iran and USA war today"         ← topic summary, not a verifiable event
  OUT: {
         "claim":             "US fighter jet shot down Iranian fighter jet",
         "subject":           "US fighter jet",
         "action":            "shot down",
         "object":            "Iranian fighter jet",
         "context":           "Iran-US military conflict",
         "normalizedQueries": [
           "Iran fighter jet shot down US aircraft",
           "US shoots down Iranian jet",
           "Iranian aircraft shootdown"
         ]
       }

Graceful degradation:
  - If the LLM fails, the original validated claims are returned as-is
    with a single normalizedQuery (the claim itself).
  - The caller (analyze.py) always receives a valid list; it never crashes.
"""

import asyncio
import json
import logging
import re
from typing import Any

from app.config.azure import get_azure_client
from app.config.settings import Config
from app.services.query_generator import generate_queries

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt engineering
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an EVENT-LEVEL CLAIM DECOMPOSITION SYSTEM used in a fact-verification pipeline.

Your task is to convert raw claim strings into structured, event-level factual claims that can be verified using news, fact-check, and knowledge databases.

The goal is to extract ONLY the core real-world events that journalists or fact-checkers would verify.

------------------------------------------------

STAGE 1 — EVENT VALIDATION

Accept a claim ONLY if it describes a CLEAR, VERIFIABLE EVENT.

A valid event must include:

• a subject (person, organization, government, object)
• an action (what happened)
• a concrete outcome or object (optional but preferred)

The claim must describe something that could appear as a NEWS HEADLINE.

REJECT the following types of sentences:

• topic summaries
• vague descriptions
• contextual information
• background explanations
• investment counts
• statements about companies or spokespersons
• commentary about the event

Examples:

BAD (REJECT)
"Iran and USA war today"
"Flood situation getting worse"
"This is Amitabh Bachchan's third investment in Ayodhya"
"The realty company said the deal was completed"

GOOD (ACCEPT)
"US fighter jet shot down an Iranian fighter jet over the Gulf"
"Amitabh Bachchan purchased land in Ayodhya"
"Government declared emergency after flooding in Mumbai"

If a sentence only provides context about an event rather than describing the event itself, REJECT it.

------------------------------------------------

STAGE 2 — CLAIM STRUCTURING

For each accepted event extract:

claim:
  A clean factual sentence describing the event.

subject:
  The main actor (person, organization, government).

action:
  The event verb (short phrase).

object:
  The target or outcome of the action.
  Use an empty string if none exists.

context:
  A short 3–5 word phrase describing the background topic.

Example:

Input:
"Amitabh Bachchan purchased a 2.67-acre plot in Ayodhya for ₹35 crore."

Output fields:

subject: "Amitabh Bachchan"
action: "purchased"
object: "2.67-acre plot in Ayodhya"
context: "Ayodhya real estate investment"

------------------------------------------------

STAGE 3 — QUERY NORMALIZATION

Generate 2–3 short search queries optimized for retrieving evidence.

Rules:

• 3–7 words only
• MUST contain the main subject
• Focus on the event, not background context
• Avoid long sentences
• Each query should be phrased differently

Example queries:

"Amitabh Bachchan Ayodhya land purchase"
"Amitabh Bachchan bought land Ayodhya"
"Amitabh Bachchan Ayodhya property investment"

These queries will be used with:

• Google Fact Check API
• Wikipedia search
• News API

------------------------------------------------

IMPORTANT RULES

1. Prioritize the MAIN EVENT claim if multiple claims describe the same story.

2. Never output contextual claims like:
   "This is his third investment."

3. If a claim describes an event reported by news media, it should be included.

4. Prefer claims that follow a news headline structure.

------------------------------------------------

OUTPUT FORMAT

Return ONLY a JSON array with no additional text.

[
  {
    "claim": "<event-level factual sentence>",
    "subject": "<actor/entity>",
    "action": "<verb/event>",
    "object": "<target/outcome or empty>",
    "context": "<3-5 word background>",
    "normalizedQueries": ["<query 1>", "<query 2>", "<query 3>"]
  }
]

------------------------------------------------

OUTPUT LIMITS

• Maximum 5 structured claims
• If no valid event claims exist, return []

------------------------------------------------
"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sync_llm_decompose(client, messages: list, max_tokens: int) -> str:
    """Synchronous Azure OpenAI call — runs via asyncio.to_thread."""
    response = client.chat.completions.create(
        model=Config.AZURE_OPENAI_DEPLOYMENT,
        messages=messages,
        temperature=0.1,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def _extract_array_from_response(raw: str) -> list:
    """
    Extract a JSON array from the LLM response.
    Handles both bare array and {"decomposed": [...]} wrapper responses.
    """
    content = raw.strip()

    # Strip markdown fences
    content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.MULTILINE)
    content = re.sub(r"\s*```\s*$", "", content, flags=re.MULTILINE).strip()

    # Try direct array parse
    try:
        result = json.loads(content)
        if isinstance(result, list):
            return result
        # LLM may wrap in a dict — look for any list value
        if isinstance(result, dict):
            for v in result.values():
                if isinstance(v, list):
                    return v
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: extract first [...] block
    arr_match = re.search(r"\[.*\]", content, re.DOTALL)
    if arr_match:
        try:
            result = json.loads(arr_match.group())
            if isinstance(result, list):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    return []


def _validate_structured_claim(item: Any) -> dict | None:
    """
    Return a clean DecomposedClaim dict if *item* is valid, else None.
    Required keys: claim, normalizedQueries.
    """
    if not isinstance(item, dict):
        return None
    claim_text = str(item.get("claim", "")).strip()
    if not claim_text:
        return None
    queries = item.get("normalizedQueries", [])
    if not isinstance(queries, list):
        queries = []
    # Ensure at least one query (fall back to the claim text itself)
    queries = [str(q).strip() for q in queries if str(q).strip()]
    if not queries:
        queries = [claim_text]
    return {
        "claim":             claim_text,
        "subject":           str(item.get("subject", "")).strip(),
        "action":            str(item.get("action",  "")).strip(),
        "object":            str(item.get("object",  "")).strip(),
        "context":           str(item.get("context", "")).strip(),
        "normalizedQueries": queries[:3],
    }


def _fallback_decomposed(claims: list[str]) -> list[dict]:
    """
    Return minimal DecomposedClaim list from raw claim strings
    (used when the LLM call fails).
    """
    result = []
    for c in claims:
        c = c.strip()
        if c:
            result.append({
                "claim":             c,
                "subject":           "",
                "action":            "",
                "object":            "",
                "context":           "",
                "normalizedQueries": [c],
            })
    return result[:5]


def _primary_claim_fallback(primary_claim: str) -> list[dict]:
    """
    Return a structured claim fallback using the primary claim.
    Validates length to prevent overly long/short queries.
    """
    if not primary_claim or not primary_claim.strip():
        return []
        
    pc = primary_claim.strip()
    words = pc.split()
    
    if len(words) < 3:
        logger.warning("Fallback claim too short for retrieval: %s", pc)
        
    if len(words) > 20:
        logger.warning("Primary claim too long, truncating to 20 words for retrieval")
        pc = " ".join(words[:20])
        
    logger.info("Using fallback claim: %s", pc)
    
    return [{
        "claim":             pc,
        "subject":           "",
        "action":            "",
        "object":            "",
        "context":           "",
        "normalizedQueries": generate_queries(pc),
    }]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def decompose_claims(
    raw_claims: list[str],
    context: str = "",
    primary_claim: str = "",
) -> list[dict[str, Any]]:
    """
    Run the 3-stage claim decomposition pipeline on *raw_claims*.

    Parameters
    ----------
    raw_claims  : list[str] — pre-validated claim strings from claim_validator
    context     : str       — original post text (for disambiguation)

    Returns
    -------
    list[DecomposedClaim] — each item has:
        claim, subject, action, object, context, normalizedQueries

    On LLM failure the original claims are returned as minimal fallback dicts
    (no crash, no silent data loss).
    """
    if not raw_claims:
        logger.info("decompose_claims: no claims to decompose — returning empty list")
        return []

    # Deduplicate input
    seen: set[str] = set()
    unique: list[str] = []
    for c in raw_claims:
        key = c.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(c.strip())

    # Build user message
    claims_block = "\n".join(f"{i+1}. {c}" for i, c in enumerate(unique))
    context_note = f"\nOriginal post context: {context[:400]}\n" if context else ""
    user_content = (
        f"{context_note}"
        f"Decompose these claims into structured, event-level form:\n\n"
        f"{claims_block}"
    )

    # Tokens: ~150 per claim output + 50 buffer
    max_tokens = min(1500, 150 * len(unique) + 200)

    logger.info("decompose_claims: calling LLM for %d claim(s)", len(unique))

    try:
        client = get_azure_client()
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]
        raw = await asyncio.to_thread(
            _sync_llm_decompose, client, messages, max_tokens
        )

        logger.info(
            "decompose_claims: LLM response (first 300 chars): %s", raw[:300]
        )

        items = _extract_array_from_response(raw)
        result: list[dict] = []
        for item in items[:5]:
            validated = _validate_structured_claim(item)
            if validated:
                result.append(validated)

        if not result:
            if primary_claim:
                logger.warning(
                    "decompose_claims: LLM returned no valid structured claims — "
                    "falling back to primaryClaim"
                )
                return _primary_claim_fallback(primary_claim)
            else:
                logger.warning(
                    "decompose_claims: LLM returned no valid structured claims — "
                    "falling back to original claims"
                )
                return _fallback_decomposed(unique)

        logger.info(
            "decompose_claims: produced %d structured claim(s): %s",
            len(result),
            [c["claim"] for c in result],
        )
        return result

    except Exception as exc:
        if primary_claim:
            logger.warning(
                "decompose_claims: LLM call failed (%s) — using primaryClaim fallback", exc
            )
            return _primary_claim_fallback(primary_claim)
        
        logger.warning(
            "decompose_claims: LLM call failed (%s) — using fallback", exc
        )
        return _fallback_decomposed(unique)
