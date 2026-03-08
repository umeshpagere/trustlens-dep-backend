"""
TrustLens Phase 1: Semantic Credibility Analysis (Phase 6 async refactor)

The Azure OpenAI Python SDK is synchronous. Rather than blocking the event loop,
both LLM functions are wrapped with asyncio.to_thread() so the SDK call runs
in a worker thread while the event loop remains available for other coroutines.

Why asyncio.to_thread() instead of a native async Azure SDK:
  The installed openai/azure-openai SDK version uses blocking httpx under the
  hood. asyncio.to_thread() is the standard pattern for running synchronous
  third-party libraries from async code — it requires zero SDK changes and
  is trivially reversible if the SDK adds native async support later.

Why CPU-only helpers (_sanitize_user_input, _parse_and_validate_semantic_response,
detect_image_mime_type) stay synchronous:
  They do no I/O. Making them async adds coroutine overhead with zero concurrency
  benefit. Pure CPU logic MUST stay synchronous.
"""

import json
import re
import base64
import asyncio
from app.config.azure import get_azure_client
from app.config.settings import Config

# --- Shared constants / helpers (unchanged) ---

SEMANTIC_SYSTEM_PROMPT = """You are a fact-checking analyst. Your role is to extract verifiable factual claims from social media content and assess how checkable and evidence-based they are. The goal is to help users get correct information, not to judge tone or sentiment.

== CLAIM EXTRACTION RULES (STRICT) ==
A valid claim MUST contain:
  • A clear subject: a person, organisation, government, event, or object
  • A specific action or event involving that subject
  • Optional: numbers, dates, or locations that make it more verifiable

Do NOT include:
  • Opinions or emotional reactions ("This is terrible", "We must act")
  • Vague descriptions with no subject or no action
  • Sentences that cannot be looked up in a public record or database

IF a statement does not contain a clear subject AND a clear action/event, do NOT include it in keyClaims.

== EXAMPLES ==
VALID claims (include these):
  ✓ "WHO declared a global health emergency."
  ✓ "Government banned bank withdrawals."
  ✓ "Police arrested protesters in Paris."
  ✓ "NASA confirmed water was found on Mars."
  ✓ "The unemployment rate rose to 8.4% in July 2020."

INVALID claims (exclude these):
  ✗ "People are suffering."  ← no specific subject or action
  ✗ "This situation is terrible."  ← opinion, not verifiable
  ✗ "Something big is happening."  ← no subject, no verifiable event
  ✗ "The situation is getting worse."  ← vague, no subject or action
  ✗ "Wake up and see the truth."  ← not a factual claim

== OTHER RULES ==
- PRIORITIZE FACTS: statistics, event descriptions, attributions, cause-effect statements.
- Do NOT base semanticScore on emotional language or tone. Score based on claim clarity and verifiability.
- primaryClaim = the single most important FACTUAL claim a fact-checker should look up first.
- keyClaims: up to 5 distinct verifiable claims. Never include opinions or vague assertions.
- evidenceStrength: Weak = no sources; Moderate = some attribution or named entities; Strong = verifiable sources or named authorities.
- Use probabilistic language in reasoningSummary. Do NOT declare content "true" or "false".
- Output ONLY valid JSON. No markdown, no code fences, no explanatory text."""

SEMANTIC_USER_PROMPT_TEMPLATE = """Extract factual claims and assess verifiability of the following post. Return a JSON object with these exact keys. Output nothing else.

Required JSON schema:
{
  "semanticScore": <number 0-100: 100 = clear verifiable claims with strong evidence/sources; 50 = mixed or unclear claims; 0 = no checkable claims>,
  "confidenceScore": <number 0-1, your confidence in this assessment>,
  "primaryClaim": "<THE single most important factual claim to fact-check — must have a subject + action, concrete and checkable>",
  "keyClaims": ["<up to 5 specific verifiable claims — each must have a subject + action; omit vague or emotional statements>"],
  "manipulationIndicators": ["<string>", ...],
  "riskFactors": ["<string>", ...],
  "evidenceStrength": "<one of: Weak | Moderate | Strong>",
  "reasoningSummary": "<short paragraph: what factual claims were found, how verifiable they are — NOT sentiment or tone>"
}

keyClaims rules:
  - Maximum 5 claims
  - Each claim must have a clear subject (person/org/government) AND a specific action or event
  - If a statement has no subject or no action, do NOT include it
  - Omit opinions, emotional reactions, and vague sentences entirely

primaryClaim: The one claim a fact-checker should verify first (the main statistic, event, or attribution).
evidence Strength: Weak = no sources; Moderate = some attribution; Strong = verifiable sources or named authorities.
semanticScore: Base on VERIFIABILITY and EVIDENCE only — NOT on emotional language or urgency.

---POST TO ANALYZE---
{text}
---END POST---"""

VALID_EVIDENCE_STRENGTH = frozenset({"Weak", "Moderate", "Strong"})
MAX_INPUT_LENGTH = 8000


def _sanitize_user_input(text: str) -> str:
    """Sanitize user input to prevent prompt injection and preserve JSON safety."""
    if not text or not isinstance(text, str):
        return ""
    sanitized = text.strip()
    if len(sanitized) > MAX_INPUT_LENGTH:
        sanitized = sanitized[:MAX_INPUT_LENGTH] + "... [truncated]"
    sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", sanitized)
    return sanitized


def _extract_json(raw: str) -> dict:
    """
    Robust multi-strategy JSON extractor for LLM responses.

    Azure OpenAI sometimes returns:
      - Clean JSON object  ← happy path
      - JSON wrapped in ```json ... ``` fences (with optional leading newline)
      - A JSON fragment starting with \n  "key" (missing outer braces)
      - JSON embedded within explanatory text

    Strategy order:
      1. Strip markdown fences (tolerant of leading whitespace) → try json.loads
      2. Try json.loads on the raw content directly
      3. Extract first {...} block from the content → try json.loads
      4. Wrap content in braces if it looks like bare key-value pairs → try json.loads
    """
    content = raw.strip()

    # Strategy 1: strip markdown code fences (handle leading newlines before ```)
    fence_stripped = re.sub(r"^```(?:json)?\s*", "", content, flags=re.MULTILINE)
    fence_stripped = re.sub(r"\s*```\s*$", "", fence_stripped, flags=re.MULTILINE).strip()
    try:
        result = json.loads(fence_stripped)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: try the raw content directly
    try:
        result = json.loads(content)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 3: extract first {...} block (handles JSON embedded in text)
    brace_match = re.search(r'\{.*\}', content, re.DOTALL)
    if brace_match:
        try:
            result = json.loads(brace_match.group())
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 4: content is bare key-value pairs (missing outer braces)
    # e.g. Azure returned: \n  "semanticScore": 75,\n  "confidenceScore": 0.8, ...
    if '"semanticScore"' in content or '"primaryClaim"' in content:
        try:
            wrapped = "{" + content.strip().rstrip(",") + "}"
            result = json.loads(wrapped)
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    raise ValueError(f"Could not extract JSON from LLM response. Raw (first 200 chars): {raw[:200]!r}")


def _parse_and_validate_semantic_response(raw: str) -> dict:
    """Parse raw LLM response and validate against semantic schema."""
    parsed = _extract_json(raw)

    required_keys = [
        "semanticScore", "confidenceScore", "primaryClaim",
        "manipulationIndicators", "riskFactors", "evidenceStrength", "reasoningSummary",
    ]
    missing = [k for k in required_keys if k not in parsed]
    if missing:
        raise ValueError(f"Invalid response: missing keys: {missing}")

    for key in ("manipulationIndicators", "riskFactors", "keyClaims"):
        val = parsed.get(key)
        if not isinstance(val, list):
            parsed[key] = []
        else:
            parsed[key] = [str(x) for x in val if x][:5]  # cap keyClaims at 5

    try:
        score = float(parsed.get("semanticScore", 50))
    except (TypeError, ValueError):
        score = 50.0
    parsed["semanticScore"] = max(0, min(100, round(score)))

    try:
        conf = float(parsed.get("confidenceScore", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    parsed["confidenceScore"] = max(0.0, min(1.0, round(conf, 2)))

    ev = str(parsed.get("evidenceStrength", "Weak")).strip()
    parsed["evidenceStrength"] = ev if ev in VALID_EVIDENCE_STRENGTH else "Weak"
    parsed["primaryClaim"] = str(parsed.get("primaryClaim", ""))[:500]
    parsed["reasoningSummary"] = str(parsed.get("reasoningSummary", ""))[:1000]
    return parsed


def detect_image_mime_type(image_bytes: bytes) -> str:
    if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
        return "image/png"
    elif image_bytes[:2] == b'\xff\xd8':
        return "image/jpeg"
    elif image_bytes[:6] in (b'GIF87a', b'GIF89a'):
        return "image/gif"
    elif image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP':
        return "image/webp"
    else:
        return "image/jpeg"


# ---------------------------------------------------------------------------
# Async public API
# ---------------------------------------------------------------------------

async def analyze_text_with_llm(text: str) -> dict:
    """
    Async: Analyze social media text for credibility risk using Azure OpenAI.

    The synchronous SDK call runs in asyncio.to_thread() so it does not block
    the event loop. The rest of the function (input validation, response parsing)
    is CPU-only and runs on the event loop thread directly.
    """
    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        raise ValueError("Text input is required")

    sanitized_text = _sanitize_user_input(text)
    if not sanitized_text:
        raise ValueError("Text input is required")

    try:
        client = get_azure_client()
        user_content = SEMANTIC_USER_PROMPT_TEMPLATE.replace("{text}", sanitized_text)

        messages = [
            {"role": "system", "content": SEMANTIC_SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]

        # The Azure SDK call is synchronous — run it in a thread pool
        # so the event loop is not blocked while waiting for the network.
        def _sdk_call() -> str:
            response = client.chat.completions.create(
                model=Config.AZURE_OPENAI_DEPLOYMENT,
                messages=messages,
                temperature=0.2,
                max_tokens=800,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content

        raw_content = await asyncio.to_thread(_sdk_call)

        if not raw_content:
            raise ValueError("No response content from Azure OpenAI")

        print(f"🔬 Raw Azure text response (first 300 chars): {raw_content[:300]!r}")
        return _parse_and_validate_semantic_response(raw_content)

    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM response as JSON: {e}")
    except ValueError:
        raise
    except Exception as e:
        print(f"Azure OpenAI text analysis failed: {e}")
        raise ValueError(f"LLM text analysis failed: {e}")


async def analyze_image_with_llm(image_bytes: bytes, accompanying_text: str = "") -> dict:
    """
    Async: Analyze image for misinformation risk using Azure OpenAI vision.
    SDK call wrapped in asyncio.to_thread() — same pattern as text analysis.
    """
    if not image_bytes:
        raise ValueError("Image bytes are required")

    try:
        mime_type = detect_image_mime_type(image_bytes)
        print(f"🔍 Detected image MIME type: {mime_type}")
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        client = get_azure_client()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert AI multi-layered image analyst, forensic investigator, "
                    "and fact-checker. Your mission is to perform a deep-dive extraction and "
                    "verification of every detail in an image to identify misinformation, "
                    "AI-generation, or manipulation.\n\n"
                    "Your analysis MUST cover these layers:\n"
                    "1. FULL TEXT EXTRACTION & VERIFICATION:\n"
                    "   - Extract ALL visible text, including small print, signs, and background text.\n"
                    "   - Verify every claim found in the text against factual knowledge.\n\n"
                    "2. VISUAL CONTENT & CONVEYED MESSAGE:\n"
                    "   - Provide a granular description of all subjects, objects, settings, and actions.\n"
                    "   - Decode the underlying narrative.\n\n"
                    "3. AI GENERATION & MANIPULATION FORENSICS:\n"
                    "   - Look for AI artifacts: distorted hands/fingers, inconsistent lighting, etc.\n\n"
                    "4. VERACITY & TRUST VERDICT:\n"
                    "   - Provide a definitive assessment based on visual evidence.\n\n"
                    "Respond ONLY in valid JSON format. No markdown, no code blocks, no extra text."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Perform a rigorous multi-layered analysis of this image.\n\n"
                            "Return JSON in this exact format:\n"
                            "{\n"
                            '  "riskLevel": "low" | "medium" | "high",\n'
                            '  "credibilityScore": number (0-100),\n'
                            '  "verdict": "Reliable" | "Questionable" | "High Risk",\n'
                            '  "extractedText": "all text extracted from the image",\n'
                            '  "textVerification": "factual verification of extracted text claims",\n'
                            '  "imageContent": "granular description of what the image depicts",\n'
                            '  "conveyedMessage": "deep analysis of the image intent",\n'
                            '  "veracityCheck": "comprehensive explanation of veracity",\n'
                            '  "visualRedFlags": ["list of specific visual anomalies"],\n'
                            '  "explanation": "concise summary of why this verdict was reached",\n'
                            '  "aiGeneratedProbability": number (0-100)\n'
                            "}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                    },
                ],
            },
        ]

        if accompanying_text:
            messages[1]["content"][0]["text"] += (
                f"\n\nThe user shared this image alongside the following text: '{accompanying_text}'.\n"
                f"CROSS-REFERENCE VERIFICATION: Cross-reference the visual evidence in the image against this text.\n"
                f"Determine if the image genuinely supports the user's text, or if the text drastically misrepresents the image context.\n"
                f"Include your findings within the 'veracityCheck', 'conveyedMessage', and 'explanation' fields."
            )

        def _sdk_call() -> str:
            response = client.chat.completions.create(
                model=Config.AZURE_OPENAI_DEPLOYMENT,
                messages=messages,
                temperature=0.1,
                max_tokens=1500,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content

        content = await asyncio.to_thread(_sdk_call)

        if not content:
            raise ValueError("No response content from Azure OpenAI")

        result = json.loads(content.strip())
        if not all(k in result for k in ["riskLevel", "credibilityScore", "verdict", "explanation"]):
            raise ValueError("Invalid response format from Azure OpenAI")
        if not isinstance(result.get("visualRedFlags"), list):
            result["visualRedFlags"] = []
        return result

    except Exception as e:
        print(f"Azure OpenAI image analysis failed: {e}")
        raise ValueError(f"LLM image analysis failed: {e}")
