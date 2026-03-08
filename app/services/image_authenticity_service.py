"""
TrustLens Phase 4: Image Context & Authenticity Service

Independent service that evaluates image authenticity across four signals:
  1. Perceptual hash reuse detection
  2. Context mismatch analysis
  3. EXIF metadata analysis
  4. AI-generation likelihood (MVP placeholder)

Produces a normalised imageAuthenticityScore (0–100) and structured metadata.

Architecture principles
------------------------
- Pure function: evaluate_image(bytes, str) → dict. No side effects.
- No dependency on LLM output, fact-check module, or domain service.
- All sub-calls are isolated; any failure yields neutral fallback, never crash.
- Designed for asyncio.to_thread() parallelism in Phase 5 — no shared state.

Parallelism readiness (Phase 5)
---------------------------------
  async def evaluate_image_async(image_bytes, claim_context):
      return await asyncio.to_thread(evaluate_image, image_bytes, claim_context)

  # Run all sub-signals concurrently:
  hash_task  = asyncio.to_thread(compute_phash, image_bytes)
  exif_task  = asyncio.to_thread(extract_exif, image_bytes)
  # After hash is known → db lookup can follow, or pre-gather all
  results = await asyncio.gather(hash_task, exif_task)

  Blocking bottlenecks today: compute_phash (~50ms), EXIF (~5ms).
  Caching: TTLCache keyed on SHA-256 of image_bytes — same image served
  multiple times in one session returns instantly.

Failure handling philosophy
-----------------------------
  A corrupted image upload, PIL bug, or missing dependency must NEVER
  crash the credibility pipeline. On any unhandled error, evaluate_image()
  returns the 'neutral' result (score 70) with an explanatory riskFactor.
  70 was chosen as neutral rather than 50 because images are assumed
  authentic until specific evidence says otherwise — over-penalising
  unknown images would unfairly bias the final score.
"""

from typing import Any

from app.utils.hash_utils import compute_phash, is_similar
from app.utils.metadata_utils import extract_exif
from app.utils.image_utils import is_valid_image
from app.services.hash_db import lookup_hash

# ---------------------------------------------------------------------------
# Scoring constants
# ---------------------------------------------------------------------------
BASELINE_SCORE: int = 70          # Images assumed authentic by default

REUSE_PENALTY: int = 25           # Hash match found in known DB
MISMATCH_PENALTY: int = 30        # Reused in a different context (strong signal)
MISSING_METADATA_PENALTY: int = 10  # No EXIF (weak signal — social media strips it)
EDITING_SOFTWARE_PENALTY: int = 15  # Photoshop/GIMP/AI tool in EXIF Software tag
AI_PROBABILITY_PENALTY: int = 20   # AI likelihood > 0.7 (future model hook)

AI_HIGH_THRESHOLD: float = 0.7    # Probability above which AI penalty applies

# Context mismatch: minimum characters of claim context before comparing.
# Short/empty claims cannot meaningfully be compared.
MIN_CONTEXT_LENGTH: int = 10


# ---------------------------------------------------------------------------
# Context mismatch detection
# ---------------------------------------------------------------------------

def _detect_context_mismatch(
    stored_context: str,
    claim_context: str | None,
) -> bool:
    """
    Determine whether the claim context differs meaningfully from the
    stored context of a known (reused) image.

    Why context mismatch is a strong misinformation signal:
      Image reuse by itself may be innocent (illustrative photo, archive image).
      But reuse WITH a different context — e.g., a 2019 flood photo captioned
      as a 2024 event — is the core mechanism of visual misinformation.

    Why this comparison must be cautious:
      Simple string equality would miss genuine matches with paraphrasing.
      We use token overlap heuristics to catch the most egregious cases
      while avoiding false positives from synonym variation.

    Current implementation: checks for absence of key event terms from the
    stored context in the claim context. Simple and deterministic — no LLM
    dependency here by design.

    Phase 5 improvement: Replace with embedding cosine similarity
      (sentence-transformers, or Azure OpenAI embeddings).
    """
    if not claim_context or len(claim_context.strip()) < MIN_CONTEXT_LENGTH:
        # Cannot assess mismatch without meaningful context
        return False

    stored_lower = stored_context.lower()
    claim_lower = claim_context.lower()

    # Extract meaningful keywords from stored context (year, country, event type)
    # Simple heuristic: split and filter stop words
    stop_words = {
        "a", "an", "the", "in", "at", "on", "of", "and", "or",
        "is", "was", "were", "has", "have", "for", "to", "by",
        "from", "with", "this", "that", "it", "its",
    }
    stored_tokens = {
        t for t in stored_lower.split()
        if len(t) > 3 and t not in stop_words
    }

    if not stored_tokens:
        return False

    # Count how many stored tokens appear in the claim
    matches = sum(1 for token in stored_tokens if token in claim_lower)
    overlap_ratio = matches / len(stored_tokens)

    # If fewer than 30% of the stored context tokens appear in the claim → mismatch
    return overlap_ratio < 0.30


# ---------------------------------------------------------------------------
# AI generation placeholder
# ---------------------------------------------------------------------------

def _estimate_ai_probability(image_bytes: bytes) -> float:
    """
    Estimate the probability that the image was AI-generated.

    MVP placeholder: returns a safe low default (0.05) to avoid false positives.

    Why synthetic detection is probabilistic:
      Current state-of-the-art models (CNNDetection, UniversalFakeDetect) have
      ~90% accuracy on their training distribution but drop to 60-70% on
      out-of-distribution generators. False positives (flagging real photos as
      AI) are more harmful than false negatives in a fact-checking context.

    How to extend:
      Replace this function with a call to a real classifier:
        prob = ai_detector_model.predict(image_bytes)
      The rest of the scoring pipeline is already wired for this value.

    Why not integrate directly yet:
      Adding a neural model requires GPU availability, model download (~500MB),
      and latency management — all Phase 5 concerns. The scoring hook exists now.
    """
    # Future: call local ONNX model or async external service
    return 0.05


# ---------------------------------------------------------------------------
# Score normalisation
# ---------------------------------------------------------------------------

def _compute_score(
    *,
    hash_matched: bool,
    context_mismatch: bool,
    metadata_present: bool,
    editing_software_detected: bool,
    ai_probability: float,
) -> tuple[int, list[str]]:
    """
    Apply incremental penalties to BASELINE_SCORE and collect risk factors.

    Why baseline is 70 (not 50):
      Domain reputation starts at 50 because unknown domains are genuinely
      uncertain. Images in circulation are more commonly authentic — starting
      at 70 avoids mass over-penalisation of images that simply lack EXIF
      (the majority, because social platforms strip it universally).

    Why incremental (not absolute) scoring:
      Signals compose naturally. A reused image (−25) that also has a context
      mismatch (−30) = 15/100 — very risky. But a reused image used correctly
      in the same context (hash match, no mismatch) = 45/100 — suspicious but
      not definitively misleading. Absolute scoring would collapse this nuance.

    Why clamping:
      The formula must never produce scores outside [0, 100] because they feed
      directly into the weighted final formula where instability propagates.
    """
    score = BASELINE_SCORE
    risk_factors: list[str] = []

    if hash_matched:
        score -= REUSE_PENALTY
        risk_factors.append("Image matches a known photograph from a previous event")

    if context_mismatch:
        score -= MISMATCH_PENALTY
        risk_factors.append(
            "Context mismatch: image appears to be reused with a different narrative"
        )

    if not metadata_present:
        score -= MISSING_METADATA_PENALTY
        risk_factors.append(
            "No EXIF metadata found — image may have been stripped or screenshot"
        )

    if editing_software_detected:
        score -= EDITING_SOFTWARE_PENALTY
        risk_factors.append(
            "Editing or AI generation software detected in image metadata"
        )

    if ai_probability > AI_HIGH_THRESHOLD:
        score -= AI_PROBABILITY_PENALTY
        risk_factors.append(
            f"High AI-generation likelihood ({ai_probability:.0%})"
        )

    return max(0, min(100, score)), risk_factors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_image(
    image_bytes: bytes | None,
    claim_context: str | None = None,
) -> dict[str, Any]:
    """
    Evaluate image authenticity from raw bytes and optional claim context.

    Parameters
    ----------
    image_bytes : bytes | None
        Raw image binary (JPEG, PNG, WebP, etc.). None/empty returns neutral.
    claim_context : str | None
        The primary claim associated with the image (e.g., extracted from
        the article or LLM primaryClaim). Used for context mismatch detection.

    Returns
    -------
    dict with keys:
        imageAuthenticityScore   : int        — 0–100 normalised score
        hashMatched              : bool       — pHash found in known DB
        matchedContext           : str | None — context from matched DB entry
        matchedEventDate         : str | None — verified date from DB entry
        contextMismatch          : bool       — reuse in wrong context
        aiGeneratedLikelihood    : float      — probability 0.0–1.0
        metadataPresent          : bool       — EXIF block found
        cameraMake               : str        — e.g. "Apple"
        cameraModel              : str        — e.g. "iPhone 14 Pro"
        editingSoftwareDetected  : bool       — suspicious software in EXIF
        riskFactors              : list[str]  — human-readable explanations

    Graceful failure:
        Any unexpected error returns the neutral default (score 70) with a
        riskFactor explaining the fallback. The pipeline never crashes.
    """
    neutral_result: dict[str, Any] = {
        "imageAuthenticityScore": BASELINE_SCORE,
        "hashMatched": False,
        "matchedContext": None,
        "matchedEventDate": None,
        "contextMismatch": False,
        "aiGeneratedLikelihood": 0.05,
        "metadataPresent": False,
        "cameraMake": "",
        "cameraModel": "",
        "editingSoftwareDetected": False,
        "riskFactors": [],
    }

    # Guard: empty / invalid input → neutral
    if not image_bytes or not isinstance(image_bytes, bytes):
        neutral_result["riskFactors"] = ["No image data provided for authenticity check"]
        return neutral_result

    # Guard: corrupt / unrecognised format → neutral
    if not is_valid_image(image_bytes):
        neutral_result["riskFactors"] = ["Image data is corrupt or in an unrecognised format"]
        return neutral_result

    try:
        # --- Signal 1: Perceptual hash + DB lookup ---
        phash_str = compute_phash(image_bytes)
        db_match = lookup_hash(phash_str) if phash_str else None

        hash_matched = db_match is not None
        matched_context = db_match.get("context") if db_match else None
        matched_event_date = db_match.get("verifiedEventDate") if db_match else None

        # --- Signal 2: Context mismatch ---
        context_mismatch = (
            hash_matched
            and matched_context is not None
            and _detect_context_mismatch(matched_context, claim_context)
        )

        # --- Signal 3: EXIF metadata ---
        exif = extract_exif(image_bytes)
        metadata_present = exif.get("metadataPresent", False)
        editing_detected = exif.get("editingSoftwareDetected", False)
        camera_make = exif.get("cameraMake", "")
        camera_model = exif.get("cameraModel", "")

        # --- Signal 4: AI probability (MVP placeholder) ---
        ai_prob = _estimate_ai_probability(image_bytes)

        # --- Score normalisation ---
        score, risk_factors = _compute_score(
            hash_matched=hash_matched,
            context_mismatch=context_mismatch,
            metadata_present=metadata_present,
            editing_software_detected=editing_detected,
            ai_probability=ai_prob,
        )

        return {
            "imageAuthenticityScore": score,
            "hashMatched": hash_matched,
            "matchedContext": matched_context,
            "matchedEventDate": matched_event_date,
            "contextMismatch": context_mismatch,
            "aiGeneratedLikelihood": round(ai_prob, 4),
            "metadataPresent": metadata_present,
            "cameraMake": camera_make,
            "cameraModel": camera_model,
            "editingSoftwareDetected": editing_detected,
            "riskFactors": risk_factors,
        }

    except Exception as exc:
        # Belt-and-suspenders: unexpected errors must not propagate
        print(f"⚠️ Image authenticity evaluation error: {exc}")
        neutral_result["riskFactors"] = [
            f"Image authenticity check encountered an error — neutral score applied"
        ]
        return neutral_result
