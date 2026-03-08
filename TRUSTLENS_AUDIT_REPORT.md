# TrustLens Full Debugging & Architecture Audit Report

**Date:** 2025  
**Scope:** Backend (`backend/`), Flask + Hypercorn, async pipeline, external APIs, video processing, scoring engine.

**Fixes applied:** Evidence verifier async (to_thread), Sightengine safe type access, video pipeline OCR metadata + duplicate imports + awaited hash storage, Deepgram defensive parsing, missing deps + Config vars, MongoDB TLS configurable, News/Wikipedia/Vision/Sightengine use Config where applicable.

---

## 1. Architecture Summary

### 1.1 Project Layout

- **Entry:** `run.py` — Hypercorn ASGI server (required for async routes).
- **App:** `app/main.py` — Flask app factory, CORS, blueprints, global exception handler.
- **Route:** `app/routes/analyze.py` — Single async POST handler; orchestrates image download → text LLM → (optional) video pipeline → evidence aggregation → `compute_full_credibility`.
- **Services:**  
  - **LLM:** `llm_analysis.py` (Azure OpenAI text/image; SDK calls wrapped in `asyncio.to_thread`).  
  - **Fact-check:** `fact_check_service.py` (Google Fact Check API via httpx async).  
  - **Domain:** `domain_reputation_service.py` (WHOIS, HTTPS, whitelist/blacklist; blocking parts in `asyncio.to_thread`).  
  - **Image auth:** `image_authenticity_service.py` (pHash, hash DB, EXIF; sync, run via `asyncio.to_thread`).  
  - **Credibility:** `credibility_engine.py` — Runs fact-check, domain, image in parallel with `asyncio.gather`, then synchronous weighted scoring + confidence.  
  - **Evidence:** `evidence_aggregator.py` (fact-check + Wikipedia + News in parallel), `evidence_verifier.py` (LLM over ranked evidence).  
  - **Video:** `video_pipeline.py` (transcript via yt-dlp/Deepgram, frame extraction via ffmpeg, OCR via Azure Vision, AI detection via Sightengine, context reuse via frame hashes).
- **Storage:** `analysis_storage_service.py` — MongoDB by content hash; cache reuse for text/image/video.
- **Config:** `app/config/settings.py` — env-only (no NEWS_API_KEY, AZURE_VISION_*, SIGHTENGINE_* in Config class).

### 1.2 Request Flow

1. **Validate** — Pydantic `AnalyzeRequest` (text, imageUrl, videoUrl; at least one required).
2. **Image** — If `imageUrl`: async `download_image` → hash → cache lookup or metadata/tracing/LLM image analysis → store.
3. **Text** — If `text`: hash → cache lookup or async `analyze_text_with_llm` → build `text_analysis` (semantic, credibilityScore, verdict, etc.) → if `primaryClaim`, run `aggregate_evidence` + `verify_claim_with_evidence` → store.
4. **Video** — If `videoUrl`: hash → cache lookup or `process_video_text` (transcript + frames + OCR + AI detection + context reuse) → `analyze_video_with_llm` (in thread) → optional evidence verification on first claim → store.
5. **Credibility** — `await compute_full_credibility(text_analysis, image_analysis, video_analysis, source_url, image_bytes)`:
   - Fact-check (async), domain (async), image authenticity (in thread) via `asyncio.gather(return_exceptions=True)`.
   - Normalise fact-check response, then `compute_weighted_final_result` (sync) with boost/penalties and `calculate_confidence`.
6. **Response** — JSON with `textAnalysis`, `imageAnalysis`, `videoAnalysis`, `finalResult` (componentScores, factCheckDetails, finalScore, riskLevel, confidence, etc.), `processingMs`, hash/reused.

### 1.3 Heavy Processing

- **Azure OpenAI:** Text and image semantic analysis (in thread); video LLM (in thread); evidence verifier LLM (currently blocking, see Critical Bugs).
- **Google Fact Check:** Async httpx in gather.
- **Domain:** WHOIS + HTTPS in thread pool inside `evaluate_domain`.
- **Image authenticity:** pHash, EXIF, hash DB (sync, in thread).
- **Video:** yt-dlp (download/subtitles/audio), ffmpeg frame extraction, Azure Vision OCR per frame (parallel), Sightengine AI detection per selected frame, frame hashing and context DB — most of this is either in thread or in async pipeline with `asyncio.to_thread` / `gather`.

### 1.4 Result Aggregation

- **Credibility:** `compute_weighted_final_result` uses fixed weights (semantic 0.35, factCheck 0.20, sourceReputation 0.15, imageAuthenticity 0.15, domainTrust 0.10, knowledgeSupport 0.05), neutral defaults per component, boost layer (cap +15), penalties (video AI, context reuse), final clamp 0–95. Confidence from `calculate_confidence` (coverage, agreement, evidence strength).
- **Evidence:** Fact-check + Wikipedia + News in parallel → rank/filter → single LLM call for `knowledgeSupportScore` (0–1) and verdict.

---

## 2. Critical Bugs

### 2.1 Evidence verifier blocks event loop (Azure SDK call)

| Item | Detail |
|------|--------|
| **File** | `backend/app/services/evidence/evidence_verifier.py` |
| **Line** | 149–155 |
| **Code** | `response = client.chat.completions.create(...)` inside `async def verify_claim_with_evidence` |
| **Problem** | The function is async but performs a synchronous Azure OpenAI call. Under Hypercorn this blocks the event loop for the duration of the LLM request (several seconds), stalling all other requests. |
| **Suggested fix** | Run the SDK call in a thread, e.g. `response = await asyncio.to_thread(_sync_verify, client, messages, ...)` where `_sync_verify` performs `client.chat.completions.create(...)` and returns the result. |
| **Impact** | High — every text (and video claim) verification blocks the server; concurrency is lost. |

### 2.2 Sightengine response: `data["type"]` may be a string

| Item | Detail |
|------|--------|
| **File** | `backend/app/services/video/video_ai_detector.py` |
| **Line** | 65–66 |
| **Code** | `if "type" in data and data.get("type", {}).get("ai_generated") is not None:` then `return float(data["type"].get("ai_generated") or 0.0)` |
| **Problem** | API may return `"type": "genai"` (string). Then `data.get("type", {})` is the string `"genai"`, and `.get("ai_generated")` raises `AttributeError`. |
| **Suggested fix** | Guard on type: `t = data.get("type"); return float(t.get("ai_generated", 0.0)) if isinstance(t, dict) else 0.0` (and keep existing top-level / `genai` handling). |
| **Impact** | Medium — video AI detection can crash for some Sightengine response shapes. |

### 2.3 Video pipeline overwrites OCR metadata

| Item | Detail |
|------|--------|
| **File** | `backend/app/services/video/video_pipeline.py` |
| **Line** | 112, 131, 152 |
| **Code** | `capped_lines, metadata = aggregate_ocr_text(all_frame_texts)` then later `metadata = {"video_id": ..., "platform": ..., "source_url": ...}` and `"ocrMetadata": metadata` in return. |
| **Problem** | The second assignment overwrites `metadata`, so the returned `ocrMetadata` is the video/hash metadata, not the OCR aggregation metadata from `aggregate_ocr_text`. |
| **Suggested fix** | Use a different variable for hash storage, e.g. `hash_metadata = {...}` and keep `metadata` from `aggregate_ocr_text` for `"ocrMetadata": metadata`. |
| **Impact** | Low — wrong metadata in response; no crash. |

### 2.4 Duplicate imports in video pipeline

| Item | Detail |
|------|--------|
| **File** | `backend/app/services/video/video_pipeline.py` |
| **Line** | 9–10 and 16–17 |
| **Code** | `from app.services.video.video_text_aggregator import aggregate_ocr_text` and `from app.services.video_analysis import extract_transcript` each appear twice. |
| **Problem** | Redundant; no runtime bug but noisy and error-prone for future edits. |
| **Suggested fix** | Remove the duplicate import lines. |
| **Impact** | Low — code quality. |

### 2.5 Deepgram response structure assumed

| Item | Detail |
|------|--------|
| **File** | `backend/app/services/video_analysis.py` |
| **Line** | 339 |
| **Code** | `transcript_text = response.results.channels[0].alternatives[0].transcript` |
| **Problem** | If the API returns a different structure (e.g. no channels/alternatives, or empty list), this raises `AttributeError` or `IndexError`. |
| **Suggested fix** | Defensive access: e.g. `channels = getattr(response.results, "channels", None) or []; alt = (channels[0].alternatives or [None])[0] if channels else None; transcript_text = getattr(alt, "transcript", None) or ""` and then handle empty transcript. |
| **Impact** | Medium — Tier 2 video transcription can crash on unexpected Deepgram response. |

---

## 3. API Integration Issues

### 3.1 Timeouts and retries

| API | Timeout | Retry | Notes |
|-----|--------|-------|--------|
| Google Fact Check | 8 s (httpx) | No | fact_check_service — add optional retry (e.g. 1 retry on 5xx/timeout). |
| Azure OpenAI | SDK default | No | LLM and evidence verifier — consider explicit timeout and retry in wrapper. |
| News API | 8 s (httpx) | No | news_service — sync `httpx.Client`; fine for asyncio.to_thread usage. |
| Domain WHOIS/HTTPS | 8 s / 3 s | No | domain_reputation_service — already in thread; optional retry for WHOIS. |
| Image download | 5 s | No | fetch_image — optional single retry on timeout. |
| Deepgram | 120 s (client options) | No | video_analysis — long timeout; consider retry on transient failure. |
| Sightengine | 30 s (requests) | No | video_ai_detector — in thread; consider retry. |

**Recommendation:** Add a small retry layer (e.g. 1–2 retries with backoff) for idempotent GETs and for LLM/transcription calls where appropriate; keep timeouts as-is or document them in config.

### 3.2 Error handling

- **Fact-check:** Catches httpx and generic exceptions, returns `{"claims": []}`; no raise — good.
- **Domain:** WHOIS/HTTPS failures yield `None`/`False` and neutral score — good.
- **Image auth:** Top-level try/except returns neutral result — good.
- **Evidence verifier:** On exception returns 0.5 / "uncertain" — good; still need to fix blocking call (see 2.1).
- **News:** On exception returns `[]`; logs — good.
- **Wikipedia:** Wrapped in try/except, returns None — good.
- **Deepgram:** Exceptions caught in `transcribe_audio_with_deepgram`; safe fallback — good once response parsing is hardened (2.5).

### 3.3 Missing / inconsistent config

- **NEWS_API_KEY:** Read via `os.getenv("NEWS_API_KEY")` in news_service; not in `Config` in settings. Consider adding to Config for consistency.
- **AZURE_VISION_KEY / AZURE_VISION_ENDPOINT:** Used in video_ocr_service; not in Config. Optional but useful for documentation and validation.
- **SIGHTENGINE_API_USER / SIGHTENGINE_API_SECRET:** Env-only; not in Config.
- **Wikipedia:** Uses `wikipedia` package; not listed in requirements.txt. Add `wikipedia` (and optionally version) to avoid ImportError in evidence aggregation.
- **Video OCR:** Uses `azure-cognitiveservices-vision-computervision` and `msrest`; not in requirements.txt. Add them (and ffmpeg-python if not system ffmpeg) for video pipeline.

---

## 4. Performance Bottlenecks

### 4.1 Sequential work that could be parallelised

- **Route:** Image download → then text LLM → then (if video) video pipeline → then credibility. Text and image paths are independent after download; currently text runs after image. Could run “text LLM” and “image analysis” (metadata + tracing + LLM image) in parallel when both text and imageUrl are present, with the caveat that credibility’s fact-check needs `primaryClaim` from text. So: start image download and text hash; if imageUrl, run download then image branch; if text, run text branch; wait for both; then run credibility (which already runs fact-check, domain, image auth in parallel). So only a small gain by overlapping “text LLM” with “image processing” (after download).
- **Evidence:** Fact-check, Wikipedia, News already run in parallel in `aggregate_evidence` — good.

### 4.2 Redundant or heavy work

- **Video:** Up to 30 frames extracted; OCR on all 30 in parallel; AI detection on up to 8. Consider lowering max_frames (e.g. 15–20) or more aggressive sampling to reduce latency and cost without losing much signal.
- **Evidence verifier:** Called for text (if primaryClaim) and again for video (if first claim). Two separate LLM calls; could be batched or skipped for video when text already ran with same claim (optimisation, not bug).

### 4.3 Caching

- **MongoDB:** Hash-based cache for text/image/video — good; avoids repeated LLM and heavy work.
- **Domain:** No cache; WHOIS/HTTPS every time. Adding a TTL cache (e.g. 1 hour) keyed by domain would cut repeated requests.
- **Image authenticity:** Hash DB is in-memory; no cross-process cache. Acceptable for MVP.

---

## 5. Async / Concurrency Problems

### 5.1 Blocking in async context

- **evidence_verifier (2.1):** Blocking Azure call inside async function — must be moved to `asyncio.to_thread`.
- **LLM text/image:** Correctly use `asyncio.to_thread` for SDK calls — good.
- **Video:** `analyze_video_with_llm` is sync and called via `asyncio.to_thread` in the route — good.
- **Domain:** `_get_domain_age_days` and `_check_https` run in `asyncio.to_thread` — good.
- **Image authenticity:** `evaluate_image` run via `asyncio.to_thread` in credibility_engine — good.

### 5.2 Fire-and-forget task

- **video_pipeline.py** line 144: `asyncio.create_task(_store_hashes())` — task is not awaited. If the task raises, the exception is not seen by the caller. Prefer awaiting it or attaching a done callback that logs errors (or use a small wrapper that logs and swallows).

### 5.3 Thread safety

- **MongoDB:** PyMongo client is shared; documented as thread-safe for concurrent operations — OK.
- **Azure client:** Global singleton in azure.py; used from multiple threads via to_thread — OpenAI SDK is generally thread-safe for separate calls.
- **Hash DB:** In-memory list, read-only at runtime — safe.

---

## 6. Database Optimization

### 6.1 MongoDB

- **Index:** Unique index on `hash` is created in `_get_collection()` — good for lookups.
- **Queries:** Only `find_one({"hash": ...})` and `replace_one({"hash": ...}, document, upsert=True)` — appropriate.
- **TLS:** `tlsAllowInvalidCertificates: True` is a security risk; use only for local/dev and document; production should use valid certs.

### 6.2 Caching effectiveness

- Prevents repeated LLM for same text/image/video hash — good.
- Evidence (fact-check, Wikipedia, news) is not keyed by claim hash; every new primaryClaim triggers API calls. Optional improvement: cache evidence by normalised claim key (e.g. hash of cleaned claim) with TTL.

---

## 7. Security Risks

### 7.1 API keys and config

- Keys are read from environment; no keys in code — good.
- `.env` should remain gitignored; ensure production uses env vars or a secrets manager, not a committed .env file.

### 7.2 Input validation

- **Route:** Pydantic validates type/length/URL for text, imageUrl, videoUrl — good.
- **LLM:** User text sanitised (length, control chars) in llm_analysis — good.
- **Fact-check:** Claim sanitised (length, control chars) in fact_check_service — good.
- **URLs:** imageUrl/videoUrl must start with http(s) — good. Optional: allowlist of hostnames for image/video to reduce SSRF risk.

### 7.3 Injection

- User text is delimited in prompts (e.g. `---POST TO ANALYZE---`) — good.
- No raw user input in MongoDB queries (hash only) — good.

### 7.4 TLS

- MongoDB: `tlsAllowInvalidCertificates: True` — acceptable only for local/debug; remove or make configurable for production.

---

## 8. Code Quality Improvements

- **Duplicate imports:** video_pipeline.py (see 2.4).
- **Docstrings:** Most services are well documented; keep pattern and add short docstrings where missing (e.g. evidence_ranker, video_context_detector).
- **Logging:** Many `print()` calls; consider `logging` with levels (e.g. info for flow, warning for fallbacks, error for failures) for production and filtering.
- **Constants:** Centralise timeouts and limits (e.g. in config or a constants module) instead of scattering 8, 5, 30, 8000 across files.
- **Type hints:** Already used in many places; complete for public function signatures and return types where missing.

---

## 9. Recommended Refactors

### 9.1 Evidence verifier non-blocking

```python
# evidence_verifier.py — add sync helper and await it
def _sync_llm_verify(client, messages, response_format, temperature, max_tokens):
    return client.chat.completions.create(
        model=Config.AZURE_OPENAI_DEPLOYMENT,
        messages=messages,
        response_format=response_format,
        temperature=temperature,
        max_tokens=max_tokens,
    )

async def verify_claim_with_evidence(claim: str, evidence: Dict[str, Any]) -> Dict[str, Any]:
    # ... build messages ...
    client = get_azure_client()
    response = await asyncio.to_thread(
        _sync_llm_verify,
        client,
        [{"role": "system", "content": system_prompt}],
        {"type": "json_schema", "json_schema": json_schema},
        0.1,
        350,
    )
    result_text = response.choices[0].message.content
    # ... rest unchanged ...
```

**Impact:** Event loop no longer blocked during evidence verification; better throughput under load.

### 9.2 Sightengine response safe access

```python
# video_ai_detector.py, after parsing data
if "ai_generated" in data:
    return float(data.get("ai_generated", 0))
t = data.get("type")
if isinstance(t, dict) and t.get("ai_generated") is not None:
    return float(t.get("ai_generated", 0))
if "genai" in data and isinstance(data["genai"], dict):
    return float(data["genai"].get("ai_generated", 0))
```

**Impact:** Avoids AttributeError when API returns `"type": "genai"` (string).

### 9.3 Video pipeline OCR metadata

```python
# video_pipeline.py
capped_lines, ocr_metadata = aggregate_ocr_text(all_frame_texts)
# ... later ...
if frame_hashes:
    hash_metadata = {"video_id": ..., "platform": ..., "source_url": ...}
    # use hash_metadata for storage only
return {
    ...
    "ocrMetadata": ocr_metadata,
    ...
}
```

**Impact:** API response carries correct OCR metadata.

### 9.4 Dependencies and config

- Add to **requirements.txt:** `wikipedia`, `ffmpeg-python`, `azure-cognitiveservices-vision-computervision`, `msrest` (and optionally pin versions).
- Add to **Config (settings.py):** `NEWS_API_KEY`, `AZURE_VISION_KEY`, `AZURE_VISION_ENDPOINT`, `SIGHTENGINE_API_USER`, `SIGHTENGINE_API_SECRET` (all from env), so they are documented and optionally validated at startup.

### 9.5 Structured logging and resilience

- Replace critical `print()` with `logging` (info/warning/error).
- Add optional retry (with backoff) for: fact-check GET, News GET, and optionally Azure LLM and Deepgram.
- Consider a simple circuit breaker or “skip after N failures” for external APIs (e.g. fact-check) so a prolonged outage does not slow every request.

### 9.6 Credibility score bounds

- `compute_weighted_final_result` already clamps base score and applies a capped boost; final score is `min(95, max(0, base_score + boost_applied))`. So the final credibility score stays in 0–100 (effectively 0–95). No change needed; just confirming behaviour.

---

## 10. Summary Table

| Category | Count | Severity |
|----------|--------|----------|
| Critical (event loop blocking, crash risk) | 2 | High |
| API / response parsing bugs | 2 | Medium |
| Logic / metadata bugs | 2 | Low |
| Missing deps / config | 3+ | Medium (env-dependent) |
| Performance (optional) | 3 | Low |
| Security (TLS config) | 1 | Medium in prod |

**Priority order for production readiness:**  
1) Fix evidence verifier blocking (2.1).  
2) Harden Sightengine (2.2) and Deepgram (2.5) parsing.  
3) Add missing dependencies and document/config all API keys.  
4) Fix video pipeline metadata (2.3) and duplicate imports (2.4).  
5) Introduce logging, optional retries, and TLS configuration for production.
