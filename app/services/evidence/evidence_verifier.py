import asyncio
import json
import logging
from typing import Dict, Any, List
from app.config.azure import get_azure_client
from app.config.settings import Config
from app.services.evidence_alignment import rank_evidence_sentences

logger = logging.getLogger(__name__)


def _sync_llm_verify(client, messages, response_format, temperature, max_tokens):
    """Synchronous Azure OpenAI call; run via asyncio.to_thread to avoid blocking the event loop."""
    return client.chat.completions.create(
        model=Config.AZURE_OPENAI_DEPLOYMENT,
        messages=messages,
        response_format=response_format,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _format_ranked_evidence(aligned_sentences: List[Dict[str, Any]]) -> str:
    """
    Builds a human-readable evidence block for the LLM prompt
    using strictly aligned distinct sentences.
    """
    if not aligned_sentences:
        return "No strictly aligned evidence sentences found."

    lines = []
    for idx, item in enumerate(aligned_sentences, 1):
        source  = item.get("source", "Unknown")
        sentence = item.get("sentence", "")
        sim = item.get("similarity", 0.0)
        lines.append(
            f"{idx}. {sentence} (Source: {source}, Sim: {sim:.2f})"
        )
    return "\n\n".join(lines)


async def verify_claim_with_evidence(claim: str, evidence: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates a claim against ranked, pre-filtered evidence using an LLM.

    Now uses `evidence['ranked_evidence']` (produced by Phase E ranking) instead of
    raw unfiltered sources. Only sources with trustScore >= 0.6 are included.

    Returns:
        Dict with knowledgeSupportScore, verdict, reasoning, trustedSourcesUsed,
        and evidenceSources list.
    """
    print("🧠 [EvidenceVerifier] Evaluating evidence via LLM...")

    ranked_evidence = evidence.get("ranked_evidence", [])
    
    # [Claim-Evidence Alignment] Slice articles down to highly correlated sentences
    aligned_sentences = rank_evidence_sentences(claim, ranked_evidence)

    # Fall back to legacy raw evidence if Phase E ranking isn't present
    has_legacy = (
        evidence.get("factChecks") or
        evidence.get("wikipedia") or
        evidence.get("newsArticles")
    )

    if not aligned_sentences and not has_legacy:
        return {
            "knowledgeSupportScore": 0.5,
            "verdict": "uncertain",
            "reasoning": "Insufficient aligned external evidence retrieved to verify the claim.",
            "trustedSourcesUsed": [],
            "evidenceSources": [],
        }

    # --- Build evidence text ---
    if aligned_sentences:
        # Take the top 5 sentences across all documents
        top_sentences = aligned_sentences[:5]
        evidence_text = _format_ranked_evidence(top_sentences)
        
        # We still return the unique publishers that provided these sentences
        trusted_source_names = list({
            item.get("source", "Unknown") for item in top_sentences
        })
        trusted_domains = trusted_source_names
    else:
        # Legacy fallback formatting (original verifier logic)
        evidence_text = ""
        fc_claims = evidence.get("factChecks", [])
        wiki = evidence.get("wikipedia")
        news = evidence.get("newsArticles", [])
        if fc_claims:
            evidence_text += "--- FACT CHECKS ---\n"
            for idx, fc in enumerate(fc_claims):
                reviews = fc.get("claimReview", [])
                rating = reviews[0].get("textualRating", "Unknown") if reviews else "Unknown"
                publisher = reviews[0].get("publisher", {}).get("name", "Unknown") if reviews else "Unknown"
                evidence_text += f"{idx+1}. {publisher} rated: {rating}\n"
        if wiki:
            evidence_text += f"\n--- WIKIPEDIA: {wiki.get('title')} ---\n{wiki.get('summary', '')}\n"
        if news:
            evidence_text += "\n--- NEWS ---\n"
            for idx, a in enumerate(news):
                evidence_text += f"{idx+1}. [{a.get('source')}]: {a.get('title')} - {a.get('description','')}\n"
        trusted_source_names = []
        trusted_domains = []

    # Cap token length
    if len(evidence_text) > 4000:
        evidence_text = evidence_text[:4000] + "\n...[EVIDENCE TRUNCATED]..."

    system_prompt = f"""You are a strict Misinformation Verification System.

Your job is to evaluate a specific CLAIM against EVIDENCE SENTENCES retrieved from trusted sources only.
All evidence sentences below have been pre-screened and aligned for high semantic relevance to the claim. Low-quality or unreliable sources have already been excluded.

CLAIM TO VERIFY:
"{claim}"

EVIDENCE SENTENCES FROM TRUSTED SOURCES:
{evidence_text}

Carefully evaluate whether the specific evidence sentences:
1. Support the claim (score close to 1.0)
2. Contradict the claim (score close to 0.0)
3. Are insufficient or unrelated (score around 0.5)

Base your verdict strictly on the evidence sentences provided. Do not fabricate or introduce external knowledge.
Return your analysis strictly as JSON without any markdown formatting wrappers.
"""

    json_schema = {
        "name": "evidence_verification",
        "description": "Structured evidence validation result.",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "knowledgeSupportScore": {
                    "type": "number",
                    "description": "0.0–1.0 : how strongly trusted evidence supports the claim."
                },
                "verdict": {
                    "type": "string",
                    "enum": ["supported", "contradicted", "uncertain"],
                    "description": "Categorical verdict from the evidence."
                },
                "reasoning": {
                    "type": "string",
                    "description": "1-2 sentence explanation citing specific sources."
                },
                "supportingSourceCount": {
                    "type": "integer",
                    "description": "Number of provided sources that actively support the claim."
                },
                "contradictingSourceCount": {
                    "type": "integer",
                    "description": "Number of provided sources that actively contradict the claim."
                }
            },
            "required": ["knowledgeSupportScore", "verdict", "reasoning", "supportingSourceCount", "contradictingSourceCount"],
            "additionalProperties": False
        }
    }

    try:
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
        result_json = json.loads(result_text)
        verdict = result_json.get("verdict")
        score   = result_json.get("knowledgeSupportScore")
        print(f"✅ [EvidenceVerifier] Verdict: {verdict} | Score: {score}")

        # Attach source metadata
        result_json["trustedSourcesUsed"] = trusted_domains
        result_json["evidenceSources"]    = list({
            item.get("type", "").replace("factcheck", "Google Fact Check API")
                                 .replace("wikipedia", "Wikipedia")
                                 .replace("news", "NewsAPI")
            for item in ranked_evidence
        }) if ranked_evidence else (
            ["Google Fact Check API"] * bool(evidence.get("factChecks")) +
            ["Wikipedia"] * bool(evidence.get("wikipedia")) +
            ["NewsAPI"] * bool(evidence.get("newsArticles"))
        )

        sup = result_json.get("supportingSourceCount", 0)
        con = result_json.get("contradictingSourceCount", 0)
        total_evals = sup + con
        result_json["agreementScore"] = round(sup / total_evals, 2) if total_evals > 0 else 0.5
        result_json["independentSourceCount"] = len(trusted_domains)

        return result_json

    except Exception as e:
        print(f"❌ [EvidenceVerifier] Azure OpenAI failed: {e}")
        return {
            "knowledgeSupportScore": 0.5,
            "verdict": "uncertain",
            "reasoning": "Internal error during LLM evidence validation.",
            "supportingSourceCount": 0,
            "contradictingSourceCount": 0,
            "agreementScore": 0.5,
            "trustedSourcesUsed": [],
            "evidenceSources": [],
            "independentSourceCount": 0,
        }
