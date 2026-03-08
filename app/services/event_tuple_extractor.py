import asyncio
import json
import logging
from typing import Dict, Any

from app.config.azure import get_azure_client
from app.config.settings import Config

logger = logging.getLogger(__name__)

def _sync_extract_event_tuple(client, messages, response_format, temperature, max_tokens):
    """Synchronous Azure OpenAI call; run via asyncio.to_thread to avoid blocking the event loop."""
    return client.chat.completions.create(
        model=Config.AZURE_OPENAI_DEPLOYMENT,
        messages=messages,
        response_format=response_format,
        temperature=temperature,
        max_tokens=max_tokens,
    )

async def extract_event_tuple(claim: str) -> Dict[str, str]:
    """
    Extract the core event structure from the claim using an LLM.
    Returns a dictionary with 'entity', 'action', and 'object'.
    """
    if not claim or not str(claim).strip():
        logger.warning("[EventTupleExtractor] Empty claim provided. Returning fallback.")
        return {"entity": "", "action": "", "object": ""}

    system_prompt = f"""Extract the core event structure from the claim.

Return JSON with:
entity – the main person, organization, or object
action – the main event verb
object – the target of the action

Only extract the primary factual event. Do not extract conversational fillers.
If an element is missing from the sentence, return an empty string for that field.

Claim:
{claim}
"""

    json_schema = {
        "name": "event_tuple_extraction",
        "description": "Extracts the entity, action, and object from an event claim.",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "entity": {
                    "type": "string",
                    "description": "The main person, organization, or object performing the action."
                },
                "action": {
                    "type": "string",
                    "description": "The main event verb or action."
                },
                "object": {
                    "type": "string",
                    "description": "The target of the action, or the event context."
                }
            },
            "required": ["entity", "action", "object"],
            "additionalProperties": False
        }
    }

    try:
        client = get_azure_client()
        response = await asyncio.to_thread(
            _sync_extract_event_tuple,
            client,
            [{"role": "system", "content": system_prompt}],
            {"type": "json_schema", "json_schema": json_schema},
            0.1,  # Low temperature for strict factual extraction
            150,
        )

        result_text = response.choices[0].message.content
        result_json = json.loads(result_text)
        
        logger.info(
            "Event tuple extracted:\nentity=%s\naction=%s\nobject=%s",
            result_json.get("entity", ""),
            result_json.get("action", ""),
            result_json.get("object", "")
        )
        return result_json

    except Exception as e:
        logger.error(f"❌ [EventTupleExtractor] Failed to extract tuple for '{claim[:30]}...': {e}")
        # Fallback to simple split logic if LLM fails
        words = claim.split()
        if len(words) >= 3:
            return {
                "entity": " ".join(words[:2]), # Guess first two words are entity
                "action": words[2],            # Guess third word is action
                "object": " ".join(words[3:])  # Guess rest is object
            }
        
        return {
            "entity": claim,
            "action": "",
            "object": ""
        }
