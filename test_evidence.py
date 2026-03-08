import asyncio
from app.services.evidence.evidence_aggregator import aggregate_evidence
from app.services.evidence.evidence_verifier import verify_claim_with_evidence

async def main():
    claim = "NASA confirmed Earth will experience 6 days of darkness"
    print(f"Testing claim: {claim}")
    
    evidence = await aggregate_evidence(claim)
    
    print("\n--- Gathered Evidence ---")
    print(f"Fact Checks: {len(evidence['factChecks'])}")
    print(f"News Articles: {len(evidence['newsArticles'])}")
    if evidence['wikipedia']:
        print(f"Wikipedia: {evidence['wikipedia']['title']}")
        
    print("\n--- LLM Verification ---")
    result = await verify_claim_with_evidence(claim, evidence)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
