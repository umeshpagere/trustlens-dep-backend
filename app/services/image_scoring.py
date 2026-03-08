def calculate_image_credibility(metadata: dict, tracing: dict, ai_prob: int = 0) -> dict:
    score = 100
    
    # Metadata risks
    if metadata and metadata.get("metadataRisk"):
        if metadata["metadataRisk"] == "medium":
            score -= 20
        elif metadata["metadataRisk"] == "high":
            score -= 40
    
    # Tracing/Reuse risks
    if tracing and tracing.get("reusedLikelihood"):
        if tracing["reusedLikelihood"] == "medium":
            score -= 20
        elif tracing["reusedLikelihood"] == "high":
            score -= 40

    # AI Generation risks (New)
    if ai_prob > 0:
        if ai_prob >= 80:
            score = min(score, 20) # High probability of AI = very low score
        elif ai_prob >= 50:
            score = min(score, 50) # Medium probability
        elif ai_prob >= 20:
            score -= 15
    
    score = max(0, min(100, score))
    
    if score >= 70:
        verdict = "Reliable"
    elif score >= 40:
        verdict = "Questionable"
    else:
        verdict = "High Risk"
    
    return {
        "score": score,
        "verdict": verdict
    }
