def calculate_credibility_score(risk_level: str) -> dict:
    credibility_score = 100
    
    if risk_level == "medium":
        credibility_score -= 30
    elif risk_level == "high":
        credibility_score -= 60
    
    credibility_score = max(0, min(100, credibility_score))
    
    if credibility_score >= 70:
        verdict = "Reliable"
    elif credibility_score >= 40:
        verdict = "Suspicious"
    else:
        verdict = "Unreliable"
    
    return {
        "credibilityScore": credibility_score,
        "verdict": verdict
    }


def calculate_final_score(text_analysis: dict, image_analysis: dict) -> dict:
    text_skipped = not text_analysis or text_analysis.get("status") == "skipped"
    
    if text_skipped:
        # Default starting score if text is skipped
        final_score = 100
    else:
        final_score = text_analysis.get("credibilityScore", 100)
    
    if final_score is None:
        final_score = 100
    
    is_image_skipped = (
        not image_analysis or
        image_analysis == "skipped" or 
        (isinstance(image_analysis, dict) and image_analysis.get("status") == "skipped")
    )
    
    if not is_image_skipped and isinstance(image_analysis, dict):
        if image_analysis.get("credibilityScore") is not None:
            if not text_skipped:
                # Weighted average if both are present
                text_weight = 0.6
                image_weight = 0.4
                final_score = round(
                    (text_analysis.get("credibilityScore", 100) * text_weight) +
                    (image_analysis.get("credibilityScore", 100) * image_weight)
                )
            else:
                # If only image is present, use image score directly
                final_score = image_analysis.get("credibilityScore", 100)
        else:
            # If image analysis doesn't have a direct score but has metadata/tracing info
            is_reused = (
                image_analysis.get("reused") is True or
                (image_analysis.get("tracing", {}).get("reusedImage") is True)
            )
            
            if is_reused:
                final_score -= 25
            
            has_metadata_risk = (
                image_analysis.get("metadataRisk") is True or
                (image_analysis.get("metadata", {}).get("possibleScreenshot") is True or
                 image_analysis.get("metadata", {}).get("hasExif") is True)
            )
            
            if has_metadata_risk:
                final_score -= 15
    
    final_score = max(0, min(100, final_score))
    
    if final_score >= 75:
        final_verdict = "Reliable"
    elif final_score >= 40:
        final_verdict = "Questionable"
    else:
        final_verdict = "High Risk"
    
    return {
        "finalScore": final_score,
        "finalVerdict": final_verdict
    }
