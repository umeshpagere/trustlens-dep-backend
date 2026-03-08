def analyze_image_metadata(image_buffer: bytes) -> dict:
    if not image_buffer or len(image_buffer) == 0:
        return {
            "hasMetadata": False,
            "possibleAI": False,
            "metadataRisk": "low"
        }
    
    buffer_size = len(image_buffer)
    size_in_kb = buffer_size / 1024
    size_in_mb = buffer_size / (1024 * 1024)
    
    possible_ai = size_in_kb < 30
    
    metadata_risk = "low"
    if size_in_mb > 5:
        metadata_risk = "medium"
    
    return {
        "hasMetadata": buffer_size > 0,
        "possibleAI": possible_ai,
        "metadataRisk": metadata_risk
    }
