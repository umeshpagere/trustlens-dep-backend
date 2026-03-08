def trace_image(image_buffer: bytes) -> dict:
    if not image_buffer or len(image_buffer) == 0:
        return {
            "reusedLikelihood": "low",
            "reason": "Empty image buffer"
        }
    
    buffer_size = len(image_buffer)
    size_in_kb = buffer_size / 1024
    
    if size_in_kb < 50:
        return {
            "reusedLikelihood": "high",
            "reason": "Small file size suggests potential reuse"
        }
    
    return {
        "reusedLikelihood": "low",
        "reason": "File size indicates original content"
    }
