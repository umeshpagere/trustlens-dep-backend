import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

def aggregate_ocr_text(frame_texts: List[List[str]]) -> Tuple[List[str], Dict]:
    """
    Flattens, deduplicates, and structures the OCR output from multiple frames.
    
    Args:
        frame_texts: A list of string lists, where each sublist represents the text found
                     on a single extracted frame.
                     
    Returns:
        A tuple of (unique_text_lines, metadata_dict)
    """
    all_lines = []
    frames_with_text = 0
    total_raw_detections = 0
    
    # Flatten the 2D list into a 1D sequence to retain temporal order
    for frame_idx, lines in enumerate(frame_texts):
        if lines:
            frames_with_text += 1
            total_raw_detections += len(lines)
            all_lines.extend(lines)
            
    # Deduplicate while preserving order (using a dict since dicts hold insertion order)
    # This removes exact phrase duplicates like static lower-third news banners
    unique_lines = list(dict.fromkeys(all_lines))
    
    # Advanced deduplication: remove near-exact matches 
    # (e.g., "breaking news" vs "breaking new")
    filtered_lines = []
    for line in unique_lines:
        # Check if this line is already a substring of a stored line,
        # or if a stored line is a substring of this line (keep the longer one)
        is_subsumed = False
        lines_to_remove = []
        
        for existing in filtered_lines:
            if line in existing:
                # The existing string fully captures this sequence
                is_subsumed = True
                break
            elif existing in line:
                # This new line fully captures an existing sequence. 
                # We should replace the existing one with this longer one.
                lines_to_remove.append(existing)
                
        if not is_subsumed:
            for item in lines_to_remove:
                filtered_lines.remove(item)
            filtered_lines.append(line)
            
    # Cap the output to the top 20 lines to avoid overwhelming the LLM
    capped_lines = filtered_lines[:20]
    
    metadata = {
        "ocr_text_count": len(capped_lines),
        "frames_with_text": frames_with_text,
        "raw_detections_filtered": total_raw_detections - len(capped_lines)
    }
    
    return capped_lines, metadata
