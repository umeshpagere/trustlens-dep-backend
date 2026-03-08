import pytest
from unittest.mock import patch, MagicMock

import app.services.video.video_ocr_service as ocr_service
from app.services.video.video_text_aggregator import aggregate_ocr_text

# 1. Test the Text Aggregator (Filtration & Deduplication)
def test_aggregate_ocr_text_deduplication():
    # Input with duplicate phrases and near-duplicate substrings
    frame_outputs = [
        ["breaking news", "live at 5"],
        ["breaking new", "live at 5", "subscribe"], # 'breaking new' is a substring, 'subscribe' should be ignored but we assume OCR service already filtered it out of this layer.
        ["exclusive interview", "10:35"] 
    ]
    
    # Run the aggregator
    capped_lines, metadata = aggregate_ocr_text(frame_outputs)
    
    # Assertions
    # "breaking news" encompasses "breaking new" so the latter should be dropped. 
    # "live at 5" appears twice but should be deduplicated to 1.
    assert "breaking news" in capped_lines
    assert "breaking new" not in capped_lines
    assert capped_lines.count("live at 5") == 1
    assert "exclusive interview" in capped_lines
    
    assert metadata["frames_with_text"] == 3
    assert metadata["ocr_text_count"] == len(capped_lines)

# 2. Test the OCR Service normalizer/scrubber
def test_ocr_filter_rules():
    # Test our regex / string checks manually
    valid_text = []
    raw_inputs = [
        "A regular sentence", # keep
        "12:45",              # Drop (timestamp)
        "1:04:22",            # Drop (timestamp)
        "500,000",            # Drop (numeric)
        "ok",                 # Drop (too short)
        "Please Subscribe",   # Drop (watermark)
        "Follow us on TikTok" # Drop (watermark)
    ]
    
    for line in raw_inputs:
        cleaned = line.strip().lower()
        if len(cleaned) < 4: continue
        if ocr_service.numeric_only_pattern.match(cleaned): continue
        if ocr_service.timestamp_pattern.match(cleaned): continue
        if any(w in cleaned for w in ocr_service.watermark_words): continue
        valid_text.append(cleaned)
        
    assert len(valid_text) == 1
    assert valid_text[0] == "a regular sentence"

# 3. Async OCR Orchestration Mock (Integration)
@pytest.mark.asyncio
async def test_video_pipeline_orchestration():
    # Mocks for Video Pipeline
    from app.services.video.video_pipeline import process_video_text
    
    with patch('app.services.video.video_pipeline.extract_transcript') as mock_transcript:
        with patch('app.services.video.video_pipeline.extract_video_frames') as mock_ffmpeg:
            with patch('app.services.video.video_pipeline.extract_text_from_frame') as mock_ocr:
                
                # Setup mocks
                mock_transcript.return_value = {"success": True, "transcript": "Spoken audio here."}
                mock_ffmpeg.return_value = ["frame1.jpg", "frame2.jpg"]
                mock_ocr.side_effect = [
                    ["test banner"], 
                    ["test banner", "new info"]
                ]
                
                # Execute Pipeline
                result = await process_video_text("http://fake.url")
                
                # Assertions
                assert result["success"] is True
                assert result["framesAnalyzed"] == 2
                assert result["visualTextDetected"] is True
                assert "test banner" in result["ocrText"]
                assert "new info" in result["ocrText"]
                
                # Check combined string structure
                assert "Transcript:" in result["combinedVideoText"]
                assert "Spoken audio here." in result["combinedVideoText"]
                assert "On-screen text" in result["combinedVideoText"]
                assert "[test banner]" in result["combinedVideoText"]
