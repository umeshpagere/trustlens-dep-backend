import asyncio
import os
import tempfile
from PIL import Image

# Setup mock for test
os.environ["SIGHTENGINE_API_USER"] = "testuser"
os.environ["SIGHTENGINE_API_SECRET"] = "testsecret"

from app.services.video.video_ai_detector import analyze_video_ai, detect_ai_frame
import app.services.video.video_ai_detector as detector

class MockResponse:
    def __init__(self, json_data, status_code=200):
        self.json_data = json_data
        self.status_code = status_code
        self.text = "test error"

    def json(self):
        return self.json_data

# Patch requests for the test
import requests
original_post = requests.post

def mock_post(*args, **kwargs):
    # Fake generic successful AI detection
    return MockResponse({
        "status": "success",
        "type": "genai",
        "ai_generated": 0.85
    })

async def main():
    print("🎬 Starting Test...")
    
    # Create some dummy images
    with tempfile.TemporaryDirectory() as tmpdir:
        frame_paths = []
        for i in range(12): # simulate 12 frames
            path = os.path.join(tmpdir, f"frame_{i}.jpg")
            img = Image.new('RGB', (100, 100), color = 'red')
            img.save(path)
            frame_paths.append(path)
            
        # Patch requests
        requests.post = mock_post
        
        print("Testing analyze_video_ai with 12 frames...")
        result = await analyze_video_ai(frame_paths)
        print("✅ Result:", result)
        
        assert result["framesAnalyzed"] == 8, f"Expected 8 frames max, got {result['framesAnalyzed']}"
        assert result["aiGeneratedProbability"] == 0.85
        assert result["isLikelyAIGenerated"] is True
        print("✅ Passed bounds and logic check!")

        # Restore original post
        requests.post = original_post

if __name__ == "__main__":
    asyncio.run(main())
