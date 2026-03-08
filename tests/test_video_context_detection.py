import asyncio
import os
import tempfile
import cv2
import numpy as np
from PIL import Image
import imagehash

from app.services.video.video_context_detector import is_similar, detect_video_reuse
from app.services.video.video_frame_hasher import compute_frame_hash, compute_video_hashes

async def main():
    print("🎬 Starting Phase C: Context Reuse Detection Test...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create "Video A" sequence with structured noise
        frame_paths_a = []
        for i in range(8):
            path = os.path.join(tmpdir, f"video_a_frame_{i}.jpg")
            # Create unique noisy pattern to ensure distinct hashing
            arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.rectangle(arr, (i*5, i*5), (i*5+40, i*5+40), (255,0,0), -1) 
            img = Image.fromarray(arr)
            img.save(path)
            frame_paths_a.append(path)
            
        # Create "Video B" sequence (Completely different noise profile)
        frame_paths_b = []
        for i in range(8):
            path = os.path.join(tmpdir, f"video_b_frame_{i}.jpg")
            arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.circle(arr, (50, 50), 20+i*2, (0,255,0), -1)
            img = Image.fromarray(arr)
            img.save(path)
            frame_paths_b.append(path)
            
        print("\n[1] Testing Hashing...")
        hashes_a = compute_video_hashes(frame_paths_a)
        hashes_b = compute_video_hashes(frame_paths_b)
        
        print("Hash A subset:", hashes_a[:2])
        print("Hash B subset:", hashes_b[:2])
        
        assert len(hashes_a) == 8
        assert hashes_a[0] != hashes_b[0] # Ensure colors caused different hashes
        
        print("\n[2] Testing Similarity Logic...")
        assert is_similar(hashes_a[0], hashes_a[0]), "Exact match failed"
        assert not is_similar(hashes_a[0], hashes_b[0]), "Different images falsely matched"
        
        print("\n[3] Testing Context Reuse Detection...")
        
        # Mock database hashes representing a historical viral video "Video A"
        database = [
            {"hash": h, "metadata": {"video_id": "historical_id_123", "platform": "twitter"}}
            for h in hashes_a
        ]
        
        # Test 1: Uploading the exact same video
        res_same = detect_video_reuse(hashes_a, database)
        print("Detect Same Video:", res_same)
        assert res_same["contextReuseDetected"] is True
        assert res_same["matchedFrames"] == 8
        assert res_same["confidence"] == 1.0
        
        # Test 2: Uploading a completely foreign new video
        res_new = detect_video_reuse(hashes_b, database)
        print("Detect Foreign Video:", res_new)
        assert res_new["contextReuseDetected"] is False
        assert res_new["matchedFrames"] == 0
        
        # Test 3: Uploading a 'Stitched/Edited' Video that contains Half old / Half new
        hashes_mixed = hashes_a[:4] + hashes_b[:4]
        res_mixed = detect_video_reuse(hashes_mixed, database)
        print("Detect Mixed Video:", res_mixed)
        # Should flag as Reused because 3+ frames matched (specifically 4)
        assert res_mixed["contextReuseDetected"] is True
        assert res_mixed["matchedFrames"] == 4
        assert res_mixed["confidence"] == 0.5
        
        print("\n✅ All Phase C Tests Passed Successfully!")

if __name__ == "__main__":
    asyncio.run(main())
