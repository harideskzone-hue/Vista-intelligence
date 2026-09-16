import time
import threading
import numpy as np
from face_engine.live_scorer import get_shared_encoder, DASHBOARD_OPTIMIZATIONS_ENABLED

assert DASHBOARD_OPTIMIZATIONS_ENABLED == True, "DASHBOARD_OPTIMIZATIONS_ENABLED should be True"

def test_concurrent_first_request():
    print("Running Concurrent First-Request Test...")
    encoder = get_shared_encoder("test_cam_1")
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    results = []
    def client_worker():
        jpeg = encoder.get_jpeg(100, frame)
        results.append(jpeg)
        
    threads = [threading.Thread(target=client_worker) for _ in range(10)]
    for t in threads: t.start()
    for t in threads: t.join()
    
    assert encoder.jpeg_encodes == 1, f"Expected 1 encode, got {encoder.jpeg_encodes}"
    assert len(results) == 10
    assert all(r == results[0] for r in results)
    print("  => PASS: 10 concurrent clients produced exactly 1 JPEG encode.")

def test_rate_limit_and_independent_cams():
    print("Running Rate Limit & Independence Test...")
    enc_A = get_shared_encoder("cam_A")
    enc_B = get_shared_encoder("cam_B")
    
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    # Send 15 frames for cam A very quickly
    start = time.time()
    for i in range(15):
        enc_A.get_jpeg(200 + i, frame)
        
    assert enc_A.jpeg_encodes == 1, f"Expected 1 encode due to rate limit, got {enc_A.jpeg_encodes}"
    
    # Wait 250 ms, then send 1 for cam A and 1 for cam B
    time.sleep(0.25)
    enc_A.get_jpeg(215, frame)
    enc_B.get_jpeg(300, frame)
    
    assert enc_A.jpeg_encodes == 2
    assert enc_B.jpeg_encodes == 1
    print("  => PASS: Rate limited to 5 FPS, independent per camera.")

def test_no_queue():
    print("Running No Queue Test...")
    enc = get_shared_encoder("cam_queue")
    # There is no array/list inside the encoder except cached_jpeg
    assert not hasattr(enc, 'queue')
    print("  => PASS: No unbounded queue exists in the encoder.")

if __name__ == "__main__":
    test_concurrent_first_request()
    test_rate_limit_and_independent_cams()
    test_no_queue()
    print("ALL TESTS PASSED.")
