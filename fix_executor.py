import re

path = "face_engine/live_scorer.py"
with open(path, "r") as f:
    content = f.read()

bad_loop = """            import concurrent.futures
            if True:
                pass # Just use a local one for simplicity
                
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(processors) or 1) as executor:
                results = executor.map(_process_one, processors.values())
                for cam_id, frame_bytes in results:
                    if cam_id and frame_bytes:
                        _STREAM_FRAMES[cam_id] = frame_bytes
                        if cam_id not in _STREAM_BUFFERS:
                            _STREAM_BUFFERS[cam_id] = deque(maxlen=900)
                        _STREAM_BUFFERS[cam_id].append((time.time(), frame_bytes))"""

good_loop = """            import concurrent.futures
            if not hasattr(_sync_processors, 'executor'):
                _sync_processors.executor = concurrent.futures.ThreadPoolExecutor(max_workers=6)
                
            results = _sync_processors.executor.map(_process_one, processors.values())
            for cam_id, frame_bytes in results:
                if cam_id and frame_bytes:
                    _STREAM_FRAMES[cam_id] = frame_bytes
                    if cam_id not in _STREAM_BUFFERS:
                        _STREAM_BUFFERS[cam_id] = deque(maxlen=900)
                    _STREAM_BUFFERS[cam_id].append((time.time(), frame_bytes))"""

if bad_loop in content:
    content = content.replace(bad_loop, good_loop)
    with open(path, "w") as f:
        f.write(content)
    print("Fixed ThreadPoolExecutor loop.")
else:
    print("Could not find the target string.")
