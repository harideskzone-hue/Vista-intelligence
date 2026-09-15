import re

path = "face_engine/live_scorer.py"
with open(path, "r") as f:
    content = f.read()

old_loop = """            # ── Process frames ───────────────────────────────────────────────
            for proc in processors.values():
                proc.process_frame()
                if getattr(proc, 'has_new_viz', False) and proc.viz_frame is not None:
                    proc.has_new_viz = False
                    _, buffer = cv2.imencode('.jpg', proc.viz_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    frame_bytes = buffer.tobytes()
                    _STREAM_FRAMES[proc.cam_id] = frame_bytes
                    if proc.cam_id not in _STREAM_BUFFERS:
                        _STREAM_BUFFERS[proc.cam_id] = deque(maxlen=900)
                    _STREAM_BUFFERS[proc.cam_id].append((time.time(), frame_bytes))"""

new_loop = """            # ── Process frames (Threaded for parallel multi-cam) ─────────
            def _process_one(proc):
                proc.process_frame()
                if getattr(proc, 'has_new_viz', False) and proc.viz_frame is not None:
                    proc.has_new_viz = False
                    _, buffer = cv2.imencode('.jpg', proc.viz_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    return proc.cam_id, buffer.tobytes()
                return None, None

            import concurrent.futures
            if not hasattr(live_scorer_module_scope_if_needed, 'executor'):
                pass # Just use a local one for simplicity
                
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(processors) or 1) as executor:
                results = executor.map(_process_one, processors.values())
                for cam_id, frame_bytes in results:
                    if cam_id and frame_bytes:
                        _STREAM_FRAMES[cam_id] = frame_bytes
                        if cam_id not in _STREAM_BUFFERS:
                            _STREAM_BUFFERS[cam_id] = deque(maxlen=900)
                        _STREAM_BUFFERS[cam_id].append((time.time(), frame_bytes))"""

if old_loop in content:
    content = content.replace(old_loop, new_loop)
    with open(path, "w") as f:
        f.write(content)
    print("Patched loop to use ThreadPoolExecutor.")
else:
    print("Old loop not found.")
