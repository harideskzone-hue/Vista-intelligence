import re

path = "face_engine/live_scorer.py"
with open(path, "r") as f:
    content = f.read()

# Add Feature Flags
if "LAZY_ENCODING =" not in content:
    content = content.replace("INFERENCE_MODE = ", 
                              "LAZY_ENCODING = os.environ.get('LAZY_ENCODING', 'false').lower() == 'true'\nOPTIMIZE_SCHEDULING = os.environ.get('OPTIMIZE_SCHEDULING', 'false').lower() == 'true'\nINFERENCE_MODE = ")

# Modify video_feed to encode on demand
old_video_feed = """        while True:
            frame = _STREAM_FRAMES.get(cam_id)
            # Always emit a frame — placeholder if real frame not yet available.
            # This prevents the browser <img> tag from stalling indefinitely.
            out = frame if frame is not None else _PLACEHOLDER_JPEG
            yield (b'--frame\\r\\n'
                   b'Content-Type: image/jpeg\\r\\n\\r\\n' + out + b'\\r\\n')"""

new_video_feed = """        while True:
            frame = _STREAM_FRAMES.get(cam_id)
            out = _PLACEHOLDER_JPEG
            if frame is not None:
                if isinstance(frame, np.ndarray): # Lazy encoding
                    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                    out = buf.tobytes()
                else:
                    out = frame
            yield (b'--frame\\r\\n'
                   b'Content-Type: image/jpeg\\r\\n\\r\\n' + out + b'\\r\\n')"""

if old_video_feed in content:
    content = content.replace(old_video_feed, new_video_feed)
    
# Modify get_buffered_frame to handle lazy numpy arrays
old_buffered = """    best_frame = buf[-1][1]
    min_diff = float('inf')
    # search backwards for closest frame
    for ts, frame_bytes in reversed(buf):
        diff = abs(ts - target_time)
        if diff <= min_diff:
            min_diff = diff
            best_frame = frame_bytes
        else:
            break
    return Response(content=best_frame, media_type="image/jpeg")"""

new_buffered = """    best_frame = buf[-1][1]
    min_diff = float('inf')
    # search backwards for closest frame
    for ts, frame_bytes in reversed(buf):
        diff = abs(ts - target_time)
        if diff <= min_diff:
            min_diff = diff
            best_frame = frame_bytes
        else:
            break
    if isinstance(best_frame, np.ndarray):
        _, buf_enc = cv2.imencode(".jpg", best_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        best_frame = buf_enc.tobytes()
    return Response(content=best_frame, media_type="image/jpeg")"""

if old_buffered in content:
    content = content.replace(old_buffered, new_buffered)

# Modify main loop to do Async Encoding for the buffer, or just skip buffer if LAZY
old_process_one = """            def _process_one(proc):
                proc.process_frame()
                if getattr(proc, 'has_new_viz', False) and proc.viz_frame is not None:
                    proc.has_new_viz = False
                    _, buffer = cv2.imencode('.jpg', proc.viz_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    return proc.cam_id, buffer.tobytes()
                return None, None

            import concurrent.futures"""

new_process_one = """            def _process_one(proc):
                proc.process_frame()
                if getattr(proc, 'has_new_viz', False) and proc.viz_frame is not None:
                    proc.has_new_viz = False
                    if LAZY_ENCODING:
                        return proc.cam_id, proc.viz_frame
                    else:
                        _, buffer = cv2.imencode('.jpg', proc.viz_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        return proc.cam_id, buffer.tobytes()
                return None, None

            # Async encoder pool for event buffers to prevent memory bloat
            import concurrent.futures
            if not hasattr(_sync_processors, 'enc_executor'):
                _sync_processors.enc_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
"""
if old_process_one in content:
    content = content.replace(old_process_one, new_process_one)
    
old_results = """            results = _sync_processors.executor.map(_process_one, processors.values())
            for cam_id, frame_bytes in results:
                if cam_id and frame_bytes:
                    _STREAM_FRAMES[cam_id] = frame_bytes
                    if cam_id not in _STREAM_BUFFERS:
                        _STREAM_BUFFERS[cam_id] = deque(maxlen=900)
                    _STREAM_BUFFERS[cam_id].append((time.time(), frame_bytes))"""

new_results = """            results = _sync_processors.executor.map(_process_one, processors.values())
            
            def _async_encode_and_buffer(cid, frm, ts):
                if isinstance(frm, np.ndarray):
                    # Downscale for memory saving in buffer
                    h, w = frm.shape[:2]
                    if w > 1280:
                        frm = cv2.resize(frm, (1280, int(1280 * h / w)))
                    _, buf = cv2.imencode('.jpg', frm, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    frm = buf.tobytes()
                if cid not in _STREAM_BUFFERS:
                    _STREAM_BUFFERS[cid] = deque(maxlen=450) # 450 frames = ~15s at 30fps
                _STREAM_BUFFERS[cid].append((ts, frm))

            for cam_id, frame_data in results:
                if cam_id and frame_data is not None:
                    # Update live stream immediately (raw numpy array if LAZY, else bytes)
                    _STREAM_FRAMES[cam_id] = frame_data
                    
                    # Offload the buffer encoding asynchronously
                    _sync_processors.enc_executor.submit(_async_encode_and_buffer, cam_id, frame_data, time.time())"""

if old_results in content:
    content = content.replace(old_results, new_results)

# Now Phase 3: Optimize Scheduling in CameraProcessor.process_frame()
old_submit = """        # We always prefer latest frame for inference
        if self.use_legacy:
            self.face_thread.submit(enhanced_frame)
        else:
            _global_central_worker.submit(self.cam_id, enhanced_frame)"""

new_submit = """        # Phase 3: Intelligent Scheduling
        # Throttle YOLO inference to max 10 FPS if enabled, else run on every frame
        now_ts = time.time()
        do_inference = True
        
        if OPTIMIZE_SCHEDULING:
            if not hasattr(self, 'last_yolo_time'):
                self.last_yolo_time = 0
            if now_ts - self.last_yolo_time < 0.1: # 10 FPS limit
                do_inference = False
            else:
                self.last_yolo_time = now_ts

        if do_inference:
            if self.use_legacy:
                self.face_thread.submit(enhanced_frame)
            else:
                _global_central_worker.submit(self.cam_id, enhanced_frame)"""

if old_submit in content:
    content = content.replace(old_submit, new_submit)

with open(path, "w") as f:
    f.write(content)
print("Phase 3 and Phase 4 patches applied.")
