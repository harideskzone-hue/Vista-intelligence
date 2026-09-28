import re
import os

path = "/Users/hariharans/Documents/SIH26187/face_engine/live_scorer.py"
with open(path, "r") as f:
    content = f.read()

# 1. Patch CameraStream initialization and logging
target_init = """        if isinstance(src, int):
            self.stream = cv2.VideoCapture(src, cv2.CAP_AVFOUNDATION)
            # Use 640x480 for local USB cameras so 3+ concurrent USB cameras fit within macOS USB bandwidth.
            # (640x360 is not supported by macOS AVFoundation and fails to read frames)
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        else:
            self.stream = cv2.VideoCapture(src)
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)"""

replacement_init = """        if isinstance(src, int):
            self.stream = cv2.VideoCapture(src, cv2.CAP_AVFOUNDATION)
            self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.stream.set(cv2.CAP_PROP_FPS, 30)
        else:
            self.stream = cv2.VideoCapture(src)
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
        if self.stream.isOpened():
            w = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.stream.get(cv2.CAP_PROP_FPS)
            fourcc_val = int(self.stream.get(cv2.CAP_PROP_FOURCC))
            def decode_fourcc(v):
                return "".join([chr((int(v) >> 8 * i) & 0xFF) for i in range(4)])
            fourcc_str = decode_fourcc(fourcc_val) if fourcc_val > 0 else "UNKNOWN"
            print(f"  [CAM {src}] Negotiated: {w}x{h} @ {fps}fps, codec: {fourcc_str}")"""

if target_init in content:
    content = content.replace(target_init, replacement_init)
    print("Patched CameraStream init.")
else:
    print("Could not find CameraStream init target.")


# 2. Patch the JPEG encoder to use a cleaner bounded queue
target_encoder = """            # Async encoder pool for event buffers to prevent memory bloat
            import concurrent.futures
            if not hasattr(_sync_processors, 'enc_executor'):
                _sync_processors.enc_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

            if not hasattr(_sync_processors, 'executor'):
                _sync_processors.executor = concurrent.futures.ThreadPoolExecutor(max_workers=6)
                
            results = _sync_processors.executor.map(_process_one, processors.values())
            
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
                    ver = _STREAM_VERSIONS.get(cam_id, 0) + 1
                    _STREAM_VERSIONS[cam_id] = ver
                    _STREAM_FRAMES[cam_id] = (ver, frame_data)
                    
                    # Offload the buffer encoding asynchronously, but drop if queue is too large to prevent OOM
                    if _sync_processors.enc_executor._work_queue.qsize() < 10:
                        _sync_processors.enc_executor.submit(_async_encode_and_buffer, cam_id, frame_data, time.time())
                    else:
                        print(f"  [CAM] Encoder queue backed up (dropping frame for {cam_id} to save memory)")"""

replacement_encoder = """            # Async encoder pool for event buffers to prevent memory bloat
            import queue
            import concurrent.futures
            
            if not hasattr(_sync_processors, 'enc_queue'):
                _sync_processors.enc_queue = queue.Queue(maxsize=15)
                
                def _encoder_worker():
                    while True:
                        try:
                            cid, frm, ts = _sync_processors.enc_queue.get()
                            if isinstance(frm, np.ndarray):
                                h, w = frm.shape[:2]
                                if w > 1280:
                                    frm = cv2.resize(frm, (1280, int(1280 * h / w)))
                                _, buf = cv2.imencode('.jpg', frm, [cv2.IMWRITE_JPEG_QUALITY, 70])
                                frm = buf.tobytes()
                            if cid not in _STREAM_BUFFERS:
                                _STREAM_BUFFERS[cid] = deque(maxlen=450)
                            _STREAM_BUFFERS[cid].append((ts, frm))
                            _sync_processors.enc_queue.task_done()
                        except Exception as e:
                            print(f"[Encoder] Error: {e}")
                            
                for _ in range(4):
                    threading.Thread(target=_encoder_worker, daemon=True).start()

            if not hasattr(_sync_processors, 'executor'):
                _sync_processors.executor = concurrent.futures.ThreadPoolExecutor(max_workers=6)
                
            results = _sync_processors.executor.map(_process_one, processors.values())

            for cam_id, frame_data in results:
                if cam_id and frame_data is not None:
                    # Update live stream immediately (raw numpy array if LAZY, else bytes)
                    ver = _STREAM_VERSIONS.get(cam_id, 0) + 1
                    _STREAM_VERSIONS[cam_id] = ver
                    _STREAM_FRAMES[cam_id] = (ver, frame_data)
                    
                    # Offload the buffer encoding asynchronously using bounded queue
                    try:
                        _sync_processors.enc_queue.put_nowait((cam_id, frame_data, time.time()))
                    except queue.Full:
                        print(f"  [CAM] Encoder queue full (dropping frame for {cam_id} to save memory)")"""

if target_encoder in content:
    content = content.replace(target_encoder, replacement_encoder)
    print("Patched Encoder queue.")
else:
    print("Could not find Encoder target.")


with open(path, "w") as f:
    f.write(content)
