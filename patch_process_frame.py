import re

path = "/Users/hariharans/Documents/SIH26187/face_engine/live_scorer.py"
with open(path, "r") as f:
    content = f.read()

target = """        # Adaptive Enhancement (Processing Branch)
        enhanced_frame = self._enhancer.enhance(frame, self.enhancement_mode)
        
        # Phase 3: Intelligent Scheduling
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
                _global_central_worker.submit(self.cam_id, enhanced_frame)

        # FPS tracking
        now = time.time()
        self.frame_times.append(now)
        if len(self.frame_times) >= 10 and len(self.frame_times) % 10 == 0:
            recent = list(self.frame_times)
            self.display_fps = max(1.0, min(60.0, 9.0 / (recent[-1] - recent[0])))"""

replacement = """        # Adaptive Enhancement (Processing Branch)
        enhanced_frame = self._enhancer.enhance(frame, self.enhancement_mode)
        
        # Phase 3: Intelligent Scheduling
        import os
        from collections import deque
        INFERENCE_FPS_CAP = int(os.environ.get('INFERENCE_FPS_CAP', '10'))
        
        now_ts = time.time()
        do_inference = True
        
        if not hasattr(self, 'last_yolo_time'):
            self.last_yolo_time = 0
            self.inference_times = deque(maxlen=30)
            self.inference_fps = 0.0
            
        if INFERENCE_FPS_CAP > 0:
            if now_ts - self.last_yolo_time < (1.0 / INFERENCE_FPS_CAP):
                do_inference = False
            else:
                self.last_yolo_time = now_ts
                self.inference_times.append(now_ts)
                if len(self.inference_times) > 1:
                    self.inference_fps = len(self.inference_times) / (self.inference_times[-1] - self.inference_times[0])

        if do_inference:
            if self.use_legacy:
                self.face_thread.submit(enhanced_frame)
            else:
                _global_central_worker.submit(self.cam_id, enhanced_frame)

        # FPS tracking
        now = time.time()
        self.frame_times.append(now)
        if len(self.frame_times) >= 10 and len(self.frame_times) % 10 == 0:
            recent = list(self.frame_times)
            self.display_fps = max(1.0, min(60.0, 9.0 / (recent[-1] - recent[0])))
            
        # Telemetry Logging
        if not hasattr(self, 'last_telemetry_print'):
            self.last_telemetry_print = time.time()
        
        if time.time() - self.last_telemetry_print > 5.0:
            self.last_telemetry_print = time.time()
            capture_fps = getattr(self.stream, 'capture_fps', 0.0)
            dropped = getattr(self.stream, 'dropped_frames', 0)
            print(f"[TELEMETRY CAM {self.cam_id}] Capture: {capture_fps:.1f}fps | Inference: {self.inference_fps:.1f}fps | Display: {self.display_fps:.1f}fps | Dropped: {dropped}")"""

if target in content:
    content = content.replace(target, replacement)
    print("Patched process_frame scheduling and telemetry.")
else:
    print("Failed to find target for process_frame.")

with open(path, "w") as f:
    f.write(content)
