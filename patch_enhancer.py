import re

path = "/Users/hariharans/Documents/SIH26187/face_engine/live_scorer.py"
with open(path, "r") as f:
    content = f.read()

target = """        # Adaptive Enhancement (Processing Branch)
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
                _global_central_worker.submit(self.cam_id, enhanced_frame)"""

replacement = """        # Phase 3: Intelligent Scheduling
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
            # Adaptive Enhancement ONLY on the inference frame to save massive CPU load on 1080p video
            enhanced_frame = self._enhancer.enhance(frame, self.enhancement_mode)
            if self.use_legacy:
                self.face_thread.submit(enhanced_frame)
            else:
                _global_central_worker.submit(self.cam_id, enhanced_frame)"""

if target in content:
    content = content.replace(target, replacement)
    print("Patched enhancer to run only on inference frames.")
else:
    print("Failed to find target for enhancer.")

with open(path, "w") as f:
    f.write(content)
