import re

path = "/Users/hariharans/Documents/SIH26187/face_engine/live_scorer.py"
with open(path, "r") as f:
    content = f.read()

target = """    def _run(self):
        while not self._stopped:
            import time
            batch = []
            cam_ids = []
            
            with self.frames_lock:
                for cid, frame in list(self.latest_frames.items()):
                    if frame is not None:
                        batch.append(frame)
                        cam_ids.append(cid)
                        self.latest_frames[cid] = None # Clear after taking
                    if len(batch) >= self.batch_size:
                        break
                        
            if not batch:
                time.sleep(0.01)
                continue
                
            try:
                # YOLO batch inference
                # device string 'mps' works in Ultralytics directly
                out = self.model(batch, verbose=False, conf=0.5, device=self.device)
                
                for i, res in enumerate(out):
                    cid = cam_ids[i]
                    boxes = []
                    if len(res.boxes) > 0:
                        for box in res.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            boxes.append((x1, y1, x2, y2))
                    self.results[cid] = boxes
                    
            except Exception as e:
                print(f"[CentralInferenceWorker] Batch inference error: {e}")"""

replacement = """    def _run(self):
        import cv2
        while not self._stopped:
            import time
            batch = []
            cam_ids = []
            orig_shapes = []
            
            with self.frames_lock:
                for cid, frame in list(self.latest_frames.items()):
                    if frame is not None:
                        orig_shapes.append(frame.shape[:2]) # (h, w)
                        # Resize for inference (16:9 ratio) to 640x360 as benchmarked
                        inference_frame = cv2.resize(frame, (640, 360))
                        batch.append(inference_frame)
                        cam_ids.append(cid)
                        self.latest_frames[cid] = None # Clear after taking
                    if len(batch) >= self.batch_size:
                        break
                        
            if not batch:
                time.sleep(0.01)
                continue
                
            try:
                # YOLO batch inference
                out = self.model(batch, verbose=False, conf=0.5, device=self.device)
                
                for i, res in enumerate(out):
                    cid = cam_ids[i]
                    orig_h, orig_w = orig_shapes[i]
                    scale_x = orig_w / 640.0
                    scale_y = orig_h / 360.0
                    
                    boxes = []
                    if len(res.boxes) > 0:
                        for box in res.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            # Scale back to original resolution
                            x1, x2 = x1 * scale_x, x2 * scale_x
                            y1, y2 = y1 * scale_y, y2 * scale_y
                            boxes.append((x1, y1, x2, y2))
                    self.results[cid] = boxes
                    
            except Exception as e:
                print(f"[CentralInferenceWorker] Batch inference error: {e}")"""

if target in content:
    content = content.replace(target, replacement)
    print("Patched CentralInferenceWorker.")
else:
    print("Failed to find target for CentralInferenceWorker.")

with open(path, "w") as f:
    f.write(content)
