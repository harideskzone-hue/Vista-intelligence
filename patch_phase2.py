import re

path = "face_engine/live_scorer.py"
with open(path, "r") as f:
    content = f.read()

# 1. Add INFERENCE_MODE at the top of the file
if "INFERENCE_MODE =" not in content:
    content = content.replace("import threading", 
                              "import threading\nimport os\n\nINFERENCE_MODE = os.environ.get('INFERENCE_MODE', 'legacy')\nINFERENCE_BATCH_SIZE = int(os.environ.get('INFERENCE_BATCH_SIZE', '4'))\n")

# 2. Add CentralInferenceWorker before FaceDetectorThread
worker_code = """
class CentralInferenceWorker:
    def __init__(self, model_path, batch_size=4, use_mps=True):
        self.batch_size = batch_size
        self.use_mps = use_mps
        
        import torch
        from ultralytics import YOLO
        
        # Load model explicitly with correct device
        self.device = 'mps' if (self.use_mps and torch.backends.mps.is_available()) else 'cpu'
        print(f"[CentralInferenceWorker] Initializing YOLO on {self.device} with batch size {self.batch_size}")
        self.model = YOLO(model_path)
        
        self.frames_lock = threading.Lock()
        self.latest_frames = {} # cam_id -> frame
        self.results = {}       # cam_id -> raw bounding boxes
        
        # Tracking states
        self.tracks = {}
        self.next_track_ids = {}
        
        self._stopped = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        
    def submit(self, cam_id, frame):
        with self.frames_lock:
            self.latest_frames[cam_id] = frame
            if cam_id not in self.tracks:
                self.tracks[cam_id] = {}
                self.next_track_ids[cam_id] = 1
                self.results[cam_id] = []
                
    def get_result(self, cam_id):
        # Tracking logic ported here to preserve state per camera
        boxes = self.results.get(cam_id, [])
        tracks = self.tracks.get(cam_id, {})
        next_id = self.next_track_ids.get(cam_id, 1)
        
        new_tracks = {}
        matched = set()
        
        def _iou(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            interArea = max(0, xB - xA) * max(0, yB - yA)
            boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
            return interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
            
        for tid, last_box in tracks.items():
            best_iou = 0
            best_idx = -1
            for idx, det in enumerate(boxes):
                if idx in matched: continue
                iou = _iou(last_box, det)
                if iou > best_iou:
                    best_iou, best_idx = iou, idx
            if best_iou > 0.3:
                new_tracks[tid] = boxes[best_idx]
                matched.add(best_idx)
                
        for idx, det in enumerate(boxes):
            if idx not in matched:
                new_tracks[next_id] = det
                next_id += 1
                
        self.tracks[cam_id] = new_tracks
        self.next_track_ids[cam_id] = next_id
        
        return [{'track_id': str(tid), 'box': b} for tid, b in new_tracks.items()]
        
    def _run(self):
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
                print(f"[CentralInferenceWorker] Batch inference error: {e}")
                
    def stop(self):
        self._stopped = True

_global_central_worker = None

# ── Non-blocking Face inference thread ───────────────────────────────────────
"""
if "class CentralInferenceWorker:" not in content:
    content = content.replace("# ── Non-blocking Face inference thread ───────────────────────────────────────", worker_code)


# 3. Patch CameraProcessor to use CentralInferenceWorker if not legacy
old_init = """        self.face_thread = FaceDetectorThread(face_det_model, conf=0.5)
        self.face_thread.start()"""
        
new_init = """        self.use_legacy = (INFERENCE_MODE == "legacy")
        if self.use_legacy:
            self.face_thread = FaceDetectorThread(face_det_model, conf=0.5)
            self.face_thread.start()
        else:
            global _global_central_worker
            if _global_central_worker is None:
                model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "yolo26n-face.pt")
                _global_central_worker = CentralInferenceWorker(model_path, batch_size=INFERENCE_BATCH_SIZE)
            self.face_thread = None"""

if old_init in content:
    content = content.replace(old_init, new_init)

old_process = """        # We always prefer latest frame for inference
        self.face_thread.submit(enhanced_frame)

        # FPS tracking"""
        
new_process = """        # We always prefer latest frame for inference
        if self.use_legacy:
            self.face_thread.submit(enhanced_frame)
        else:
            _global_central_worker.submit(self.cam_id, enhanced_frame)

        # FPS tracking"""

if old_process in content:
    content = content.replace(old_process, new_process)
    
old_result = """        faces = self.face_thread.result"""
new_result = """        faces = self.face_thread.result if self.use_legacy else _global_central_worker.get_result(self.cam_id)"""

if old_result in content:
    content = content.replace(old_result, new_result)

with open(path, "w") as f:
    f.write(content)
print("Phase 2 patches applied.")
