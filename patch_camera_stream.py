import re

path = "/Users/hariharans/Documents/SIH26187/face_engine/live_scorer.py"
with open(path, "r") as f:
    content = f.read()

# Patch 1: CameraStream.__init__
target1 = """        if isinstance(src, int):
            self.stream = cv2.VideoCapture(src, cv2.CAP_AVFOUNDATION)
            self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.stream.set(cv2.CAP_PROP_FPS, 30)
        else:
            self.stream = cv2.VideoCapture(src)
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)"""

replacement1 = """        if isinstance(src, int):
            self.stream = cv2.VideoCapture(src, cv2.CAP_AVFOUNDATION)
            self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.stream.set(cv2.CAP_PROP_FPS, 30)
        else:
            self.stream = cv2.VideoCapture(src)
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)"""

# Patch 2: CameraStream.__init__ Telemetry
target2 = """        self.stopped = False
        self.frame_id = 0
        self._lock = threading.Lock()
        self._thread: "threading.Thread | None" = None"""

replacement2 = """        self.stopped = False
        self.frame_id = 0
        self._lock = threading.Lock()
        self._thread: "threading.Thread | None" = None
        
        # Telemetry
        from collections import deque
        self.capture_fps = 0.0
        self._frame_times = deque(maxlen=30)
        self.dropped_frames = 0
        self._frame_read = True"""

# Patch 3: CameraStream._update
target3 = """            # Decode the grabbed frame and expose it to the main thread
            ret, frame = self.stream.retrieve()
            if ret:
                with self._lock:
                    self.ret = ret
                    self.frame = frame
                    self.frame_id += 1"""

replacement3 = """            # Decode the grabbed frame and expose it to the main thread
            ret, frame = self.stream.retrieve()
            if ret:
                with self._lock:
                    if self.frame_id > 0 and not getattr(self, '_frame_read', True):
                        self.dropped_frames += 1
                        
                    self.ret = ret
                    self.frame = frame
                    self.frame_id += 1
                    self._frame_read = False
                    
                    self._frame_times.append(time.time())
                    if len(self._frame_times) > 1:
                        self.capture_fps = len(self._frame_times) / (self._frame_times[-1] - self._frame_times[0])"""

# Patch 4: CameraStream.read
target4 = """    def read(self):
        with self._lock:
            return self.ret, self.frame, self.frame_id"""

replacement4 = """    def read(self):
        with self._lock:
            self._frame_read = True
            return self.ret, self.frame, self.frame_id"""

content = content.replace(target1, replacement1)
content = content.replace(target2, replacement2)
content = content.replace(target3, replacement3)
content = content.replace(target4, replacement4)

with open(path, "w") as f:
    f.write(content)
print("Patched CameraStream.")
