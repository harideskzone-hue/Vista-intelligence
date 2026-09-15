path = "face_engine/live_scorer.py"
with open(path, "r") as f:
    content = f.read()

old = """            # Use 1280x720 for local USB cameras as 640x360 is not supported by AVFoundation and breaks the read
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)"""

new = """            # Use 640x480 for local USB cameras so 3+ concurrent USB cameras fit within macOS USB bandwidth.
            # (640x360 is not supported by macOS AVFoundation and fails to read frames)
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)"""

if old in content:
    content = content.replace(old, new)
    with open(path, "w") as f:
        f.write(content)
    print("Patched live_scorer.py to 640x480 successfully.")
else:
    print("Could not find the target string in live_scorer.py")
