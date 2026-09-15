path = "face_engine/live_scorer.py"
with open(path, "r") as f:
    content = f.read()

old = """            # Use 640x360 for local USB cameras so 3+ concurrent USB cameras
            # fit comfortably within macOS USB host controller isochronous bandwidth.
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)"""

new = """            # Use 1280x720 for local USB cameras as 640x360 is not supported by AVFoundation and breaks the read
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)"""

if old in content:
    content = content.replace(old, new)
    with open(path, "w") as f:
        f.write(content)
    print("Patched live_scorer.py successfully.")
else:
    print("Could not find the target string in live_scorer.py")
