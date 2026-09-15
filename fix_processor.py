import re

path = "face_engine/live_scorer.py"
with open(path, "r") as f:
    content = f.read()

# It's missing self.video_writer = None in __init__
old = """        self.has_new_viz = False
        
        # Load local cameras.json settings if this is a known ID"""

new = """        self.has_new_viz = False
        self.video_writer = None
        
        # Load local cameras.json settings if this is a known ID"""

if old in content:
    content = content.replace(old, new)
    with open(path, "w") as f:
        f.write(content)
    print("Fixed CameraProcessor init.")
else:
    print("Could not find the target string.")
