import re

path = "face_engine/live_scorer.py"
with open(path, "r") as f:
    content = f.read()

old = "        self.face_thread.stop()"
new = "        if hasattr(self, 'face_thread') and self.face_thread:\n            self.face_thread.stop()"

if old in content:
    content = content.replace(old, new)
    with open(path, "w") as f:
        f.write(content)
    print("Fixed.")
else:
    print("Not found.")
