import re

path = "/Users/hariharans/Documents/SIH26187/face_engine/live_scorer.py"
with open(path, "r") as f:
    content = f.read()

target = "self.recording_enabled = True # TODO: Read from config"
replacement = "import os; self.recording_enabled = os.environ.get('ENABLE_RECORDING', 'false').lower() == 'true'"

if target in content:
    content = content.replace(target, replacement)
    print("Disabled recording by default.")
else:
    print("Failed to find target for recording.")

with open(path, "w") as f:
    f.write(content)
