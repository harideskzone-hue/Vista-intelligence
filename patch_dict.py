import re

path = "/Users/hariharans/Documents/SIH26187/face_engine/live_scorer.py"
with open(path, "r") as f:
    content = f.read()

target = "results = _sync_processors.executor.map(_process_one, processors.values())"
replacement = "results = _sync_processors.executor.map(_process_one, list(processors.values()))"

if target in content:
    content = content.replace(target, replacement)
    print("Patched processors.values() iteration.")
else:
    print("Failed to find target.")

with open(path, "w") as f:
    f.write(content)
