with open("scripts/evaluate_tracking_mot17.py", "r") as f:
    content = f.read()

content = content.replace(
    "adapter = BoundaryTrackingAdapter()",
    "adapter = BoundaryTrackingAdapter()\n    if adapter.model is not None and __import__('torch').backends.mps.is_available():\n        adapter.model.to('mps')"
)

with open("scripts/evaluate_tracking_mot17.py", "w") as f:
    f.write(content)
print("Patched!")
