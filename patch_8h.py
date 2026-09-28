with open("scripts/run_phase8h.py", "r") as f:
    content = f.read()

content = content.replace(
    "res_raw = adapter.model.predict(frame, conf=0.35, imgsz=480, classes=[0], verbose=False)",
    "res_raw = raw_model.predict(frame, conf=0.35, imgsz=480, classes=[0], verbose=False)"
)

content = content.replace(
    "adapter = BoundaryTrackingAdapter()",
    "adapter = BoundaryTrackingAdapter()\n    from ultralytics import YOLO\n    raw_model = YOLO(adapter._model_path)\n    if __import__('torch').backends.mps.is_available(): raw_model.to('mps')"
)

content = content.replace(
    "def evaluate_sequence(seq_path, adapter, target_fps=None):",
    "def evaluate_sequence(seq_path, adapter, raw_model, target_fps=None):"
)

content = content.replace(
    "res = evaluate_sequence(seq_path, adapter, target_fps=rate)",
    "res = evaluate_sequence(seq_path, adapter, raw_model, target_fps=rate)"
)

with open("scripts/run_phase8h.py", "w") as f:
    f.write(content)
print("Patched successfully!")
