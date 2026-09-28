with open("scripts/run_phase8g.py", "r") as f:
    content = f.read()

content = content.replace(
    "metrics=mm.metrics.motchallenge_metrics,",
    "metrics=mm.metrics.motchallenge_metrics + ['num_objects', 'num_predictions'],"
)

with open("scripts/run_phase8g.py", "w") as f:
    f.write(content)
print("Patched script successfully!")
