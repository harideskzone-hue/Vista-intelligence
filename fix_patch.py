import re

path = "face_engine/live_scorer.py"
with open(path, "r") as f:
    content = f.read()

bad = "if not hasattr(live_scorer_module_scope_if_needed, 'executor'):"
good = "if True:"

content = content.replace(bad, good)
with open(path, "w") as f:
    f.write(content)

print("Fixed NameError bug.")
