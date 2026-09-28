import sys

with open("run_phase7f_endurance.py", "r") as f:
    content = f.read()

content = content.replace(
    'p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)',
    'p = subprocess.Popen(cmd, stdout=open(f"ffmpeg_{i}.log", "w"), stderr=subprocess.STDOUT)'
)

with open("run_phase7f_endurance.py", "w") as f:
    f.write(content)

print("Replaced!")
