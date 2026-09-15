import json

path = "cameras.json"
with open(path, "r") as f:
    cameras = json.load(f)

for cam in cameras:
    if cam.get("id") == "cam_usb_2" and cam.get("source") == "2":
        cam["source"] = "3"

with open(path, "w") as f:
    json.dump(cameras, f, indent=2)

print("Patched cameras.json successfully.")
