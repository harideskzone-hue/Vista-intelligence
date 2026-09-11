from pathlib import Path
from collections import defaultdict

import cv2
from ultralytics import YOLO


VIDEO = Path(
    "vehicle_ai/source/compressed_output_TrafficPolice.mp4"
)

MODEL = YOLO("yolo11n.pt")

CLASSES = [2, 3, 5, 7]

cap = cv2.VideoCapture(str(VIDEO))

if not cap.isOpened():
    raise RuntimeError(f"Unable to open {VIDEO}")

frame_count = 0

# track_id -> statistics
tracks = defaultdict(lambda: {
    "first_frame": None,
    "last_frame": None,
    "frames": 0,
    "max_conf": 0.0,
    "class_counts": defaultdict(int),
})

# Number of active track IDs per frame.
active_counts = []

while True:
    success, frame = cap.read()

    if not success:
        break

    frame_count += 1

    results = MODEL.track(
        source=frame,
        classes=CLASSES,
        conf=0.40,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
    )

    if not results:
        active_counts.append(0)
        continue

    result = results[0]

    if (
        result.boxes is None
        or result.boxes.id is None
    ):
        active_counts.append(0)
        continue

    ids = (
        result.boxes.id
        .int()
        .cpu()
        .tolist()
    )

    classes = (
        result.boxes.cls
        .int()
        .cpu()
        .tolist()
    )

    confidences = (
        result.boxes.conf
        .float()
        .cpu()
        .tolist()
    )

    active_counts.append(len(ids))

    for track_id, class_id, conf in zip(
        ids,
        classes,
        confidences,
    ):
        stats = tracks[track_id]

        if stats["first_frame"] is None:
            stats["first_frame"] = frame_count

        stats["last_frame"] = frame_count
        stats["frames"] += 1
        stats["max_conf"] = max(
            stats["max_conf"],
            float(conf),
        )
        stats["class_counts"][int(class_id)] += 1


cap.release()


print()
print("=== VEHICLE TRACK CENSUS ===")
print(f"Frames processed        : {frame_count}")
print(f"Unique ByteTrack IDs    : {len(tracks)}")

if active_counts:
    print(
        f"Max simultaneous tracks : {max(active_counts)}"
    )

    print(
        f"Average active tracks   : "
        f"{sum(active_counts) / len(active_counts):.2f}"
    )


# Track duration statistics
durations = []

for track_id, stats in tracks.items():
    duration = (
        stats["last_frame"]
        - stats["first_frame"]
        + 1
    )

    durations.append(duration)


if durations:
    print()
    print("=== TRACK DURATION ===")
    print(f"Shortest track         : {min(durations)} frames")
    print(f"Longest track          : {max(durations)} frames")
    print(
        f"Average track duration : "
        f"{sum(durations) / len(durations):.1f} frames"
    )


# Track-length buckets
buckets = {
    "<=5 frames": 0,
    "6-15 frames": 0,
    "16-30 frames": 0,
    "31-60 frames": 0,
    ">60 frames": 0,
}

for duration in durations:
    if duration <= 5:
        buckets["<=5 frames"] += 1
    elif duration <= 15:
        buckets["6-15 frames"] += 1
    elif duration <= 30:
        buckets["16-30 frames"] += 1
    elif duration <= 60:
        buckets["31-60 frames"] += 1
    else:
        buckets[">60 frames"] += 1


print()
print("=== TRACK LENGTH DISTRIBUTION ===")

for name, count in buckets.items():
    print(f"{name:15s}: {count}")


# Vehicle class distribution across track IDs
class_names = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

print()
print("=== TRACK CLASS DISTRIBUTION ===")

class_track_counts = defaultdict(int)

for stats in tracks.values():

    if not stats["class_counts"]:
        continue

    dominant_class = max(
        stats["class_counts"],
        key=stats["class_counts"].get,
    )

    class_track_counts[dominant_class] += 1


for class_id, name in class_names.items():
    print(
        f"{name:12s}: "
        f"{class_track_counts[class_id]}"
    )


print()
print("=== LONGEST TRACKS ===")

longest = sorted(
    tracks.items(),
    key=lambda item: item[1]["frames"],
    reverse=True,
)[:30]

for track_id, stats in longest:

    duration = (
        stats["last_frame"]
        - stats["first_frame"]
        + 1
    )

    dominant_class = max(
        stats["class_counts"],
        key=stats["class_counts"].get,
    )

    print(
        f"Track {track_id:3d} | "
        f"frames={stats['frames']:4d} | "
        f"span={duration:4d} | "
        f"class={class_names.get(dominant_class, dominant_class):10s} | "
        f"max_conf={stats['max_conf']:.3f}"
    )


print()
print("STATUS: VEHICLE TRACK CENSUS COMPLETE")
