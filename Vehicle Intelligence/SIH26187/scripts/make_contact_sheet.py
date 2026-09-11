import cv2
import numpy as np
from pathlib import Path
import json

OUTPUT_DIR = Path("output/phase_4i")
CROPS_DIR = OUTPUT_DIR / "merges"

# Get all merge images
images = list(CROPS_DIR.glob("merge_*.jpg"))
if not images:
    print("No images found")
    exit(0)

# Sort by name
images.sort()

# Read images and add labels
labeled_images = []
for img_path in images:
    img = cv2.imread(str(img_path))
    if img is None: continue
    
    # Extract pair name from filename e.g. merge_145_155.jpg -> 145 -> 155
    name = img_path.stem.replace("merge_", "").replace("_", " -> ")
    
    # Add black bar at top for text
    bar = np.zeros((40, img.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    labeled_img = np.vstack([bar, img])
    
    # Resize to a consistent width for contact sheet
    target_w = 400
    scale = target_w / labeled_img.shape[1]
    target_h = int(labeled_img.shape[0] * scale)
    labeled_img = cv2.resize(labeled_img, (target_w, target_h))
    
    labeled_images.append(labeled_img)

# Arrange in a grid
cols = 4
rows = (len(labeled_images) + cols - 1) // cols

# Ensure all images have same height for vstack
max_h = max(img.shape[0] for img in labeled_images)
for i in range(len(labeled_images)):
    if labeled_images[i].shape[0] < max_h:
        pad = max_h - labeled_images[i].shape[0]
        labeled_images[i] = np.pad(labeled_images[i], ((0, pad), (0,0), (0,0)), mode='constant')

# Pad list to fill grid
while len(labeled_images) < rows * cols:
    labeled_images.append(np.zeros((max_h, target_w, 3), dtype=np.uint8))

grid_rows = []
for i in range(0, len(labeled_images), cols):
    row = np.hstack(labeled_images[i:i+cols])
    grid_rows.append(row)

contact_sheet = np.vstack(grid_rows)
cv2.imwrite(str(OUTPUT_DIR / "contact_sheet.jpg"), contact_sheet)
print(f"Created contact sheet with {len(images)} images.")
