import os
import sys
import argparse
import asyncio
import base64
import cv2  # type: ignore
import numpy as np  # type: ignore

# Add parent directory to path to allow 'app' imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.face_service import FaceService  # type: ignore
from app.core.vector_db import get_vector_db  # type: ignore
from app.core.image_utils import encode_image_to_bytes

def delete_and_check():
    fs = FaceService()
    vector_db = get_vector_db()
    
    # 1. Identify the person and image
    target_pid = "1b50434d-9aea-40f6-82de-83382d393363"
    target_img_path = r"c:\projects\face_api\data\images\1b50434d-9aea-40f6-82de-83382d393363\b7bc89ce-f650-4d01-af2f-796d6c1af7b3.jpg"
    
    print(f"--- Action: Delete Person {target_pid} and Re-check Image ---")
    
    if not os.path.exists(target_img_path):
        print("CRITICAL: Image file not found at path. Maybe already deleted?")
        # Try to execute search anyway if we have the file bytes in memory? 
        # No, we assume it exists for this test.
        # Check if we can find it elsewhere or if we should skip reading it from disk if missing
        return

    # 2. Read image into memory (because delete_person will wipe the file)
    print(f"Reading image from: {target_img_path}")
    with open(target_img_path, "rb") as f:
        img_bytes = f.read()
    
    # Also decode for shape check just to be sure
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    print(f"Image loaded. Shape: {image.shape}")

    # Convert to base64 for search_image API
    b64_img = base64.b64encode(img_bytes).decode('utf-8')

    # 3. Delete the person
    print(f"Deleting person {target_pid}...")
    del_res = fs.delete_person(target_pid)
    print(f"Deletion Result: {del_res}")
    
    # 4. Search for the image again
    print("\n--- Searching for the image in the database (expecting NO MATCH) ---")
    # We use cropped=True as requested by the original issue, but our fix ensures it will perform detection
    search_res = fs.search_image(b64_img, is_cropped=True)
    
    print("Search Result:", search_res)
    
    if search_res['success'] and search_res.get('match'):
        print(f"WARNING: Match found! Person ID: {search_res.get('person_id')}")
        print("This implies another person has a similar face, or the deletion failed.")
    else:
        print("SUCCESS: No match found, as expected.")

if __name__ == "__main__":
    delete_and_check()
