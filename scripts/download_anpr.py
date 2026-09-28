import os
import sys
import json
from datetime import datetime

DATA_DIR = "data/external/anpr"
MANIFEST_PATH = "data/external/anpr/manifest.json"

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Check for kaggle token
    kaggle_json_path = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_json_path):
        print("[!] Kaggle authentication token not found at ~/.kaggle/kaggle.json")
        print("[!] Attempting alternative public source... (None available without auth)")
        print("[!] Marking ANPR dataset as BLOCKED.")
        sys.exit(1)
        
    try:
        import kaggle
        kaggle.api.authenticate()
        print("Downloading Kaggle dataset pratik2901/indian-number-plate-dataset...")
        kaggle.api.dataset_download_files("pratik2901/indian-number-plate-dataset", path=DATA_DIR, unzip=True)
        print("Download complete.")
    except Exception as e:
        print(f"[!] Error downloading via Kaggle API: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
