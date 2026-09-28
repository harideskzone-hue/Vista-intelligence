import os
import sys
import urllib.request
import zipfile

MOT17_URL = "https://motchallenge.net/data/MOT17.zip"
DATA_DIR = "data/external/mot17"

def download_file(url, dest):
    print(f"Downloading {url} to {dest}...")
    try:
        urllib.request.urlretrieve(url, dest)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        sys.exit(1)

def extract_zip(filepath, dest_dir):
    print(f"Extracting {filepath} to {dest_dir}...")
    try:
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
        print("Extraction complete.")
    except Exception as e:
        print(f"Error extracting {filepath}: {e}")
        sys.exit(1)

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "MOT17.zip")
    
    if not os.path.exists(zip_path):
        print(f"[!] The MOT17 dataset is approximately 2.8GB. Starting download...")
        download_file(MOT17_URL, zip_path)
        
    extract_zip(zip_path, DATA_DIR)
    
    # Check if expected folders exist
    if os.path.exists(os.path.join(DATA_DIR, "MOT17", "train")) and os.path.exists(os.path.join(DATA_DIR, "MOT17", "test")):
        print("[✓] MOT17 dataset successfully prepared.")
    else:
        print("[!] Validation failed: 'train' and 'test' folders not found.")

if __name__ == "__main__":
    main()
