import os
import sys
import json
import urllib.request
import urllib.error
import hashlib
import tarfile
import zipfile

RECORD_ID = "815657"
API_URL = f"https://zenodo.org/api/records/{RECORD_ID}"
DATA_DIR = "data/external/chokepoint"

def fetch_metadata():
    print(f"Fetching metadata from {API_URL}...")
    try:
        req = urllib.request.Request(API_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 403:
            print("\n[!] ERROR 403: Forbidden")
            print("[!] The ChokePoint dataset on Zenodo requires authorization to download.")
            print("[!] You must request access through the Zenodo interface and provide an access token to this script.")
            print("[!] Alternatively, download the dataset manually and place it in the data/external/chokepoint/raw/ directory.")
            sys.exit(1)
        else:
            print(f"HTTP Error: {e.code}")
            sys.exit(1)
    except Exception as e:
        print(f"Error fetching metadata: {e}")
        sys.exit(1)

def verify_checksum(filepath, expected_md5):
    print(f"Verifying {filepath}...")
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest() == expected_md5

def main():
    os.makedirs(os.path.join(DATA_DIR, "raw"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "annotations"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "frames"), exist_ok=True)

    # 1. Attempt to fetch metadata
    metadata = fetch_metadata()
    
    # 2. Check for files
    if 'files' not in metadata:
        print("ERROR: No files found in Zenodo record. Access might be restricted.")
        sys.exit(1)

    print("\n[✓] Metadata fetched successfully. Found files:")
    for f in metadata['files']:
        print(f" - {f['key']} ({f['size']} bytes)")

    # 3. Save metadata manifest
    manifest_path = os.path.join(DATA_DIR, "metadata.json")
    with open(manifest_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[✓] Metadata saved to {manifest_path}")
    
    print("\nDataset preparation script skeleton created.")
    print("Full download logic to be implemented once file structure is confirmed.")

if __name__ == "__main__":
    main()
