import os
import requests
import sys

def enroll_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} not found.")
        return

    # Load session secret
    try:
        with open('data/.session_secret', 'r') as f:
            token = f.read().strip()
    except FileNotFoundError:
        print("Could not find session secret. Ensure the server is running.")
        return

    headers = {"x-internal-token": token}
    url = "http://localhost:5001/api/enroll"

    for person_name in os.listdir(folder_path):
        person_dir = os.path.join(folder_path, person_name)
        if not os.path.isdir(person_dir): continue

        print(f"\\nEnrolling {person_name}...")
        
        for img_name in os.listdir(person_dir):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')): continue
            
            img_path = os.path.join(person_dir, img_name)
            with open(img_path, 'rb') as img_file:
                files = {'image': (img_name, img_file, 'image/jpeg')}
                data = {'name': person_name, 'role': 'suspect'}
                
                res = requests.post(url, files=files, data=data, headers=headers)
                
                if res.status_code == 200:
                    print(f"  [+] Success: {img_name} -> {res.json().get('person_id')}")
                else:
                    print(f"  [-] Failed {img_name}: {res.text}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 bulk_enroll.py <path_to_person_folders>")
    else:
        enroll_folder(sys.argv[1])
