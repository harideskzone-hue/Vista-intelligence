import os
import json
import argparse

def create_scenario(name):
    base_dir = os.path.join("test_videos", "boundary")
    os.makedirs(base_dir, exist_ok=True)
    
    json_path = os.path.join(base_dir, f"{name}.json")
    
    if os.path.exists(json_path):
        print(f"[!] Scenario {name} already exists at {json_path}")
        return
        
    template = {
        "video": f"{name}.mp4",
        "description": "Enter scenario description here (e.g. 'Person crosses boundary', 'Loitering', 'Watchlisted face')",
        "boundary": {
            "id": "B1",
            "points": [[100, 200], [800, 200]]
        },
        "events": [
            {
                "track_id": "GT01",
                "timestamp": 12.43,
                "direction": "IN",
                "event_type": "boundary_crossing"
            }
        ],
        "expected_system_alert": {
            "person_detected": True,
            "boundary_event": True,
            "face_match": False,
            "risk_event": True,
            "alert": True
        }
    }
    
    with open(json_path, "w") as f:
        json.dump(template, f, indent=2)
        
    print(f"[✓] Created scenario template at {json_path}")
    print(f"[!] Please place the corresponding {name}.mp4 in {base_dir}")
    print(f"[!] Then manually edit the JSON to set the exact ground truth timestamps and expected events.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Name of the scenario (e.g., scenario_01_boundary_cross)")
    args = parser.parse_args()
    create_scenario(args.name)
