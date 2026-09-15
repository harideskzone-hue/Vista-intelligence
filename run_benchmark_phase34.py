import subprocess
import json
import sys
import os
import time

cams = [1, 3, 6, 10, 12]
configs = [
    {"name": "Phase 3A (Scheduling Only)", "scheduling": "true", "lazy": "false"},
    {"name": "Phase 4 (Scheduling + Lazy Encode)", "scheduling": "true", "lazy": "true"},
]

results = []

print("Starting Phase 3 & 4 benchmark suite...")

for config in configs:
    for count in cams:
        env = os.environ.copy()
        env["INFERENCE_MODE"] = "mps_batch"
        env["INFERENCE_BATCH_SIZE"] = "1"
        env["OPTIMIZE_SCHEDULING"] = config["scheduling"]
        env["LAZY_ENCODING"] = config["lazy"]
        
        try:
            res = subprocess.run(
                [sys.executable, "benchmark_runner.py", str(count)],
                env=env,
                capture_output=True,
                text=True,
                timeout=45
            )
            
            # Find JSON line in stdout
            found = False
            for line in res.stdout.splitlines():
                if line.startswith("{"):
                    data = json.loads(line)
                    data["config_name"] = config["name"]
                    results.append(data)
                    print(f"Done: {count} cams | {config['name']} | FPS: {data['avg_output_fps_per_camera']:.1f} | CPU: {data['avg_cpu_percent']:.1f}%")
                    found = True
            
            if not found:
                print(f"Error parsing JSON output for {count} cams | {config['name']}")
                print(res.stdout)
                
        except subprocess.TimeoutExpired:
            print(f"Timeout: {count} cams | {config['name']}")
            
        time.sleep(2) # Cool down
        
with open("phase34_metrics.json", "w") as f:
    json.dump(results, f, indent=2)
    
print("Phase 3 & 4 benchmark completed. Results saved to phase34_metrics.json.")
