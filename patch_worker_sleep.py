import re

path = "/Users/hariharans/Documents/SIH26187/face_engine/live_scorer.py"
with open(path, "r") as f:
    content = f.read()

target = """                    self.results[cid] = boxes
                    
            except Exception as e:
                print(f"[CentralInferenceWorker] Batch inference error: {e}")"""

replacement = """                    self.results[cid] = boxes
                    
            except Exception as e:
                print(f"[CentralInferenceWorker] Batch inference error: {e}")
                
            # Yield GIL to UI thread (prevents UI starving when inference is at 100% duty cycle)
            time.sleep(0.02)"""

if target in content:
    content = content.replace(target, replacement)
    print("Patched CentralInferenceWorker to yield GIL.")
else:
    print("Failed to find target for GIL yield.")

with open(path, "w") as f:
    f.write(content)
