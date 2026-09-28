import re

path = "/Users/hariharans/Documents/SIH26187/face_engine/live_scorer.py"
with open(path, "r") as f:
    content = f.read()

# 1. Stagger initialization
target1 = """            threading.Thread(target=_connect_worker, args=(cam_cfg, cid), daemon=True).start()"""
replacement1 = """            threading.Thread(target=_connect_worker, args=(cam_cfg, cid), daemon=True).start()
            import time
            time.sleep(1.5) # Stagger initialization to prevent USB bus lockup"""

# 2. Dynamic thread pool
target2 = """            if not hasattr(_sync_processors, 'executor'):
                _sync_processors.executor = concurrent.futures.ThreadPoolExecutor(max_workers=6)
                
            # ── Process frames (Threaded for parallel multi-cam) ─────────
            results = _sync_processors.executor.map(_process_one, list(processors.values()))"""
replacement2 = """            if not hasattr(_sync_processors, 'executor'):
                _sync_processors.executor = concurrent.futures.ThreadPoolExecutor(max_workers=16)
                
            # ── Process frames (Threaded for parallel multi-cam) ─────────
            results = _sync_processors.executor.map(_process_one, list(processors.values()))"""

if target1 in content:
    content = content.replace(target1, replacement1)
    print("Patched staggered init.")
else:
    print("Failed to patch staggered init.")

if target2 in content:
    content = content.replace(target2, replacement2)
    print("Patched max_workers.")
else:
    print("Failed to patch max_workers.")

with open(path, "w") as f:
    f.write(content)
