import re

path = "/Users/hariharans/Documents/SIH26187/face_engine/live_scorer.py"
with open(path, "r") as f:
    content = f.read()

target = """    # ── Initial camera load ─────────────────────────────────────────────────
    active_cfg   = _get_active_cameras()
    processors: dict[str, "CameraProcessor"] = {}   # cam_id → processor
    retry_times: dict[str, float]            = {}   # cam_id → next retry timestamp
    retry_counts: dict[str, int]             = {}   # cam_id → number of retries

    def _sync_processors(configs: list):
        \"\"\"
        Reconcile running processors with the desired config list.
        Starts new cameras, stops removed ones. Modifies dict in-place.
        \"\"\"
        desired_ids = {c['id'] for c in configs}

        # Stop processors for removed cameras
        removed_ids = set(processors.keys()) - desired_ids
        for cid in removed_ids:
            print(f"  [CAM] Stopping removed camera: {cid}")
            try:
                processors[cid].stop()
            except Exception as e:
                print(f"  [CAM] Warning during stop of {cid}: {e}")
            processors.pop(cid, None)
            retry_times.pop(cid, None)
            retry_counts.pop(cid, None)

        # Start processors for new cameras (not already running)
        for cam_cfg in configs:
            cid = cam_cfg['id']
            if cid in processors:
                continue  # already running
            now = time.time()
            if cid in retry_times and now < retry_times[cid]:
                continue  # still in back-off
            proc = _start_processor(cam_cfg, face_det)
            if proc:
                processors[cid] = proc
                retry_times.pop(cid, None)
                retry_counts.pop(cid, None)
            else:
                # Back-off: pick next delay based on how many retries done
                count = retry_counts.get(cid, 0)
                delay = _CAM_RETRY_DELAYS[min(count, len(_CAM_RETRY_DELAYS) - 1)]
                retry_counts[cid] = count + 1
                retry_times[cid] = time.time() + delay
                print(f"  [CAM] Will retry {cid} in {delay}s")"""

replacement = """    # ── Initial camera load ─────────────────────────────────────────────────
    active_cfg   = _get_active_cameras()
    processors: dict[str, "CameraProcessor"] = {}   # cam_id → processor
    retry_times: dict[str, float]            = {}   # cam_id → next retry timestamp
    retry_counts: dict[str, int]             = {}   # cam_id → number of retries
    _connecting_cams = set()

    def _sync_processors(configs: list):
        \"\"\"
        Reconcile running processors with the desired config list.
        Starts new cameras, stops removed ones. Modifies dict in-place.
        \"\"\"
        desired_ids = {c['id'] for c in configs}

        # Stop processors for removed cameras
        removed_ids = set(processors.keys()) - desired_ids
        for cid in removed_ids:
            print(f"  [CAM] Stopping removed camera: {cid}")
            try:
                processors[cid].stop()
            except Exception as e:
                print(f"  [CAM] Warning during stop of {cid}: {e}")
            processors.pop(cid, None)
            retry_times.pop(cid, None)
            retry_counts.pop(cid, None)

        # Start processors for new cameras (not already running)
        for cam_cfg in configs:
            cid = cam_cfg['id']
            if cid in processors or cid in _connecting_cams:
                continue  # already running or currently connecting
            now = time.time()
            if cid in retry_times and now < retry_times[cid]:
                continue  # still in back-off
                
            _connecting_cams.add(cid)
            
            def _connect_worker(c_cfg, c_id):
                proc = _start_processor(c_cfg, face_det)
                if proc:
                    processors[c_id] = proc
                    retry_times.pop(c_id, None)
                    retry_counts.pop(c_id, None)
                else:
                    count = retry_counts.get(c_id, 0)
                    delay = _CAM_RETRY_DELAYS[min(count, len(_CAM_RETRY_DELAYS) - 1)]
                    retry_counts[c_id] = count + 1
                    retry_times[c_id] = time.time() + delay
                    print(f"  [CAM] Will retry {c_id} in {delay}s")
                _connecting_cams.remove(c_id)
                
            threading.Thread(target=_connect_worker, args=(cam_cfg, cid), daemon=True).start()"""

if target in content:
    content = content.replace(target, replacement)
    print("Patched Reconnect logic.")
else:
    print("Could not find Reconnect logic target.")

with open(path, "w") as f:
    f.write(content)
