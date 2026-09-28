import re

path = "/Users/hariharans/Documents/SIH26187/face_engine/live_scorer.py"
with open(path, "r") as f:
    content = f.read()

target = """    def process_frame(self) -> None:
        if self.stream.stopped:
            return

        # Lifecycle guard: pause processing for STALE/OFFLINE remote cameras"""

replacement = """    def process_frame(self) -> None:
        if self.stream.stopped:
            return
            
        import time
        t_start = time.time()

        # Lifecycle guard: pause processing for STALE/OFFLINE remote cameras"""

target2 = """        cv2.putText(viz, f"{self.display_fps:.1f} FPS", (w_f - 180, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1, cv2.LINE_AA)

        self.viz_frame = viz
        self.has_new_viz = True
        
        # Handle Recording
        if self.recording_enabled and _storage:
            w_f, h_f = frame.shape[1], frame.shape[0]
            fps = self.display_fps if self.display_fps > 0 else 15.0
            
            if self.video_writer is None:
                self._start_new_segment(w_f, h_f, fps)
                
            if self.video_writer is not None:
                self.video_writer.write(viz)
                self.recording_frames += 1
                
                # Close segment after ~45 seconds
                if self.recording_start_time and time.time() - self.recording_start_time > 45:
                    self._close_segment()"""

replacement2 = """        cv2.putText(viz, f"{self.display_fps:.1f} FPS", (w_f - 180, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1, cv2.LINE_AA)

        self.viz_frame = viz
        self.has_new_viz = True
        
        # Handle Recording
        if self.recording_enabled and _storage:
            w_f, h_f = frame.shape[1], frame.shape[0]
            fps = self.display_fps if self.display_fps > 0 else 15.0
            
            if self.video_writer is None:
                self._start_new_segment(w_f, h_f, fps)
                
            if self.video_writer is not None:
                self.video_writer.write(viz)
                self.recording_frames += 1
                
                # Close segment after ~45 seconds
                if self.recording_start_time and time.time() - self.recording_start_time > 45:
                    self._close_segment()
                    
        t_total = time.time() - t_start
        if t_total > 0.05:
            pass
            # print(f"[CAM {self.cam_id}] process_frame took {t_total:.3f}s")"""

if target in content and target2 in content:
    content = content.replace(target, replacement)
    content = content.replace(target2, replacement2)
    print("Patched process_frame timing.")
else:
    print("Failed to find target for timing.")

with open(path, "w") as f:
    f.write(content)
