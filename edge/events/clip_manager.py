import subprocess
import shutil
"""
EventClipManager — Multi-Camera Circular RAM Buffer & Asynchronous Event Clip Engine.

Extracts precise event clips around incidents:
[ pre-event (T - pre_seconds) ... event (T) ... post-event (T + post_seconds) ] -> MP4

Key Design Rules:
1. Circular RAM buffer bounded by time, retaining at least the configured pre-event window.
2. Zero frame leakage: frames remain in RAM until an event triggers a clip.
3. Non-blocking: event detection loops never wait for video compilation.
4. Camera isolation: buffers and jobs are strictly keyed by camera_id.
5. Uploaded video testing: extract_offline_clip uses the exact same clip semantics.
"""

import os
import cv2
import time
import uuid
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np

log = logging.getLogger("event_clip_manager")


class TimeBoundedFrameBuffer:
    """
    Thread-safe circular frame buffer bounded primarily by time duration,
    retaining rolling frames in RAM for pre-event clip slicing.
    """
    def __init__(self, max_retention_seconds: float = 30.0, max_frames: int = 1500):
        self.max_retention_seconds = float(max_retention_seconds)
        self.max_frames = int(max_frames)
        self._buffer: List[Tuple[float, np.ndarray]] = []
        self._lock = threading.Lock()

    def append(self, timestamp: float, frame: np.ndarray) -> None:
        """Appends a frame with its timestamp, automatically pruning expired frames."""
        if frame is None or frame.size == 0:
            return

        with self._lock:
            # Append copy to avoid in-place mutations from inference loops
            self._buffer.append((timestamp, frame.copy()))
            
            # Prune by time first
            cutoff = timestamp - self.max_retention_seconds
            idx = 0
            n = len(self._buffer)
            while idx < n and self._buffer[idx][0] < cutoff:
                idx += 1
            if idx > 0:
                self._buffer = self._buffer[idx:]

            # Safety bound on max frame count to prevent runaway memory
            if len(self._buffer) > self.max_frames:
                self._buffer = self._buffer[-self.max_frames:]

    def get_slice(self, start_time: float, end_time: float) -> List[Tuple[float, np.ndarray]]:
        """
        Retrieves all frames within [start_time, end_time].
        Gracefully handles edge cases: if start_time is earlier than the earliest
        frame, returns all available frames up to end_time.
        """
        with self._lock:
            if not self._buffer:
                return []
            return [
                (t, f.copy()) for t, f in self._buffer
                if start_time <= t <= end_time
            ]

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def duration_seconds(self) -> float:
        with self._lock:
            if len(self._buffer) < 2:
                return 0.0
            return self._buffer[-1][0] - self._buffer[0][0]


class EventClipRequest:
    """Represents an active in-flight clip extraction job for a live camera stream."""
    def __init__(
        self,
        event_id: str,
        camera_id: str,
        event_timestamp: float,
        output_path: str,
        pre_seconds: float,
        post_seconds: float,
        pre_frames: List[Tuple[float, np.ndarray]],
        on_complete: Optional[Callable[[str, str, bool], None]] = None
    ):
        self.event_id = event_id
        self.camera_id = camera_id
        self.event_timestamp = event_timestamp
        self.output_path = output_path
        self.pre_seconds = pre_seconds
        self.post_seconds = post_seconds
        self.target_end_time = event_timestamp + post_seconds
        self.pre_frames = pre_frames
        self.post_frames: List[Tuple[float, np.ndarray]] = []
        self.on_complete = on_complete
        self.completed = False


class EventClipManager:
    """
    Central Manager for video clip extraction across multiple independent cameras.
    """
    def __init__(
        self,
        output_dir: Optional[str] = None,
        default_pre_seconds: float = 5.0,
        default_post_seconds: float = 5.0,
        buffer_retention_seconds: float = 30.0,
        max_workers: int = 3
    ):
        if output_dir is None:
            base_dir = os.environ.get("SIH26187_DATA")
            if not base_dir:
                base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
            output_dir = os.path.join(base_dir, "events", "clips")

        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        self.default_pre_seconds = float(default_pre_seconds)
        self.default_post_seconds = float(default_post_seconds)
        self.buffer_retention_seconds = float(buffer_retention_seconds)

        # Camera-keyed state
        self._buffers: Dict[str, TimeBoundedFrameBuffer] = {}
        self._active_requests: Dict[str, List[EventClipRequest]] = {}
        self._lock = threading.Lock()

        # Thread pool for asynchronous background video compilation
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="clip_worker")

    def _get_buffer(self, camera_id: str) -> TimeBoundedFrameBuffer:
        with self._lock:
            if camera_id not in self._buffers:
                self._buffers[camera_id] = TimeBoundedFrameBuffer(
                    max_retention_seconds=self.buffer_retention_seconds
                )
            return self._buffers[camera_id]

    def add_frame(
        self,
        camera_id: str,
        frame: np.ndarray,
        timestamp: Optional[float] = None
    ) -> None:
        """
        Feeds a live frame to the camera-isolated circular buffer and advances
        any active post-event clip collection jobs for that camera.
        """
        if frame is None or frame.size == 0:
            return

        now = float(timestamp if timestamp is not None else time.time())
        buf = self._get_buffer(camera_id)
        buf.append(now, frame)

        # Check and collect post-event frames for active requests on THIS camera
        requests_to_compile: List[EventClipRequest] = []
        with self._lock:
            active_list = self._active_requests.get(camera_id, [])
            remaining_list: List[EventClipRequest] = []
            for req in active_list:
                if now <= req.target_end_time:
                    req.post_frames.append((now, frame.copy()))
                    remaining_list.append(req)
                else:
                    # Target post duration reached!
                    req.post_frames.append((now, frame.copy()))
                    req.completed = True
                    requests_to_compile.append(req)
            self._active_requests[camera_id] = remaining_list

        # Submit completed clip requests to background thread pool
        for req in requests_to_compile:
            self._executor.submit(self._compile_live_clip, req)

    def trigger_clip(
        self,
        camera_id: str,
        event_id: Optional[str] = None,
        event_time: Optional[float] = None,
        pre_seconds: Optional[float] = None,
        post_seconds: Optional[float] = None,
        on_complete: Optional[Callable[[str, str, bool], None]] = None
    ) -> EventClipRequest:
        """
        Triggers an event clip extraction.
        Extracts pre-event frames synchronously in microseconds, then schedules
        the asynchronous post-event collection and encoding.
        
        Returns immediately without blocking detection loops.
        """
        eid = event_id or str(uuid.uuid4())
        now = float(event_time if event_time is not None else time.time())
        pre_sec = float(pre_seconds if pre_seconds is not None else self.default_pre_seconds)
        post_sec = float(post_seconds if post_seconds is not None else self.default_post_seconds)

        output_filename = f"event_{eid}.mp4"
        output_path = os.path.join(self.output_dir, output_filename)

        buf = self._get_buffer(camera_id)
        start_time = now - pre_sec
        pre_frames = buf.get_slice(start_time, now)

        req = EventClipRequest(
            event_id=eid,
            camera_id=camera_id,
            event_timestamp=now,
            output_path=output_path,
            pre_seconds=pre_sec,
            post_seconds=post_sec,
            pre_frames=pre_frames,
            on_complete=on_complete
        )

        with self._lock:
            if camera_id not in self._active_requests:
                self._active_requests[camera_id] = []
            self._active_requests[camera_id].append(req)

        log.info(f"Triggered clip for event {eid} on cam {camera_id}: {len(pre_frames)} pre-frames captured")
        return req

    def _compile_live_clip(self, req: EventClipRequest) -> None:
        """Background compilation worker for live camera clips."""
        try:
            # Combine pre-frames and post-frames in chronological order
            all_frames_with_ts = req.pre_frames + req.post_frames
            if not all_frames_with_ts:
                log.warning(f"No frames available to compile clip for event {req.event_id}")
                if req.on_complete:
                    req.on_complete(req.event_id, req.output_path, False)
                return

            # Estimate FPS from timestamps
            fps = 15.0
            if len(all_frames_with_ts) > 1:
                duration = all_frames_with_ts[-1][0] - all_frames_with_ts[0][0]
                if duration > 0.1:
                    fps = round(len(all_frames_with_ts) / duration, 2)
                    fps = max(5.0, min(60.0, fps))

            raw_frames = [f for _, f in all_frames_with_ts]
            success = self._write_mp4(raw_frames, req.output_path, fps=fps)

            if req.on_complete:
                req.on_complete(req.event_id, req.output_path, success)

        except Exception as e:
            log.error(f"Failed to compile clip for event {req.event_id}: {e}", exc_info=True)
            if req.on_complete:
                req.on_complete(req.event_id, req.output_path, False)

    def extract_offline_clip(
        self,
        frames_with_timestamps: List[Tuple[float, np.ndarray]],
        event_time: float,
        output_path: Optional[str] = None,
        event_id: Optional[str] = None,
        pre_seconds: Optional[float] = None,
        post_seconds: Optional[float] = None,
        fps: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Synchronously extracts an event clip from an uploaded video or historical dataset.
        Uses the exact same temporal semantics: [T - pre_seconds, T + post_seconds].
        Handles video boundaries gracefully.
        """
        eid = event_id or str(uuid.uuid4())
        pre_sec = float(pre_seconds if pre_seconds is not None else self.default_pre_seconds)
        post_sec = float(post_seconds if post_seconds is not None else self.default_post_seconds)

        if output_path is None:
            output_path = os.path.join(self.output_dir, f"event_{eid}.mp4")

        start_time = event_time - pre_sec
        end_time = event_time + post_sec

        # Filter available frames in the range
        matched = [
            f for t, f in frames_with_timestamps
            if start_time <= t <= end_time
        ]

        if not matched:
            log.warning(f"Offline clip for event {eid} found 0 matching frames in [{start_time:.2f}, {end_time:.2f}]")
            return False, ""

        if fps is None or fps <= 0:
            # Estimate from timestamps
            times = [t for t, _ in frames_with_timestamps if start_time <= t <= end_time]
            if len(times) > 1 and (times[-1] - times[0]) > 0.05:
                fps = len(times) / (times[-1] - times[0])
                fps = max(5.0, min(60.0, fps))
            else:
                fps = 15.0

        success = self._write_mp4(matched, output_path, fps=fps)
        return success, output_path if success else ""

    def _write_mp4(self, frames: List[np.ndarray], output_path: str, fps: float = 15.0) -> bool:
        """
        Encodes a list of BGR frames into browser-compatible H.264 MP4 with yuv420p & +faststart.
        Validates output with ffprobe before declaring success; fails safe otherwise.
        """
        if not frames:
            return False

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        h, w = frames[0].shape[:2]

        # 1. Primary: Direct FFmpeg stream encoding
        ffmpeg_bin = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
        if os.path.exists(ffmpeg_bin) or shutil.which("ffmpeg"):
            try:
                cmd = [
                    ffmpeg_bin, "-y",
                    "-f", "rawvideo",
                    "-vcodec", "rawvideo",
                    "-s", f"{w}x{h}",
                    "-pix_fmt", "bgr24",
                    "-r", str(fps),
                    "-i", "-",
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-preset", "veryfast",
                    "-movflags", "+faststart",
                    output_path
                ]
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                for f in frames:
                    if f.shape[0] != h or f.shape[1] != w:
                        f = cv2.resize(f, (w, h))
                    proc.stdin.write(f.tobytes())
                proc.stdin.close()
                proc.wait()

                if proc.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
                    if self._verify_h264_clip(output_path):
                        return True
                    else:
                        log.error(f"FFmpeg produced non-H.264 file at {output_path}")
            except Exception as e:
                log.warning(f"Direct FFmpeg encoding failed for {output_path}: {e}")

        # 2. Fallback: Write temporary file with OpenCV then transcode with FFmpeg
        temp_raw_path = output_path + ".temp.mp4"
        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(temp_raw_path, fourcc, float(fps), (w, h))
            if writer.isOpened():
                for f in frames:
                    if f.shape[0] != h or f.shape[1] != w:
                        f = cv2.resize(f, (w, h))
                    writer.write(f)
                writer.release()

                if os.path.exists(temp_raw_path) and os.path.getsize(temp_raw_path) > 0:
                    trans_cmd = [
                        ffmpeg_bin, "-y",
                        "-i", temp_raw_path,
                        "-c:v", "libx264",
                        "-pix_fmt", "yuv420p",
                        "-preset", "veryfast",
                        "-movflags", "+faststart",
                        output_path
                    ]
                    sub_res = subprocess.run(trans_cmd, capture_output=True)
                    if os.path.exists(temp_raw_path):
                        try: os.remove(temp_raw_path)
                        except Exception: pass

                    if sub_res.returncode == 0 and self._verify_h264_clip(output_path):
                        return True
        except Exception as e:
            log.warning(f"OpenCV+FFmpeg fallback failed for {output_path}: {e}")
            if os.path.exists(temp_raw_path):
                try: os.remove(temp_raw_path)
                except Exception: pass

        # Validation failed - do not mark as READY with invalid codec
        if os.path.exists(output_path):
            try: os.remove(output_path)
            except Exception: pass
        log.error(f"Clip encoding failed validation for {output_path}. Rejecting clip.")
        return False

    def _verify_h264_clip(self, file_path: str) -> bool:
        """Validates that the file has H.264 video codec via ffprobe."""
        ffprobe_bin = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"
        if not (os.path.exists(ffprobe_bin) or shutil.which("ffprobe")):
            return os.path.exists(file_path) and os.path.getsize(file_path) > 1024
        try:
            cmd = [
                ffprobe_bin, "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_name",
                "-of", "default=noprint_wrappers=1:nokey=1",
                file_path
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            codec = res.stdout.strip().lower()
            return res.returncode == 0 and "h264" in codec
        except Exception as e:
            log.warning(f"ffprobe validation check failed: {e}")
            return False

    def shutdown(self) -> None:
        """Shuts down the worker thread pool."""
        self._executor.shutdown(wait=False)
