#!/usr/bin/env python3
"""
camera_relay.py — Vista Intelligence Edge Gateway

Connects a local IP camera (MJPEG/RTSP/HTTP) to a remote Vista Intelligence
server via WebSocket. Runs on any device that can see the camera locally.

Usage:
    python camera_relay.py \\
        --camera-url http://192.168.0.218:8080/video \\
        --server-ws  wss://xxxx.trycloudflare.com/api/camera/ws/cam_gate_1 \\
        --camera-id  cam_gate_1 \\
        --fps 15 \\
        --quality 75 \\
        --token YOUR_AUTH_TOKEN

Transport is fully configurable — works with:
    ws://SERVER_IP:5001/api/camera/ws/CAM_ID   (same LAN / known IP)
    wss://xxxx.trycloudflare.com/...            (cloudflared tunnel)
    wss://xxxx.ngrok.io/...                     (ngrok tunnel)
    wss://vista.yourdomain.com/...              (VPS / production)
"""

import argparse
import logging
import signal
import sys
import time
import threading
from collections import deque
from typing import Optional

# ── Dependency check ─────────────────────────────────────────────────────────
try:
    import cv2
    import numpy as np
    import requests
    import websocket  # websocket-client
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    print("Install with:  pip install opencv-python requests websocket-client")
    sys.exit(1)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [RELAY] %(levelname)s %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("camera_relay")

# ── Constants ─────────────────────────────────────────────────────────────────
_RECONNECT_DELAYS = [2, 4, 8, 16, 30, 60]   # seconds, exponential back-off cap at 60s
_PING_INTERVAL    = 10                        # seconds between WebSocket pings
_STALL_TIMEOUT    = 15                        # seconds without a frame before reconnect
_FRAME_QUEUE_MAX  = 4                         # max queued frames before dropping


# ── MJPEG Frame Reader ────────────────────────────────────────────────────────
class MJPEGReader:
    """
    Reads JPEG frames from an MJPEG HTTP stream in a background thread.
    Decodes each boundary-separated frame and puts it in a thread-safe queue.
    """

    def __init__(self, url: str, fps_cap: float = 25.0, timeout: float = 10.0):
        self.url = url
        self.fps_cap = fps_cap
        self.frame_interval = 1.0 / fps_cap
        self.timeout = timeout
        self._queue: deque = deque(maxlen=_FRAME_QUEUE_MAX)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_frame_time = 0.0
        self._last_push_ts = 0.0    # monotonic time of last successful frame push
        self.error: Optional[str] = None

    def start(self) -> "MJPEGReader":
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    def get_frame(self) -> Optional[bytes]:
        """Return the latest JPEG bytes, or None if none available."""
        with self._lock:
            if self._queue:
                return self._queue[-1]
            return None

    def _read_loop(self):
        """Reads the MJPEG multipart stream and extracts JPEG frames."""
        while not self._stop_event.is_set():
            try:
                log.info(f"Connecting to camera: {self.url}")
                with requests.get(self.url, stream=True, timeout=self.timeout) as resp:
                    resp.raise_for_status()
                    log.info(f"Camera connected ✓ (Content-Type: {resp.headers.get('Content-Type', '?')})")
                    self.error = None

                    buffer = b""
                    for chunk in resp.iter_content(chunk_size=4096):
                        if self._stop_event.is_set():
                            break
                        buffer += chunk

                        # Parse MJPEG multipart: find JPEG start (FFD8) and end (FFD9)
                        while True:
                            start = buffer.find(b'\xff\xd8')
                            end   = buffer.find(b'\xff\xd9', start + 2) if start != -1 else -1
                            if start == -1 or end == -1:
                                break
                            jpeg_bytes = buffer[start:end + 2]
                            buffer = buffer[end + 2:]

                            # FPS cap: drop frame if too soon
                            now = time.monotonic()
                            if now - self._last_frame_time < self.frame_interval:
                                continue
                            self._last_frame_time = now

                            with self._lock:
                                self._queue.append(jpeg_bytes)
                                self._last_push_ts = time.monotonic()

            except Exception as e:
                self.error = str(e)
                log.warning(f"Camera stream error: {e}")
                if not self._stop_event.is_set():
                    time.sleep(2.0)


class OpenCVReader:
    """
    Fallback frame reader using cv2.VideoCapture.
    Works with RTSP, USB cameras (/dev/video0), and HTTP MJPEG.
    """

    def __init__(self, source, fps_cap: float = 15.0, quality: int = 75):
        self.source = source
        self.fps_cap = fps_cap
        self.quality = quality
        self.frame_interval = 1.0 / fps_cap
        self._cap: Optional[cv2.VideoCapture] = None
        self._queue: deque = deque(maxlen=_FRAME_QUEUE_MAX)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_frame_time = 0.0
        self.error: Optional[str] = None

    def start(self) -> "OpenCVReader":
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._cap:
            self._cap.release()

    def get_frame(self) -> Optional[bytes]:
        with self._lock:
            if self._queue:
                return self._queue[-1]
            return None

    def _read_loop(self):
        src = int(self.source) if str(self.source).lstrip('-').isdigit() else self.source
        while not self._stop_event.is_set():
            try:
                log.info(f"Opening camera via OpenCV: {src}")
                self._cap = cv2.VideoCapture(src)
                if not self._cap.isOpened():
                    raise RuntimeError(f"Could not open: {src}")
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                log.info("Camera opened via OpenCV ✓")
                self.error = None

                while not self._stop_event.is_set():
                    ret, frame = self._cap.read()
                    if not ret or frame is None:
                        raise RuntimeError("Frame read failed")

                    now = time.monotonic()
                    if now - self._last_frame_time < self.frame_interval:
                        continue
                    self._last_frame_time = now

                    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                    with self._lock:
                        self._queue.append(jpeg.tobytes())

                self._cap.release()
            except Exception as e:
                self.error = str(e)
                log.warning(f"OpenCV camera error: {e}")
                if self._cap:
                    self._cap.release()
                if not self._stop_event.is_set():
                    time.sleep(2.0)


# ── WebSocket Relay ───────────────────────────────────────────────────────────
class CameraRelay:
    """
    Reads JPEG frames from a local camera and pushes them over WebSocket.

    Features:
    - Exponential backoff reconnect on disconnect
    - Heartbeat ping/pong keepalive
    - Auth token via URL query param
    - Live stats (FPS, bytes/s, connection state)
    - Graceful shutdown on SIGINT/SIGTERM
    """

    def __init__(
        self,
        camera_url: str,
        server_ws:  str,
        camera_id:  str,
        fps:        float = 15.0,
        quality:    int   = 75,
        token:      Optional[str] = None,
        use_opencv: bool  = False,
    ):
        self.camera_url = camera_url
        self.camera_id  = camera_id
        self.fps        = fps
        self.quality    = quality
        self.use_opencv = use_opencv

        # Append auth token to WS URL if provided
        if token:
            sep = "&" if "?" in server_ws else "?"
            self.server_ws = f"{server_ws}{sep}token={token}"
        else:
            self.server_ws = server_ws

        self._stop = False
        self._ws: Optional[websocket.WebSocket] = None
        self._reader = None

        # Stats
        self._frames_sent   = 0
        self._bytes_sent    = 0
        self._connect_time  = 0.0
        self._retry_count   = 0
        self._stats_lock    = threading.Lock()

    # ── Camera reader factory ────────────────────────────────────────────────
    def _make_reader(self):
        if self.use_opencv:
            return OpenCVReader(self.camera_url, fps_cap=self.fps, quality=self.quality).start()
        return MJPEGReader(self.camera_url, fps_cap=self.fps).start()

    # ── Stats printer ────────────────────────────────────────────────────────
    def _print_stats(self):
        while not self._stop:
            time.sleep(5.0)
            elapsed = time.time() - self._connect_time if self._connect_time else 0
            with self._stats_lock:
                fps = self._frames_sent / elapsed if elapsed > 0 else 0
                kbps = (self._bytes_sent / 1024) / elapsed if elapsed > 0 else 0
                status = "CONNECTED" if self._ws and not self._stop else "RECONNECTING"
                log.info(
                    f"[{status}] camera={self.camera_id} "
                    f"fps={fps:.1f} kbps={kbps:.0f} "
                    f"frames={self._frames_sent} retries={self._retry_count}"
                )

    # ── Main relay loop ──────────────────────────────────────────────────────
    def run(self):
        log.info(f"Starting relay: {self.camera_id}")
        log.info(f"  Camera  : {self.camera_url}")
        log.info(f"  Server  : {self.server_ws.split('?')[0]}")  # hide token in log
        log.info(f"  FPS cap : {self.fps}  Quality: {self.quality}%")

        # Start camera reader
        self._reader = self._make_reader()
        time.sleep(1.0)  # let reader warm up

        # Stats printer thread
        threading.Thread(target=self._print_stats, daemon=True).start()

        frame_interval = 1.0 / self.fps
        last_frame_time = 0.0

        while not self._stop:
            # ── Connect ──────────────────────────────────────────────────────
            try:
                log.info(f"Connecting to server WebSocket...")
                ws = websocket.create_connection(
                    self.server_ws,
                    timeout=10,
                    skip_utf8_validation=True,
                )
                self._ws = ws
                self._connect_time = time.time()
                self._retry_count = 0
                log.info(f"✓ WebSocket connected to server")

                last_ping = time.time()
                last_frame_received = time.time()

                # ── Relay loop ────────────────────────────────────────────────
                while not self._stop:
                    now = time.time()

                    # Ping keepalive
                    if now - last_ping >= _PING_INTERVAL:
                        try:
                            ws.ping()
                            last_ping = now
                        except Exception:
                            break

                    # Check for camera stall using reader's actual last push timestamp
                    # (avoids false "alive" when get_frame() returns same stale bytes)
                    reader_last_push = getattr(self._reader, '_last_push_ts', 0)
                    stall_ref = reader_last_push if reader_last_push > 0 else last_frame_received
                    if now - stall_ref > _STALL_TIMEOUT:
                        log.warning("Camera stall detected — reconnecting reader")
                        self._reader.stop()
                        self._reader = self._make_reader()
                        last_frame_received = now

                    # FPS gate
                    if now - last_frame_time < frame_interval:
                        time.sleep(0.005)
                        continue

                    # Get frame
                    jpeg = self._reader.get_frame()
                    if jpeg is None:
                        time.sleep(0.01)
                        continue

                    last_frame_received = now
                    last_frame_time = now

                    # Send frame as binary WebSocket message
                    try:
                        ws.send_binary(jpeg)
                        with self._stats_lock:
                            self._frames_sent += 1
                            self._bytes_sent  += len(jpeg)
                    except Exception as e:
                        log.warning(f"Send error: {e}")
                        break

                ws.close()

            except KeyboardInterrupt:
                self._stop = True
                break
            except Exception as e:
                log.warning(f"WebSocket error: {e}")

            if self._stop:
                break

            # ── Backoff reconnect ─────────────────────────────────────────────
            delay = _RECONNECT_DELAYS[min(self._retry_count, len(_RECONNECT_DELAYS) - 1)]
            self._retry_count += 1
            log.info(f"Reconnecting in {delay}s (attempt {self._retry_count})...")
            for _ in range(delay * 10):
                if self._stop:
                    break
                time.sleep(0.1)

        # ── Shutdown ──────────────────────────────────────────────────────────
        log.info("Relay shutting down...")
        if self._reader:
            self._reader.stop()
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        log.info("Relay stopped.")

    def stop(self):
        self._stop = True


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Vista Intelligence Camera Relay — stream any local camera to a remote server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local camera to same-LAN server
  python camera_relay.py \\
      --camera-url http://192.168.0.218:8080/video \\
      --server-ws  ws://192.168.0.46:5001/api/camera/ws/cam_gate \\
      --camera-id  cam_gate

  # Local camera to cloudflared tunnel
  python camera_relay.py \\
      --camera-url http://192.168.0.218:8080/video \\
      --server-ws  wss://xxxx.trycloudflare.com/api/camera/ws/cam_gate \\
      --camera-id  cam_gate --fps 10

  # RTSP camera via OpenCV
  python camera_relay.py \\
      --camera-url rtsp://admin:pass@192.168.1.100:554/stream \\
      --server-ws  ws://SERVER:5001/api/camera/ws/cam_rtsp \\
      --camera-id  cam_rtsp --opencv
        """
    )
    parser.add_argument("--camera-url", required=True,
                        help="Camera source URL (http MJPEG, rtsp://, or 0 for webcam)")
    parser.add_argument("--server-ws",  required=True,
                        help="Server WebSocket URL: ws://HOST:PORT/api/camera/ws/CAM_ID")
    parser.add_argument("--camera-id",  default="cam_remote",
                        help="Unique camera identifier (default: cam_remote)")
    parser.add_argument("--fps",        type=float, default=15.0,
                        help="Target frames per second (default: 15)")
    parser.add_argument("--quality",    type=int,   default=75,
                        help="JPEG compression quality 1-100 (default: 75)")
    parser.add_argument("--token",      default=None,
                        help="Auth token for the server (optional)")
    parser.add_argument("--opencv",     action="store_true",
                        help="Use OpenCV instead of MJPEG reader (for RTSP/webcam)")

    args = parser.parse_args()

    relay = CameraRelay(
        camera_url = args.camera_url,
        server_ws  = args.server_ws,
        camera_id  = args.camera_id,
        fps        = args.fps,
        quality    = args.quality,
        token      = args.token,
        use_opencv = args.opencv,
    )

    # Graceful shutdown on SIGINT/SIGTERM
    def _handle_signal(sig, frame):
        log.info("Signal received — stopping relay...")
        relay.stop()

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    relay.run()


if __name__ == "__main__":
    main()
