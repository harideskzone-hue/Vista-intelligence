"""
tests/test_remote_camera.py

Unit tests for the remote camera ingestion layer:
  - VirtualCameraStream interface compatibility
  - Frame push and read correctness
  - Status reporting
  - Registry management
"""
import sys
import os
import time
import numpy as np
import pytest

# Allow importing face_api modules without running the server
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'face_api'))


def make_jpeg(width=64, height=64) -> bytes:
    """Create a tiny valid JPEG for testing."""
    import cv2
    frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
    return buf.tobytes()


# -- VirtualCameraStream tests ------------------------------------------------

class TestVirtualCameraStream:
    def setup_method(self):
        from app.api.remote_camera_routes import VirtualCameraStream
        self.VCS = VirtualCameraStream

    def test_initial_state(self):
        stream = self.VCS("test_cam")
        ret, frame, fid = stream.read()
        assert ret == False
        assert frame is None
        assert fid == 0
        assert stream.stopped == False

    def test_push_and_read_frame(self):
        stream = self.VCS("test_cam")
        jpeg = make_jpeg()
        stream.push_jpeg(jpeg)

        ret, frame, fid = stream.read()
        assert ret == True
        assert frame is not None
        assert frame.shape[2] == 3  # BGR
        assert fid == 1

    def test_frame_id_increments(self):
        stream = self.VCS("test_cam")
        for i in range(5):
            stream.push_jpeg(make_jpeg())
        _, _, fid = stream.read()
        assert fid == 5

    def test_frames_received_counter(self):
        stream = self.VCS("test_cam")
        for _ in range(10):
            stream.push_jpeg(make_jpeg())
        assert stream.frames_received == 10

    def test_status_waiting(self):
        stream = self.VCS("test_cam")
        assert stream.status == "WAITING"

    def test_status_live_after_frame(self):
        stream = self.VCS("test_cam")
        stream.push_jpeg(make_jpeg())
        assert stream.status == "LIVE"

    def test_status_stale_after_timeout(self):
        stream = self.VCS("test_cam")
        stream.push_jpeg(make_jpeg())
        stream._last_seen = time.time() - 35  # fake 35s ago
        assert stream.status == "STALE"

    def test_status_offline_after_timeout(self):
        stream = self.VCS("test_cam")
        stream.push_jpeg(make_jpeg())
        stream._last_seen = time.time() - 65  # fake 65s ago
        assert stream.status == "OFFLINE"

    def test_to_dict_structure(self):
        stream = self.VCS("test_cam")
        stream.push_jpeg(make_jpeg())
        d = stream.to_dict()
        assert "camera_id" in d
        assert "status" in d
        assert "frames_received" in d
        assert "fps" in d
        assert d["frames_received"] == 1

    def test_start_returns_self(self):
        stream = self.VCS("test_cam")
        assert stream.start() is stream

    def test_stop_sets_stopped(self):
        stream = self.VCS("test_cam")
        stream.stop()
        assert stream.stopped == True

    def test_invalid_jpeg_ignored(self):
        stream = self.VCS("test_cam")
        stream.push_jpeg(b"not a jpeg at all")
        ret, frame, fid = stream.read()
        assert ret == False
        assert stream.frames_received == 0


# -- Registry tests -----------------------------------------------------------

class TestRemoteStreamRegistry:
    def setup_method(self):
        import app.api.remote_camera_routes as rcr
        self.rcr = rcr
        # Clear registry before each test
        rcr._REMOTE_STREAMS.clear()

    def test_register_new_stream(self):
        stream = self.rcr.register_remote_stream("cam_x")
        assert "cam_x" in self.rcr._REMOTE_STREAMS
        assert stream.cam_id == "cam_x"

    def test_register_same_id_returns_existing(self):
        s1 = self.rcr.register_remote_stream("cam_y")
        s2 = self.rcr.register_remote_stream("cam_y")
        assert s1 is s2

    def test_get_remote_stream(self):
        self.rcr.register_remote_stream("cam_z")
        assert self.rcr.get_remote_stream("cam_z") is not None
        assert self.rcr.get_remote_stream("nonexistent") is None

    def test_unregister_stream(self):
        self.rcr.register_remote_stream("cam_del")
        self.rcr.unregister_remote_stream("cam_del")
        assert self.rcr.get_remote_stream("cam_del") is None

    def test_get_all_streams(self):
        self.rcr.register_remote_stream("a")
        self.rcr.register_remote_stream("b")
        all_s = self.rcr.get_all_remote_streams()
        assert "a" in all_s
        assert "b" in all_s
