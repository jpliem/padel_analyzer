import pytest
import numpy as np


class TestReplayBuffer:
    def test_add_frame(self):
        from pipeline.replay_buffer import ReplayBuffer
        rb = ReplayBuffer(max_frames=10)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        rb.add(frame, timestamp=0.0)
        assert len(rb) == 1

    def test_ring_buffer_wraps(self):
        from pipeline.replay_buffer import ReplayBuffer
        rb = ReplayBuffer(max_frames=3)
        for i in range(5):
            frame = np.ones((480, 640, 3), dtype=np.uint8) * i
            rb.add(frame, timestamp=float(i))
        assert len(rb) == 3

    def test_get_frames_ordered(self):
        from pipeline.replay_buffer import ReplayBuffer
        rb = ReplayBuffer(max_frames=5)
        for i in range(5):
            frame = np.ones((100, 100, 3), dtype=np.uint8) * i
            rb.add(frame, timestamp=float(i))
        frames = rb.get_frames()
        assert len(frames) == 5
        assert frames[0]["timestamp"] == 0.0
        assert frames[4]["timestamp"] == 4.0

    def test_get_frames_after_wrap(self):
        from pipeline.replay_buffer import ReplayBuffer
        rb = ReplayBuffer(max_frames=3)
        for i in range(5):
            frame = np.ones((100, 100, 3), dtype=np.uint8) * i
            rb.add(frame, timestamp=float(i))
        frames = rb.get_frames()
        assert len(frames) == 3
        assert frames[0]["timestamp"] == 2.0
        assert frames[2]["timestamp"] == 4.0

    def test_frames_stored_as_jpeg(self):
        from pipeline.replay_buffer import ReplayBuffer
        rb = ReplayBuffer(max_frames=5, jpeg_quality=70)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        rb.add(frame, timestamp=0.0)
        frames = rb.get_frames()
        assert isinstance(frames[0]["jpeg"], bytes)

    def test_clear(self):
        from pipeline.replay_buffer import ReplayBuffer
        rb = ReplayBuffer(max_frames=5)
        rb.add(np.zeros((100, 100, 3), dtype=np.uint8), 0.0)
        rb.clear()
        assert len(rb) == 0
