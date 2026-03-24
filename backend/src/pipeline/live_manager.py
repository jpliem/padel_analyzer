import asyncio
import base64
import cv2
import numpy as np
import threading
from typing import Optional, Dict, List
from pipeline.video_analyzer import VideoAnalyzer, FrameResult
from pipeline.replay_buffer import ReplayBuffer


class LiveManager:
    def __init__(self, analyzer: VideoAnalyzer, device_id=0,
                 record: bool = False, record_path: str = None):
        self._analyzer = analyzer
        self._device_id = device_id
        self._replay_buffer = ReplayBuffer(max_frames=900)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_no = 0
        self._latest_result: Optional[FrameResult] = None
        self._latest_jpeg: Optional[bytes] = None
        self._record = record
        self._record_path = record_path
        self._writer: Optional[cv2.VideoWriter] = None
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self.fps: float = 0.0

    def start(self):
        self._cap = cv2.VideoCapture(self._device_id)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera: {self._device_id}")
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        if self._cap:
            self._cap.release()
        if self._writer:
            self._writer.release()

    def _run_loop(self):
        import time
        retry_count = 0
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                retry_count += 1
                if retry_count >= 3:
                    self._running = False
                    break
                time.sleep(1)
                continue
            retry_count = 0

            t0 = time.monotonic()
            result = self._analyzer.process_frame(frame, self._frame_no)
            self._latest_result = result
            self._frame_no += 1

            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            self._latest_jpeg = jpeg.tobytes()

            self._replay_buffer.add(frame, timestamp=self._frame_no / 30.0)

            if self._record and self._writer is None and self._record_path:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self._writer = cv2.VideoWriter(self._record_path, fourcc, 30.0, (w, h))
            if self._writer:
                self._writer.write(frame)

            if result.events:
                for event in result.events:
                    try:
                        self._event_queue.put_nowait({
                            "type": "event",
                            "data": {
                                "event_type": event.event_type.value,
                                "timestamp": event.timestamp,
                                "frame": event.frame_number,
                            }
                        })
                    except asyncio.QueueFull:
                        pass

            elapsed = time.monotonic() - t0
            self.fps = 1.0 / elapsed if elapsed > 0 else 0.0

    def get_latest_frame_b64(self) -> Optional[str]:
        if self._latest_jpeg:
            return base64.b64encode(self._latest_jpeg).decode()
        return None

    def get_replay(self) -> List[Dict]:
        return self._replay_buffer.get_frames()

    @property
    def is_running(self) -> bool:
        return self._running
