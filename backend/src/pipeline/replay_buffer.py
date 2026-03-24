import cv2
import numpy as np
from typing import List, Dict


class ReplayBuffer:
    def __init__(self, max_frames: int = 900, jpeg_quality: int = 70):
        self._max = max_frames
        self._quality = jpeg_quality
        self._buffer: List[Dict] = []
        self._index = 0
        self._full = False

    def add(self, frame: np.ndarray, timestamp: float):
        _, jpeg = cv2.imencode('.jpg', frame,
                               [cv2.IMWRITE_JPEG_QUALITY, self._quality])
        entry = {"jpeg": jpeg.tobytes(), "timestamp": timestamp}

        if not self._full:
            self._buffer.append(entry)
            if len(self._buffer) >= self._max:
                self._full = True
                self._index = 0
        else:
            self._buffer[self._index] = entry
            self._index = (self._index + 1) % self._max

    def get_frames(self) -> List[Dict]:
        if not self._full:
            return list(self._buffer)
        return self._buffer[self._index:] + self._buffer[:self._index]

    def __len__(self):
        return len(self._buffer)

    def clear(self):
        self._buffer.clear()
        self._index = 0
        self._full = False
