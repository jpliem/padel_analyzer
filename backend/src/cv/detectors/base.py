from abc import ABC, abstractmethod
from typing import Optional, List
import numpy as np


class BallDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray, frame_id: int = 0) -> Optional[List[float]]:
        """Detect ball in frame. Returns [x1,y1,x2,y2] bbox or None."""
        ...

    @abstractmethod
    def warm_up(self) -> None:
        """Run dummy inference to avoid first-frame latency."""
        ...

    @property
    @abstractmethod
    def device(self) -> str:
        ...


class PlayerDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray, frame_id: int = 0) -> np.ndarray:
        """Detect players in frame. Returns N×6 array [x1,y1,x2,y2,conf,cls]."""
        ...

    @abstractmethod
    def warm_up(self) -> None:
        ...

    @property
    @abstractmethod
    def device(self) -> str:
        ...
