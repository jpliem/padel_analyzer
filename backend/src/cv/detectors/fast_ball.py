"""Fast ball detector using frame differencing + background subtraction.

For live mode where TrackNet is too slow (~100ms/frame).
Uses cv2.absdiff + MOG2 to find the ball as the fastest-moving small blob.
Runs at ~2-5ms/frame instead of 100ms.
"""

import cv2
import numpy as np
from typing import Optional, List
from cv.detectors.base import BallDetector
from cv.detectors.device import get_device


class FastBallDetector(BallDetector):
    """Ball detector using frame differencing — 20-50x faster than TrackNet.

    How it works:
    1. MOG2 background subtraction isolates moving objects
    2. Frame differencing between consecutive frames highlights fast motion
    3. Combined mask is filtered for small blobs (ball-sized)
    4. Brightest/most-moving small blob = ball candidate
    """

    def __init__(self, min_area: int = 30, max_area: int = 500,
                 yolo_fallback=None):
        self._device_str = "cpu"  # pure OpenCV, no GPU needed
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=40, detectShadows=False)
        self._prev_frame_gray: Optional[np.ndarray] = None
        self._min_area = min_area
        self._max_area = max_area
        self._yolo_fallback = yolo_fallback

    def detect(self, frame: np.ndarray, frame_id: int = 0) -> Optional[List[float]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # 1. Background subtraction mask (moving objects)
        fg_mask = self._bg_subtractor.apply(frame)

        # 2. Frame differencing (fast motion)
        if self._prev_frame_gray is not None:
            diff = cv2.absdiff(gray, self._prev_frame_gray)
            _, diff_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        else:
            diff_mask = np.zeros_like(gray)

        self._prev_frame_gray = gray.copy()

        # 3. Color filter — padel ball is yellow/green in HSV (wide range)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([15, 50, 60])
        upper_yellow = np.array([50, 255, 255])
        color_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        # Dilate color mask to be generous
        color_mask = cv2.dilate(color_mask, None, iterations=3)

        # 4. Combine: moving AND recently changed
        motion_mask = cv2.bitwise_and(fg_mask, diff_mask)
        # Prefer yellow regions but don't require it (70% motion + 30% color)
        combined = motion_mask.copy()
        # Boost score of yellow regions
        self._color_boost = color_mask

        # 4. Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        combined = cv2.dilate(combined, kernel, iterations=1)

        # 5. Find contours — ball is a small, roughly circular blob
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_candidate = None
        best_score = -1

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self._min_area or area > self._max_area:
                continue

            # Circularity check — ball should be roughly round
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.5:  # ball must be fairly round
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                continue

            # Ball size constraint — must be small (not a player blob)
            if w > 60 or h > 60:
                continue

            # Color boost: prefer yellow-ish blobs (likely the ball)
            roi = self._color_boost[y:y+h, x:x+w] if hasattr(self, '_color_boost') else None
            color_ratio = (roi.sum() / 255) / max(area, 1) if roi is not None and roi.size > 0 else 0

            # Score: prefer small, circular, yellow blobs
            score = circularity * (1.0 / (1 + area / 50)) * (1 + 2 * color_ratio)

            if score > best_score:
                best_score = score
                best_candidate = [float(x), float(y), float(x + w), float(y + h)]

        if best_candidate is not None:
            return best_candidate

        # Fallback to YOLO if no motion-based detection
        if self._yolo_fallback:
            return self._yolo_fallback.detect(frame, frame_id)
        return None

    def warm_up(self) -> None:
        # Feed a few blank frames to initialize the background model
        dummy = np.zeros((360, 640, 3), dtype=np.uint8)
        for _ in range(5):
            self._bg_subtractor.apply(dummy)

    @property
    def device(self) -> str:
        return self._device_str
