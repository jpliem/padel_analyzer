import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Optional, List
from cv.detectors.base import BallDetector
from cv.detectors.device import get_device


# ── Tennis TrackNet (yastrebksv/TrackNet) ─────────────────────────────


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pad=1, stride=1, bias=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.block(x)


class TrackNetV2Model(nn.Module):
    """TrackNet for tennis/padel ball detection.

    Architecture from yastrebksv/TrackNet, trained on tennis ball data.
    Input: (B, 9, 360, 640) — 3 consecutive RGB frames.
    Output: (B, 256, 230400) — per-pixel softmax over 256 heatmap bins.
    """

    def __init__(self, out_channels=256):
        super().__init__()
        self.out_channels = out_channels

        # Encoder
        self.conv1 = ConvBlock(9, 64)
        self.conv2 = ConvBlock(64, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = ConvBlock(128, 256)
        self.conv6 = ConvBlock(256, 256)
        self.conv7 = ConvBlock(256, 256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv8 = ConvBlock(256, 512)
        self.conv9 = ConvBlock(512, 512)
        self.conv10 = ConvBlock(512, 512)

        # Decoder
        self.ups1 = nn.Upsample(scale_factor=2)
        self.conv11 = ConvBlock(512, 256)
        self.conv12 = ConvBlock(256, 256)
        self.conv13 = ConvBlock(256, 256)
        self.ups2 = nn.Upsample(scale_factor=2)
        self.conv14 = ConvBlock(256, 128)
        self.conv15 = ConvBlock(128, 128)
        self.ups3 = nn.Upsample(scale_factor=2)
        self.conv16 = ConvBlock(128, 64)
        self.conv17 = ConvBlock(64, 64)
        self.conv18 = ConvBlock(64, out_channels)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor, testing: bool = True) -> torch.Tensor:
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.ups1(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.ups2(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.ups3(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        out = x.reshape(batch_size, self.out_channels, -1)
        if testing:
            out = self.softmax(out)
        return out


class TrackNetBallDetector(BallDetector):
    """Ball detector using TrackNet with 3-frame temporal input and optional YOLO fallback.

    Uses tennis-trained weights. Output is 256-class per-pixel softmax.
    Peak of the "ball present" probability map gives ball center.
    """

    HEATMAP_W, HEATMAP_H = 640, 360

    def __init__(self, model: TrackNetV2Model = None, model_path: str = None,
                 conf_threshold: float = 0.5, yolo_fallback=None):
        self._device_str = get_device()
        self._torch_device = torch.device(self._device_str)

        if model is not None:
            self._model = model
        elif model_path is not None:
            self._model = TrackNetV2Model(out_channels=256)
            self._model.load_state_dict(
                torch.load(model_path, map_location=self._torch_device, weights_only=False))
        else:
            raise ValueError("Either model or model_path must be provided")

        self._model.to(self._torch_device)
        self._model.eval()
        self._buffer: List[np.ndarray] = []
        self._conf_threshold = conf_threshold
        self._yolo_fallback = yolo_fallback

    def detect(self, frame: np.ndarray, frame_id: int = 0) -> Optional[List[float]]:
        self._buffer.append(frame)
        if len(self._buffer) > 3:
            self._buffer.pop(0)
        if len(self._buffer) < 3:
            if self._yolo_fallback:
                return self._yolo_fallback.detect(frame, frame_id)
            return None

        heatmap = self._infer(self._buffer)
        peak_conf, peak_y, peak_x = self._find_peak(heatmap)

        if peak_conf >= self._conf_threshold:
            frame_h, frame_w = frame.shape[:2]
            scale_x = frame_w / self.HEATMAP_W
            scale_y = frame_h / self.HEATMAP_H
            cx = peak_x * scale_x
            cy = peak_y * scale_y
            radius = 10
            return [cx - radius, cy - radius, cx + radius, cy + radius]

        if self._yolo_fallback:
            return self._yolo_fallback.detect(frame, frame_id)
        return None

    def warm_up(self) -> None:
        dummy_frames = [np.zeros((self.HEATMAP_H, self.HEATMAP_W, 3), dtype=np.uint8)] * 3
        self._infer(dummy_frames)

    @property
    def device(self) -> str:
        return self._device_str

    def _infer(self, frames: List[np.ndarray]) -> np.ndarray:
        resized = []
        for f in frames:
            r = cv2.resize(f, (self.HEATMAP_W, self.HEATMAP_H))
            r = r.astype(np.float32) / 255.0
            resized.append(r)

        # Stack 3 frames → 9 channels
        stacked = np.concatenate(resized, axis=2)  # (360, 640, 9)
        tensor = torch.from_numpy(stacked).permute(2, 0, 1).unsqueeze(0)  # (1, 9, 360, 640)
        tensor = tensor.to(self._torch_device)

        with torch.no_grad():
            output = self._model(tensor, testing=True)  # (1, 256, 230400)

        # Reshape to (256, 360, 640), take argmax per pixel → class index
        # Then get the "ball present" probability
        out = output[0]  # (256, 230400)
        out = out.reshape(self._model.out_channels, self.HEATMAP_H, self.HEATMAP_W)

        # Each pixel has 256 classes — higher class index = higher ball probability
        # The heatmap is the probability of the ball being present (sum of high classes)
        # or more simply: weighted sum where higher class = more likely ball
        # Take argmax per pixel — if class > threshold, ball is there
        class_map = out.argmax(dim=0).cpu().numpy().astype(np.float32)  # (360, 640)

        # Normalize to 0-1 range
        heatmap = class_map / 255.0
        return heatmap

    @staticmethod
    def _find_peak(heatmap: np.ndarray):
        smoothed = cv2.GaussianBlur(heatmap, (5, 5), 2.0)
        peak_idx = smoothed.argmax()
        peak_y, peak_x = np.unravel_index(peak_idx, smoothed.shape)
        peak_conf = float(smoothed[peak_y, peak_x])
        return peak_conf, peak_y, peak_x
