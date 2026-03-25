import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Optional, List
from cv.detectors.base import BallDetector
from cv.detectors.device import get_device


class TrackNetV2Model(nn.Module):
    """TrackNetV2: U-Net encoder-decoder for ball detection from 3 consecutive frames."""

    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = self._conv_block(9, 64, 2)
        self.enc2 = self._conv_block(64, 128, 2)
        self.enc3 = self._conv_block(128, 256, 3)
        self.enc4 = self._conv_block(256, 512, 3)
        self.pool = nn.MaxPool2d(2, 2)

        # Decoder
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec4 = self._conv_block(512, 256, 3)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = self._conv_block(256, 128, 2)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = self._conv_block(128, 64, 2)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),
        )

    def _conv_block(self, in_ch: int, out_ch: int, n_convs: int) -> nn.Sequential:
        layers = []
        for i in range(n_convs):
            layers.append(nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Decoder with skip connections (additive)
        d4 = self.dec4(self.up4(e4))
        # Handle size mismatch from odd dimensions
        if d4.shape != e3.shape:
            d4 = nn.functional.interpolate(d4, size=e3.shape[2:])
        d4 = d4 + e3

        d3 = self.dec3(self.up3(d4))
        if d3.shape != e2.shape:
            d3 = nn.functional.interpolate(d3, size=e2.shape[2:])
        d3 = d3 + e2

        d2 = self.dec2(self.up2(d3))
        if d2.shape != e1.shape:
            d2 = nn.functional.interpolate(d2, size=e1.shape[2:])
        d2 = d2 + e1

        out = self.dec1(self.up1(d2))
        # Ensure output matches expected size
        if out.shape[2:] != (360, 640):
            out = nn.functional.interpolate(out, size=(360, 640))
        return out


class TrackNetBallDetector(BallDetector):
    """Ball detector using TrackNetV2 with 3-frame temporal input and optional YOLO fallback."""

    HEATMAP_W, HEATMAP_H = 640, 360

    def __init__(self, model: TrackNetV2Model = None, model_path: str = None,
                 conf_threshold: float = 0.5, yolo_fallback=None):
        self._device_str = get_device()
        self._torch_device = torch.device(self._device_str)

        if model is not None:
            self._model = model
        elif model_path is not None:
            self._model = TrackNetV2Model()
            self._model.load_state_dict(
                torch.load(model_path, map_location=self._torch_device, weights_only=True))
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

        # Stack 3 frames into 9-channel input: [H, W, 9]
        stacked = np.concatenate(resized, axis=2)  # (360, 640, 9)
        tensor = torch.from_numpy(stacked).permute(2, 0, 1).unsqueeze(0)  # (1, 9, 360, 640)
        tensor = tensor.to(self._torch_device)

        with torch.no_grad():
            output = self._model(tensor)

        return output[0, 0].cpu().numpy()  # (360, 640) heatmap

    @staticmethod
    def _find_peak(heatmap: np.ndarray):
        # Gaussian blur to smooth noise
        smoothed = cv2.GaussianBlur(heatmap, (5, 5), 2.0)
        peak_idx = smoothed.argmax()
        peak_y, peak_x = np.unravel_index(peak_idx, smoothed.shape)
        peak_conf = float(smoothed[peak_y, peak_x])
        return peak_conf, peak_y, peak_x
