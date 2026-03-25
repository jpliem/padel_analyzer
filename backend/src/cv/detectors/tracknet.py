import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Optional, List
from cv.detectors.base import BallDetector
from cv.detectors.device import get_device


# ── TrackNet architecture (from padel_analytics) ─────────────────────


class Conv2DBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding='same', bias=False)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Double2DConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim)
        self.conv_2 = Conv2DBlock(out_dim, out_dim)

    def forward(self, x):
        return self.conv_2(self.conv_1(x))


class Triple2DConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim)
        self.conv_2 = Conv2DBlock(out_dim, out_dim)
        self.conv_3 = Conv2DBlock(out_dim, out_dim)

    def forward(self, x):
        return self.conv_3(self.conv_2(self.conv_1(x)))


class TrackNetV2Model(nn.Module):
    """TrackNet from padel_analytics — trained on actual padel footage.

    Input: (B, in_dim, 288, 512) — seq_len frames × 3 RGB concatenated.
    Output: (B, out_dim, 288, 512) — heatmap per frame in sequence.
    """

    def __init__(self, in_dim=27, out_dim=8):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.down_block_1 = Double2DConv(in_dim, 64)
        self.down_block_2 = Double2DConv(64, 128)
        self.down_block_3 = Triple2DConv(128, 256)
        self.bottleneck = Triple2DConv(256, 512)
        self.up_block_1 = Triple2DConv(768, 256)  # 512 + 256 skip
        self.up_block_2 = Double2DConv(384, 128)   # 256 + 128 skip
        self.up_block_3 = Double2DConv(192, 64)    # 128 + 64 skip
        self.predictor = nn.Conv2d(64, out_dim, (1, 1))
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.MaxPool2d((2, 2), stride=(2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.down_block_1(x)
        x = self.pool(x1)
        x2 = self.down_block_2(x)
        x = self.pool(x2)
        x3 = self.down_block_3(x)
        x = self.pool(x3)
        x = self.bottleneck(x)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x3], dim=1)
        x = self.up_block_1(x)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x2], dim=1)
        x = self.up_block_2(x)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x1], dim=1)
        x = self.up_block_3(x)
        x = self.predictor(x)
        x = self.sigmoid(x)
        return x


class TrackNetBallDetector(BallDetector):
    """Ball detector using padel-trained TrackNet with 9-frame temporal input."""

    HEATMAP_W, HEATMAP_H = 512, 288
    SEQ_LEN = 8  # uses 9 frames (seq_len + 1 with bg_mode=concat)
    N_INPUT_FRAMES = 9

    def __init__(self, model: TrackNetV2Model = None, model_path: str = None,
                 conf_threshold: float = 0.5, yolo_fallback=None):
        self._device_str = get_device()
        self._torch_device = torch.device(self._device_str)

        if model is not None:
            self._model = model
        elif model_path is not None:
            self._model = TrackNetV2Model(in_dim=27, out_dim=8)
            checkpoint = torch.load(model_path, map_location=self._torch_device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                self._model.load_state_dict(checkpoint['model'])
            else:
                self._model.load_state_dict(checkpoint)
        else:
            raise ValueError("Either model or model_path must be provided")

        self._model.to(self._torch_device)
        self._model.eval()
        self._buffer: List[np.ndarray] = []
        self._conf_threshold = conf_threshold
        self._yolo_fallback = yolo_fallback

    def detect(self, frame: np.ndarray, frame_id: int = 0) -> Optional[List[float]]:
        self._buffer.append(frame)
        if len(self._buffer) > self.N_INPUT_FRAMES:
            self._buffer.pop(0)
        if len(self._buffer) < self.N_INPUT_FRAMES:
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
        dummy = [np.zeros((self.HEATMAP_H, self.HEATMAP_W, 3), dtype=np.uint8)] * self.N_INPUT_FRAMES
        self._infer(dummy)

    @property
    def device(self) -> str:
        return self._device_str

    def _infer(self, frames: List[np.ndarray]) -> np.ndarray:
        resized = []
        for f in frames:
            r = cv2.resize(f, (self.HEATMAP_W, self.HEATMAP_H))
            r = r.astype(np.float32) / 255.0
            resized.append(r)

        # Stack 9 frames → 27 channels
        stacked = np.concatenate(resized, axis=2)  # (288, 512, 27)
        tensor = torch.from_numpy(stacked).permute(2, 0, 1).unsqueeze(0)  # (1, 27, 288, 512)
        tensor = tensor.to(self._torch_device)

        with torch.no_grad():
            output = self._model(tensor)  # (1, 8, 288, 512)

        # Use last channel (most recent frame's heatmap)
        return output[0, -1].cpu().numpy()  # (288, 512)

    @staticmethod
    def _find_peak(heatmap: np.ndarray):
        smoothed = cv2.GaussianBlur(heatmap, (5, 5), 2.0)
        peak_idx = smoothed.argmax()
        peak_y, peak_x = np.unravel_index(peak_idx, smoothed.shape)
        peak_conf = float(smoothed[peak_y, peak_x])
        return peak_conf, peak_y, peak_x
