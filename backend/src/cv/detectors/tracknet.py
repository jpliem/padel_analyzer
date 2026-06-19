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


class LegacyConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        return self.block(x)


class LegacyTrackNetModel(nn.Module):
    """Older 3-frame TrackNet architecture used by `tracknet_tennis.pt`."""

    def __init__(self):
        super().__init__()
        self.conv1 = LegacyConvBlock(9, 64)
        self.conv2 = LegacyConvBlock(64, 64)
        self.conv3 = LegacyConvBlock(64, 128)
        self.conv4 = LegacyConvBlock(128, 128)
        self.conv5 = LegacyConvBlock(128, 256)
        self.conv6 = LegacyConvBlock(256, 256)
        self.conv7 = LegacyConvBlock(256, 256)
        self.conv8 = LegacyConvBlock(256, 512)
        self.conv9 = LegacyConvBlock(512, 512)
        self.conv10 = LegacyConvBlock(512, 512)
        self.conv11 = LegacyConvBlock(512, 256)
        self.conv12 = LegacyConvBlock(256, 256)
        self.conv13 = LegacyConvBlock(256, 256)
        self.conv14 = LegacyConvBlock(256, 128)
        self.conv15 = LegacyConvBlock(128, 128)
        self.conv16 = LegacyConvBlock(128, 64)
        self.conv17 = LegacyConvBlock(64, 64)
        self.conv18 = LegacyConvBlock(64, 256)
        self.pool = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.up(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.up(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.up(x)
        x = self.conv16(x)
        x = self.conv17(x)
        return self.conv18(x)


def _state_dict_from_checkpoint(checkpoint):
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def build_tracknet_model_from_state_dict(state_dict):
    if "down_block_1.conv_1.conv.weight" in state_dict:
        return TrackNetV2Model(in_dim=27, out_dim=8), 9
    if "conv1.block.0.weight" in state_dict:
        return LegacyTrackNetModel(), 3
    raise ValueError("Unsupported TrackNet checkpoint architecture")


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
            self.N_INPUT_FRAMES = getattr(model, "n_input_frames", self.N_INPUT_FRAMES)
        elif model_path is not None:
            checkpoint = torch.load(model_path, map_location=self._torch_device, weights_only=False)
            state_dict = _state_dict_from_checkpoint(checkpoint)
            self._model, self.N_INPUT_FRAMES = build_tracknet_model_from_state_dict(state_dict)
            self._model.load_state_dict(state_dict)
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

        # Stack temporal RGB frames into channels.
        stacked = np.concatenate(resized, axis=2)  # (288, 512, 27)
        tensor = torch.from_numpy(stacked).permute(2, 0, 1).unsqueeze(0)  # (1, 27, 288, 512)
        tensor = tensor.to(self._torch_device)

        with torch.no_grad():
            output = self._model(tensor)

        # New padel weights output sequence heatmaps; legacy tennis weights
        # output a 256-channel response volume. Reduce both to one heatmap.
        if output.shape[1] == 1:
            heatmap = output[0, 0]
        elif output.shape[1] <= self.N_INPUT_FRAMES:
            heatmap = output[0, -1]
        else:
            heatmap = output[0].amax(dim=0)
        return heatmap.cpu().numpy()

    @staticmethod
    def _find_peak(heatmap: np.ndarray):
        smoothed = cv2.GaussianBlur(heatmap, (5, 5), 2.0)
        peak_idx = smoothed.argmax()
        peak_y, peak_x = np.unravel_index(peak_idx, smoothed.shape)
        peak_conf = float(smoothed[peak_y, peak_x])
        return peak_conf, peak_y, peak_x
