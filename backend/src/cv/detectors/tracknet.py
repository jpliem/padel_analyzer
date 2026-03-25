import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Optional, List
from cv.detectors.base import BallDetector
from cv.detectors.device import get_device


class Conv(nn.Module):
    """Conv + ReLU + BatchNorm block.

    The bc (batch-norm channels) parameter exists because the pretrained weights
    were converted from TensorFlow where BatchNorm runs on the last axis (W dimension
    after NCHW→NWHC transpose), not the channel dimension.
    """
    def __init__(self, ic, oc, bc, k=(3, 3), p="same", act=True):
        super().__init__()
        self.conv = nn.Conv2d(ic, oc, kernel_size=k, padding=p)
        self.bn = nn.BatchNorm2d(bc)
        self.act = nn.ReLU() if act else nn.Identity()

    def forward(self, x):
        x = self.act(self.conv(x))
        x = x.transpose(1, 3)    # NCHW → NWHC
        x = self.bn(x)
        x = x.transpose(1, 3)    # NWHC → NCHW
        return x


class TrackNetV2Model(nn.Module):
    """TrackNetV2: VGG16-based U-Net with concatenation skip connections.

    Architecture matches ChgygLin/TrackNetV2-pytorch tf2torch weights.
    Input: (B, 9, H, W) — 3 consecutive RGB frames concatenated.
    Output: (B, 3, H, W) — 3 heatmaps (one per input frame).
    """

    def __init__(self):
        super().__init__()
        # Encoder (VGG16-like) — bc values match pretrained weight dimensions
        self.conv2d_1 = Conv(9, 64, 512)
        self.conv2d_2 = Conv(64, 64, 512)
        self.max_pooling_1 = nn.MaxPool2d((2, 2), stride=(2, 2))

        self.conv2d_3 = Conv(64, 128, 256)
        self.conv2d_4 = Conv(128, 128, 256)
        self.max_pooling_2 = nn.MaxPool2d((2, 2), stride=(2, 2))

        self.conv2d_5 = Conv(128, 256, 128)
        self.conv2d_6 = Conv(256, 256, 128)
        self.conv2d_7 = Conv(256, 256, 128)
        self.max_pooling_3 = nn.MaxPool2d((2, 2), stride=(2, 2))

        self.conv2d_8 = Conv(256, 512, 64)
        self.conv2d_9 = Conv(512, 512, 64)
        self.conv2d_10 = Conv(512, 512, 64)

        # Decoder with concatenation skip connections
        self.up_sampling_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv2d_11 = Conv(768, 256, 128)
        self.conv2d_12 = Conv(256, 256, 128)
        self.conv2d_13 = Conv(256, 256, 128)

        self.up_sampling_2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv2d_14 = Conv(384, 128, 256)
        self.conv2d_15 = Conv(128, 128, 256)

        self.up_sampling_3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv2d_16 = Conv(192, 64, 512)
        self.conv2d_17 = Conv(64, 64, 512)
        self.conv2d_18 = nn.Conv2d(64, 3, kernel_size=(1, 1), padding='same')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x = self.conv2d_1(x)
        x1 = self.conv2d_2(x)
        x = self.max_pooling_1(x1)

        x = self.conv2d_3(x)
        x2 = self.conv2d_4(x)
        x = self.max_pooling_2(x2)

        x = self.conv2d_5(x)
        x = self.conv2d_6(x)
        x3 = self.conv2d_7(x)
        x = self.max_pooling_3(x3)

        x = self.conv2d_8(x)
        x = self.conv2d_9(x)
        x = self.conv2d_10(x)

        # Decoder
        x = self.up_sampling_1(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv2d_11(x)
        x = self.conv2d_12(x)
        x = self.conv2d_13(x)

        x = self.up_sampling_2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv2d_14(x)
        x = self.conv2d_15(x)

        x = self.up_sampling_3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv2d_16(x)
        x = self.conv2d_17(x)
        x = self.conv2d_18(x)

        return torch.sigmoid(x)


class TrackNetBallDetector(BallDetector):
    """Ball detector using TrackNetV2 with 3-frame temporal input and optional YOLO fallback."""

    HEATMAP_W, HEATMAP_H = 512, 288

    def __init__(self, model: TrackNetV2Model = None, model_path: str = None,
                 conf_threshold: float = 0.5, yolo_fallback=None):
        self._device_str = get_device()
        self._torch_device = torch.device(self._device_str)

        if model is not None:
            self._model = model
        elif model_path is not None:
            self._model = TrackNetV2Model()
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

        stacked = np.concatenate(resized, axis=2)  # (H, W, 9)
        tensor = torch.from_numpy(stacked).permute(2, 0, 1).unsqueeze(0)  # (1, 9, H, W)
        tensor = tensor.to(self._torch_device)

        with torch.no_grad():
            output = self._model(tensor)

        # Output is (1, 3, H, W) — use last channel (heatmap for most recent frame)
        return output[0, 2].cpu().numpy()

    @staticmethod
    def _find_peak(heatmap: np.ndarray):
        smoothed = cv2.GaussianBlur(heatmap, (5, 5), 2.0)
        peak_idx = smoothed.argmax()
        peak_y, peak_x = np.unravel_index(peak_idx, smoothed.shape)
        peak_conf = float(smoothed[peak_y, peak_x])
        return peak_conf, peak_y, peak_x
