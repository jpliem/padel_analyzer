# TrackNetV2 Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace YOLO ball detection (8.4%) with TrackNetV2 temporal detector (~60%+) behind the existing BallDetector interface.

**Architecture:** New `TrackNetV2Model` (PyTorch U-Net) + `TrackNetBallDetector` (implements `BallDetector` ABC with internal 3-frame buffer and YOLO fallback). `VideoAnalyzer` gets a `detector_type` parameter to choose between YOLO and TrackNet.

**Tech Stack:** PyTorch (existing), no new dependencies

**Spec:** `docs/superpowers/specs/2026-03-25-tracknetv2-integration-design.md`

**Working directory:** `/Users/jonathan/Documents/Github/padel_analyzer/backend/`

**Test command:** `cd /Users/jonathan/Documents/Github/padel_analyzer && source venv/bin/activate && cd backend && python -m pytest tests/ -v --tb=short`

---

## File Map

### New Files
| File | Responsibility |
|------|---------------|
| `src/cv/detectors/tracknet.py` | `TrackNetV2Model` (PyTorch network) + `TrackNetBallDetector` (BallDetector impl) |
| `tests/test_tracknet.py` | Unit tests for model shape + detector behavior |

### Modified Files
| File | Changes |
|------|---------|
| `src/pipeline/video_analyzer.py` | Add `detector_type` parameter, create TrackNet when selected |
| `main.py` | Pass `detector_type` through analysis endpoints |

---

### Task 1: TrackNetV2 Model (PyTorch Network)

**Files:**
- Create: `src/cv/detectors/tracknet.py`
- Create: `tests/test_tracknet.py`

- [ ] **Step 1: Write failing test for model forward pass**

Create `tests/test_tracknet.py`:

```python
import pytest
import torch
import numpy as np


class TestTrackNetV2Model:
    def test_forward_pass_shape(self):
        from cv.detectors.tracknet import TrackNetV2Model
        model = TrackNetV2Model()
        model.eval()
        # Input: batch=1, 9 channels (3 frames × 3 RGB), 360×640
        x = torch.randn(1, 9, 360, 640)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, 360, 640)

    def test_output_range_0_to_1(self):
        from cv.detectors.tracknet import TrackNetV2Model
        model = TrackNetV2Model()
        model.eval()
        x = torch.randn(1, 9, 360, 640)
        with torch.no_grad():
            out = model(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_batch_support(self):
        from cv.detectors.tracknet import TrackNetV2Model
        model = TrackNetV2Model()
        model.eval()
        x = torch.randn(2, 9, 360, 640)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 1, 360, 640)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_tracknet.py::TestTrackNetV2Model -v --tb=short`
Expected: FAIL — module not found

- [ ] **Step 3: Implement TrackNetV2Model**

Create `src/cv/detectors/tracknet.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_tracknet.py::TestTrackNetV2Model -v --tb=short`
Expected: All 3 pass

- [ ] **Step 5: Commit**

```bash
git add src/cv/detectors/tracknet.py tests/test_tracknet.py
git commit -m "feat: TrackNetV2 PyTorch model (U-Net encoder-decoder for ball detection)"
```

---

### Task 2: TrackNetBallDetector

**Files:**
- Modify: `src/cv/detectors/tracknet.py` (add TrackNetBallDetector class)
- Modify: `tests/test_tracknet.py` (add detector tests)

- [ ] **Step 1: Write failing tests for detector**

Add to `tests/test_tracknet.py`:

```python
from unittest.mock import MagicMock, patch


class TestTrackNetBallDetector:
    def _make_detector_with_mock_model(self, peak_conf=0.9, peak_x=320, peak_y=180):
        """Create detector with a mock model that returns a heatmap with a planted peak."""
        from cv.detectors.tracknet import TrackNetBallDetector, TrackNetV2Model

        # Create a mock model that returns a heatmap with a peak at the specified location
        mock_model = MagicMock(spec=TrackNetV2Model)
        heatmap = torch.zeros(1, 1, 360, 640)
        heatmap[0, 0, peak_y, peak_x] = peak_conf
        mock_model.return_value = heatmap
        mock_model.eval = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)

        detector = TrackNetBallDetector.__new__(TrackNetBallDetector)
        detector._model = mock_model
        detector._device_str = "cpu"
        detector._torch_device = torch.device("cpu")
        detector._buffer = []
        detector._conf_threshold = 0.5
        detector._yolo_fallback = None
        return detector

    def test_returns_none_for_first_two_frames(self):
        detector = self._make_detector_with_mock_model()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        assert detector.detect(frame, 0) is None
        assert detector.detect(frame, 1) is None

    def test_returns_bbox_on_third_frame(self):
        detector = self._make_detector_with_mock_model(peak_conf=0.9, peak_x=320, peak_y=180)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        detector.detect(frame, 0)
        detector.detect(frame, 1)
        result = detector.detect(frame, 2)
        assert result is not None
        assert len(result) == 4
        # Peak at heatmap (320, 180) scaled to frame (1280, 720):
        # scale_x = 1280/640 = 2, scale_y = 720/360 = 2
        # cx = 320*2 = 640, cy = 180*2 = 360
        expected_cx, expected_cy = 640, 360
        assert abs(result[0] - (expected_cx - 10)) < 2
        assert abs(result[1] - (expected_cy - 10)) < 2

    def test_returns_none_when_low_confidence(self):
        detector = self._make_detector_with_mock_model(peak_conf=0.1)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        detector.detect(frame, 0)
        detector.detect(frame, 1)
        result = detector.detect(frame, 2)
        assert result is None

    def test_falls_back_to_yolo_first_frames(self):
        detector = self._make_detector_with_mock_model()
        mock_yolo = MagicMock()
        mock_yolo.detect.return_value = [100, 200, 120, 220]
        detector._yolo_fallback = mock_yolo

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = detector.detect(frame, 0)
        assert result == [100, 200, 120, 220]
        mock_yolo.detect.assert_called_once()

    def test_falls_back_to_yolo_low_confidence(self):
        detector = self._make_detector_with_mock_model(peak_conf=0.1)
        mock_yolo = MagicMock()
        mock_yolo.detect.return_value = [100, 200, 120, 220]
        detector._yolo_fallback = mock_yolo

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        detector.detect(frame, 0)
        detector.detect(frame, 1)
        result = detector.detect(frame, 2)
        assert result == [100, 200, 120, 220]

    def test_buffer_stays_at_3(self):
        detector = self._make_detector_with_mock_model()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        for i in range(10):
            detector.detect(frame, i)
        assert len(detector._buffer) == 3

    def test_warm_up_does_not_affect_buffer(self):
        detector = self._make_detector_with_mock_model()
        detector.warm_up()
        assert len(detector._buffer) == 0

    def test_device_property(self):
        detector = self._make_detector_with_mock_model()
        assert detector.device == "cpu"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_tracknet.py::TestTrackNetBallDetector -v --tb=short`
Expected: FAIL — TrackNetBallDetector not defined

- [ ] **Step 3: Implement TrackNetBallDetector**

Add to `src/cv/detectors/tracknet.py` (after the TrackNetV2Model class):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_tracknet.py -v --tb=short`
Expected: All pass (3 model + 8 detector = 11)

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All 133 + 11 = 144 pass

- [ ] **Step 6: Commit**

```bash
git add src/cv/detectors/tracknet.py tests/test_tracknet.py
git commit -m "feat: TrackNetBallDetector with 3-frame buffer, YOLO fallback, coordinate scaling"
```

---

### Task 3: Wire TrackNet into VideoAnalyzer

**Files:**
- Modify: `src/pipeline/video_analyzer.py`
- Modify: `tests/test_video_analyzer.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_video_analyzer.py`:

```python
    def test_detector_type_tracknet(self, mock_deps):
        from pipeline.video_analyzer import VideoAnalyzer
        calibration, config = mock_deps

        with patch('pipeline.video_analyzer.UnifiedYoloDetector') as MockUnified, \
             patch('pipeline.video_analyzer.TrackNetBallDetector') as MockTrackNet:
            mock_unified = MockUnified.return_value
            mock_result = MagicMock()
            xyxy = np.array([]).reshape(0, 4)
            mock_result.boxes.xyxy.cpu.return_value.numpy.return_value = xyxy
            mock_result.boxes.cls.cpu.return_value.numpy.return_value = np.array([])
            mock_result.boxes.conf.cpu.return_value.numpy.return_value = np.array([])
            mock_unified.run.return_value = mock_result

            mock_tracknet = MockTrackNet.return_value
            mock_tracknet.detect.return_value = None

            va = VideoAnalyzer(
                match_id="test",
                calibration=calibration,
                config=config,
                detector_type="tracknet",
            )
            # Should have created TrackNetBallDetector
            MockTrackNet.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_video_analyzer.py::TestVideoAnalyzer::test_detector_type_tracknet -v --tb=short`
Expected: FAIL — unexpected keyword argument 'detector_type'

- [ ] **Step 3: Modify VideoAnalyzer**

In `src/pipeline/video_analyzer.py`, update the import and `__init__`:

Add import at top:
```python
from cv.detectors.tracknet import TrackNetBallDetector
```

Change `__init__` signature and detector creation:

```python
class VideoAnalyzer:
    def __init__(self, match_id: str, calibration,
                 config: EventDetectorConfig = None,
                 match_config: MatchConfig = None,
                 detector_type: str = "yolo",
                 tracknet_model_path: str = "models/tracknetv2.pt"):
        config = config or EventDetectorConfig()
        match_config = match_config or MatchConfig()

        # Always create YOLO (needed for player detection + ball fallback)
        unified = UnifiedYoloDetector()
        self.player_detector = YoloPlayerDetector(unified)

        # Ball detector: YOLO or TrackNet
        if detector_type == "tracknet":
            yolo_fallback = YoloBallDetector(unified)
            self.ball_detector = TrackNetBallDetector(
                model_path=tracknet_model_path,
                yolo_fallback=yolo_fallback,
            )
        else:
            self.ball_detector = YoloBallDetector(unified)

        # ... rest unchanged
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_video_analyzer.py -v --tb=short`
Expected: All pass

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add src/pipeline/video_analyzer.py tests/test_video_analyzer.py
git commit -m "feat: VideoAnalyzer supports detector_type='tracknet' with YOLO fallback"
```

---

### Task 4: Wire detector_type Through API

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Update analysis endpoints**

In `main.py`, modify the upload endpoint to accept `detector_type`:

In the upload handler, store it in the job:
```python
@app.post("/analyze/upload")
async def upload_video(match_id: str, file: UploadFile = File(...),
                       detector_type: str = "yolo"):
    _load_match(match_id)
    match_dir = _match_dir(match_id)
    os.makedirs(match_dir, exist_ok=True)
    video_path = os.path.join(match_dir, "video.mp4")
    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    job_id = match_id
    _analysis_jobs[job_id] = {
        "state": "uploaded", "percent": 0,
        "match_id": match_id, "detector_type": detector_type,
    }
    return {"job_id": job_id, "status": "uploaded"}
```

In the start handler, pass it to VideoAnalyzer:
```python
    detector_type = job.get("detector_type", "yolo")
    analyzer = VideoAnalyzer(match_id=match_id, calibration=cal, config=config,
                             detector_type=detector_type)
```

- [ ] **Step 2: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All pass

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "feat: API passes detector_type to VideoAnalyzer (yolo or tracknet)"
```
