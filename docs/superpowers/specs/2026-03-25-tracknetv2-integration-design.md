# TrackNetV2 Integration — Better Ball Detection

## Overview

Replace YOLO cls 32 ball detection (8.4% detection rate) with TrackNetV2, a purpose-built temporal ball detector trained on racket sports footage. Expected improvement: 8% → 60%+, enabling the event detection state machine to actually score matches.

**Scope:** Backend only. New detector class behind existing `BallDetector` interface. No changes to trackers, event detection, API, or frontend.

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Pretrained weights | Tennis/badminton from TrackNet repo | Zero training, prove pipeline works first |
| 3-frame buffering | Internal to detector | No interface changes to BallDetector or VideoAnalyzer |
| Fallback | YOLO cls 32 when TrackNet confidence low | Best of both |
| Fine-tuning | Designed for but not built yet | Need labeled padel data first |

---

## Architecture

### TrackNetV2 Model

TrackNetV2 is a U-Net encoder-decoder network:
- **Input:** 3 consecutive RGB frames concatenated → 9-channel tensor, resized to 640×360
- **Output:** 640×360 single-channel heatmap. Peak location = ball center pixel coordinates.
- **Architecture:** Encoder (VGG16-like conv blocks with max pooling) → Decoder (upsampling + conv blocks with skip connections)
- **Reference:** [TrackNetV2 paper](https://arxiv.org/abs/2007.09124), [GitHub repo](https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2)

### TrackNetBallDetector

Implements `BallDetector` ABC with internal 3-frame buffer:

```python
class TrackNetBallDetector(BallDetector):
    def __init__(self, model_path: str, conf_threshold: float = 0.5,
                 yolo_fallback: Optional[YoloBallDetector] = None):
        self._model = TrackNetV2Model()
        self._model.load_state_dict(torch.load(model_path))
        self._buffer: List[np.ndarray] = []  # last 3 frames
        self._conf_threshold = conf_threshold
        self._yolo_fallback = yolo_fallback  # optional YOLO for low-confidence frames

    def detect(self, frame: np.ndarray, frame_id: int = 0) -> Optional[List[float]]:
        self._buffer.append(frame)
        if len(self._buffer) > 3:
            self._buffer.pop(0)
        if len(self._buffer) < 3:
            # Not enough frames yet — use YOLO fallback if available
            if self._yolo_fallback:
                return self._yolo_fallback.detect(frame, frame_id)
            return None

        # Run TrackNet inference
        heatmap = self._infer(self._buffer)
        peak_conf, peak_x, peak_y = self._find_peak(heatmap)

        if peak_conf >= self._conf_threshold:
            # Convert heatmap coords to original frame coords
            # Return as [x1, y1, x2, y2] bbox (small box around center)
            radius = 10
            return [peak_x - radius, peak_y - radius,
                    peak_x + radius, peak_y + radius]

        # Low confidence — fall back to YOLO
        if self._yolo_fallback:
            return self._yolo_fallback.detect(frame, frame_id)
        return None
```

### Detector Selection in VideoAnalyzer

`VideoAnalyzer.__init__` accepts `detector_type: str = "yolo"`:

- `"yolo"` (default): Current behavior — `UnifiedYoloDetector` → `YoloBallDetector`
- `"tracknet"`: `TrackNetBallDetector` with YOLO fallback for first 2 frames and low-confidence frames

The player detector always uses YOLO regardless of ball detector choice.

### Weight Management

- Weights stored in `backend/models/` directory (gitignored via `*.pt` rule)
- `TrackNetBallDetector` constructor takes `model_path` parameter
- Default path: `models/tracknetv2.pt`
- If weights file doesn't exist, raise a clear error with download instructions
- Weights file: ~30MB (VGG16-based encoder)

---

## TrackNetV2 Model Architecture (PyTorch)

```
Input: [B, 9, 360, 640]  (3 frames × 3 RGB channels)

Encoder:
  Conv2d(9, 64) → Conv2d(64, 64) → MaxPool → 180×320
  Conv2d(64, 128) → Conv2d(128, 128) → MaxPool → 90×160
  Conv2d(128, 256) → Conv2d(256, 256) → Conv2d(256, 256) → MaxPool → 45×80
  Conv2d(256, 512) → Conv2d(512, 512) → Conv2d(512, 512) → MaxPool → 22×40

Decoder (with skip connections from encoder):
  Upsample → 45×80 → Conv2d(512, 256) → Conv2d(256, 256) → Conv2d(256, 256)
  Upsample → 90×160 → Conv2d(256, 128) → Conv2d(128, 128)
  Upsample → 180×320 → Conv2d(128, 64) → Conv2d(64, 64)
  Upsample → 360×640 → Conv2d(64, 1) → Sigmoid

Output: [B, 1, 360, 640]  (heatmap, 0-1 probability)
```

### Heatmap → Ball Position

1. Apply Gaussian blur (σ=2) to smooth noise
2. Find global maximum: `peak_y, peak_x = np.unravel_index(heatmap.argmax(), heatmap.shape)`
3. Confidence = heatmap value at peak
4. Scale coordinates from heatmap resolution (640×360) back to original frame resolution
5. Return as bbox: `[peak_x - r, peak_y - r, peak_x + r, peak_y + r]` where r=10 pixels

---

## File Structure

```
backend/
├── src/cv/detectors/
│   ├── tracknet.py          # TrackNetV2Model + TrackNetBallDetector
│   └── (existing files unchanged)
├── src/pipeline/
│   └── video_analyzer.py    # Add detector_type parameter
├── models/                  # Weight files (gitignored)
│   └── tracknetv2.pt        # Downloaded pretrained weights
└── main.py                  # Pass detector_type from API
```

### Changes to Existing Files

- `src/pipeline/video_analyzer.py`: Add `detector_type` parameter to `__init__`. When `"tracknet"`, create `TrackNetBallDetector` with YOLO fallback instead of `YoloBallDetector`.
- `backend/main.py`: Add optional `detector_type` field to analysis start endpoint so user can choose.
- No changes to: `BallDetector` ABC, `BallTracker`, event detectors, state machine, frontend.

---

## Testing Strategy

- **Unit tests for TrackNetV2Model**: Verify forward pass shape (9-channel input → 1-channel heatmap output)
- **Unit tests for TrackNetBallDetector**:
  - Returns None for first 2 frames (buffering)
  - Returns bbox from frame 3 onward (with mock model)
  - Falls back to YOLO when confidence low
  - Falls back to YOLO for first 2 frames when fallback provided
- **Integration test**: Run TrackNetBallDetector through VideoAnalyzer pipeline with mock model, verify FrameResult output
- Tests use a mock model (random heatmap with planted peak) — no real weights needed for tests

## Dependencies

No new pip packages. TrackNetV2 is pure PyTorch (Conv2d, BatchNorm, Upsample) — all already in `torch`.
