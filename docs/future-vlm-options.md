# Future / Alternative Options — VLM Semantic Layer

Status: **exploratory.** Captured 2026-06-16; updated 2026-06-18 after a local Molmo2 probe attempt. Current pipeline stays YOLO/TrackNet + geometry. This documents a VLM augmentation to revisit later.

## Why a VLM at all

The geometric pipeline (YOLO/TrackNet for detection, homography + multi-camera triangulation for 3D) handles *where* things are. It does not natively handle *semantic* questions:

- Which player served? Which player hit the smash?
- What shot type was that (smash / bandeja / volley / lob / chiquita)?
- Why did the point end (out, double bounce, into glass before bounce, net)?
- Highlight tagging / rally segmentation for replay.

These are language-conditioned, frame-window-scale tasks. A small VLM is a good fit; geometry is not.

## What a VLM is NOT for (settled)

| Task | Tool | Reason VLM loses |
|------|------|------------------|
| Ball detection / tracking | YOLO / TrackNet | Ball is tiny, fast, blurry. VLM point-grounding targets salient describable objects; ball is sub-salient. |
| Per-frame realtime (30fps) | YOLO / TrackNet | A 4B+ VLM forward pass per frame is not realtime on normal hardware. VLM tracking demos run at ~0.5s+ timestamp gaps. |
| 3D depth / bounce-vs-flyover | Multi-camera triangulation | VLM outputs 2D points only. No metric depth. Same monocular ambiguity as a single camera. |
| Plain player tracking | YOLO + ByteTrack | Already cheaper and faster. VLM only wins when identity must be tied to a *role/description*. |

## Candidate models

- **Molmo2** (AI2, Apache-2.0). Variants 4B / 7B-O / 8B. Does video understanding, text-driven pointing, and object tracking (point/box at sparse timestamps). Smallest = 4B (not 0.8B).
- **Molmo2-VideoPoint-4B** (`allenai/Molmo2-VideoPoint-4B`) — the most relevant Molmo2 variant for point grounding/counting. A local attempt reached checkpoint download but did not reach inference; four checkpoint shards stayed at 0% after 7m41s.
- **Qwen3-VL / Qwen2.5-VL-3B** — general VLM, strong VQA + grounding.

Note: there is no ~0.8B padel-grade VLM. Smallest realistic tier is 3-4B.

## Proposed hybrid architecture (when revisited)

```
                     [ video frames ]
                            |
        +-------------------+-------------------+
        |  GEOMETRIC LAYER (realtime, per-frame) |
        |  YOLO/TrackNet ball + player detect    |
        |  BallTracker / PlayerTracker (Kalman)  |
        |  Multi-cam WorldFusion -> 3D state     |
        |  EventDetector (bounce/serve/wall/...) |
        +-------------------+-------------------+
                            |
                  key segments / clips only
                  (rally start->end, candidate events)
                            |
        +-------------------v-------------------+
        |  SEMANTIC LAYER (sparse, async)        |
        |  VLM (Molmo2-4B / Qwen3-VL-3B)         |
        |  - shot-type classification            |
        |  - player-by-role tracking             |
        |  - point-end reason (verify vs geom)   |
        |  - highlight / rally tagging           |
        +----------------------------------------+
```

Key rules:
- VLM runs on **cropped key segments**, never per-frame realtime.
- VLM outputs are **labels/explanations**, validated against geometric truth (geometry wins on conflict — VLM hallucinates).
- Plugs in as a new detector alongside `logic/detectors/`, consuming clips around `EventDetector` triggers.

## Open questions to resolve before committing

1. Molmo2-4B / Molmo2-VideoPoint-4B download, inference latency, and VRAM on target hardware — confirm "sparse only" is required.
2. Does a VLM beat a small supervised shot-classifier (e.g. CNN/temporal model on PadelTracker100 shot-event labels)? VLM = zero-shot/few-shot convenience vs trained accuracy.
3. Annotation cost for fine-tuning vs zero-shot prompting.

## Local Probe

Use `scripts/vlm_ball_probe.py` when the checkpoint is available locally:

```bash
backend/venv/bin/python scripts/vlm_ball_probe.py \
  --video /tmp/padelvic_synthetic_2s.mp4 \
  --model-id allenai/Molmo2-VideoPoint-4B \
  --prompt "Point to the small yellow padel ball in the video. Return points only." \
  --out /tmp/vlm_ball_probe_molmo2.json
```

The script also supports `--raw-text` to test parsing without loading the VLM.

## Related

- Core 3D problem and multi-cam direction: see `docs/superpowers/specs/2026-03-26-multi-camera-wall-detection-design.md`.
- Datasets: PADELVIC (multi-cam + mocap), PadelTracker100 (single-cam pro, has shot-event labels — useful for both VLM eval and a trained classifier).
