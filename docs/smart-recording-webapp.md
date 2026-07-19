# Single-camera smart recording web app

Padel Analyzer is a **padel match analysis and training-review app**. Its current
product position, capabilities, and limitations are summarized in
[product-overview.md](product-overview.md).

## Run it

From the repository root:

```bash
bash run_webapp.sh
```

Open `http://localhost:3000`, select **New Match**, and upload a fixed-court
recording. The app will try automatic court calibration. When it cannot find
the lines confidently, it opens the manual court-point screen using the
already-uploaded recording.

The launcher uses API port 8000 when available and automatically selects the
next free port when another local service already owns it.

## Product workflow

1. Upload one MP4/MOV recording and enter player names.
2. Automatically detect the court or confirm it manually.
3. Run player and active-ball tracking in the background.
4. Review the original or annotated recording.
5. Seek to detected rallies and download individual clips.
6. Confirm/reject uncertain point decisions or add a manual point.
7. Download the event table as CSV or the full analysis as JSON.

Uploads, job state, results, corrections, and review decisions are stored in
`backend/data/matches/<match-id>/`. Completed matches survive an API restart.
An interrupted analysis becomes a visible retryable error instead of remaining
stuck as “processing”.

## Honest single-camera boundary

This is a smart recorder and review assistant. Court homography gives useful
top-view coordinates on the ground plane, but one camera cannot uniquely
measure depth or recover every hidden ball. Glass reflections, occlusion,
lighting, balls lying on the floor, net contact, and close line decisions can
still be ambiguous. The result screen therefore exposes confidence/review
state and does not claim that uncertain events are automatically correct.

## Intelligence actually connected to the app

- The production candidate is `backend/models/tracknet_padel.pt`. It matched
  33 of 52 visible reviewed labels (63.5%) within 15 pixels on one held-out
  Panasonic rally. This is a small, same-view detector benchmark—not scoring
  accuracy and not proof of club-to-club generalization.
- TrackNet now exposes several ball-like heatmap peaks. A temporal active-ball
  selector rejects spare balls, reflections, and candidates that require an
  implausible image-space jump. Ambiguous frames remain uncertain.
- Audio is decoded when present and its impulses can support a contact
  proposal. Silent Panasonic audio is reported as silent. Audio never awards a
  point by itself.
- Detected serves, faults, bounces, player hits, wall contacts, and net contacts
  are translated into semantic observations for the padel rules engine. The
  engine models first/second service, service lets, legal returns, rally
  contacts, exits, and double bounces. Perception-derived point decisions stay
  in the review ledger until scoring accuracy is independently validated.
- Player re-identification, court homography, visibility state, and cautious
  monocular trajectory fitting run in the single-camera pipeline. Pose evidence
  is only available when a pose checkpoint is explicitly configured.

CalTennis triangulation, tennis/table-tennis dataset work, and VLM probing are
research tools, not hidden claims about the single-camera runtime. The result
screen lists these boundaries under **What this build really uses**.

## Verification

```bash
cd backend && .venv/bin/python -m pytest -q
cd frontend && npm run build
```

The smart-recording API smoke tests are in
`backend/tests/test_smart_recording_api.py`.
