# Phase 1: honest single-camera baseline

## ELI5

The analyzer has two jobs:

1. **Eyes:** find the tiny ball, players, bounces, hits, walls, doors, and moments
   when the ball is hidden.
2. **Referee:** apply padel rules to those observations.

The referee is now substantially implemented and tested. The eyes are not yet
accurate enough. A missing ball detection no longer means `OUT`; it means “the
camera is uncertain.” Low-confidence point calls go to a review queue and do
not change the score until confirmed.

## What is implemented

- A semantic padel rule engine covering first/second serve, service faults,
  lets, valid service glass contacts, fence faults, receiver volleys, rally
  bounces, wall/fence ordering, interference, and authorised gate exits.
- Explicit ball states: visible, occluded, outside the field of view, outside
  the court but recoverable, and unknown.
- Court doors and configurable outside-play safety zones.
- A gravity-constrained, uncertainty-reporting 3D arc fit from one calibrated
  camera. It only replaces the 2D estimate when the fit passes reliability
  checks.
- Player tracking plus an appearance gallery to reconnect identities after a
  tracker ID changes.
- Audio impulse candidates and multi-signal contact fusion. Audio is evidence,
  never a score command.
- A versioned ball-label schema, browser labeler, validator, rally-safe data
  splits, TrackNet fine-tuning command, and pixel-error evaluation harness.
- A VLM/manual proposal API and review UI. VLM proposals can never auto-confirm.
- An immutable point ledger: correcting an old call rebuilds the score from
  confirmed point history.

## Current accuracy evidence

There are now 197 human-reviewed Panasonic labels across three separate
rallies: 169 visible and 28 occluded. Whole rallies are assigned to train,
validation, and test, so neighboring frames cannot leak between splits.

Using real consecutive nine-frame TrackNet inputs and a 15-pixel correctness
threshold, the bundled `tracknet_padel.pt` checkpoint measured:

- validation rally: 59.3% precision and 59.3% recall;
- held-out test rally: 63.5% precision and 63.5% recall.

These numbers are valid for this reviewed set but are not a cross-camera or
production-accuracy claim. On the test rally, 28 of 52 visible labels exactly
matched reviewer-accepted TrackNet suggestions, so the test can be anchored
toward the baseline model. A future gold test set should be labeled without
model suggestions or independently audited.

The simple color/motion baseline measured only 1.4% precision and 1.7% recall
on the validation rally, commonly selecting shoes, heads, player clothing, and
reflections. A first unweighted fine-tune collapsed to zero recall because
background pixels overwhelmed the tiny ball heatmap. A positive-weighted,
frozen-backbone candidate avoided collapse but reached 57.6% validation
precision/recall, below the 59.3% baseline, so it was rejected. Production
continues to use `tracknet_padel.pt`.

PadelVic's synthetic CSV coordinates are Xsens positional ground truth and
visually align with the motion-captured player's root/feet, not the ball. The
evaluator refuses those CSVs by default. The earlier 9% / 141 px result was a
target-schema error and must not be cited.

## Single camera versus multiple cameras

| Capability | One calibrated camera | Synchronized multiple cameras |
|---|---|---|
| Ball/player 2D tracking | Yes | Yes, with redundancy |
| Court coordinates | Good on the floor plane | Better and less occlusion-prone |
| Ball height/depth | Approximate temporal physics fit, with uncertainty | Direct triangulation when cameras see the ball |
| Hidden/out-of-frame ball | Predict briefly, then require review | Another view may observe it |
| Glass/net/fence ordering | Possible but difficult; fuse motion, audio, geometry | Much stronger evidence |
| Player identity | Appearance + motion + court constraints | Cross-camera ReID is still required |
| Automatic official scoring | Only after measured perception thresholds are met | More achievable, still needs rules and review |

PadelVic's camera recordings are approximately synchronized, not guaranteed
frame-accurate. Phase 1 treats Panasonic, GoPro, Samsung, and iPhone as four
independent single-camera tests. It does **not** claim their frames are valid
multiview triangulation ground truth.

## What still prevents an accuracy claim

1. Add more independently reviewed training rallies and a suggestion-free gold
   test set containing shoes, bald heads, lights, reflections, blur, and
   occlusions as explicit hard cases.
2. Fine-tune again only when there are several training rallies; keep a model
   only if it beats the frozen baseline on untouched validation and test sets.
3. Evaluate the same frozen model independently on all four PadelVic views.
4. Tune pose/ReID/contact thresholds from labeled data. The Panasonic master
   has a silent AAC track, so audio evidence is unavailable for this view.
5. Measure complete rally and point-decision accuracy, not only pixel error.

The full 5.2 GB Panasonic master is available locally: 73:08, 3626x1960 at
50 FPS. GoPro, Samsung, and iPhone views are still unavailable, so cross-view
accuracy remains unmeasured rather than guessed.

Reviewed manifests and evaluation reports are under `data/labels/`. The merged
training manifest is `padelvic_panasonic_combined/labels.json`.

## Reproducible commands

From `backend/`:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python ../scripts/validate_ball_labels.py LABELS.json --assign-splits
.venv/bin/python ../scripts/prelabel_ball_suggestions.py LABELS.json --detector fast
.venv/bin/python ../scripts/merge_ball_labels.py LABELS_A.json LABELS_B.json --output COMBINED.json
.venv/bin/python ../scripts/train_temporal_ball.py --labels COMBINED.json --output models/tracknet_phase1_candidate.pt --init-checkpoint models/tracknet_padel.pt --freeze-backbone --positive-weight 250 --input-width 256 --input-height 144
.venv/bin/python ../scripts/eval_ball_labels.py --labels LABELS.json --detector tracknet
.venv/bin/python ../scripts/eval_single_camera_views.py
```

The frontend production bundle is checked with:

```bash
node node_modules/react-scripts/bin/react-scripts.js build
```
