# Padel Analyzer: product overview

> **Current R&D summary:** see
> [current-vlm-scoring-status.md](current-vlm-scoring-status.md) for the
> plain-English account of the Qwen, OpenCV, audio, and scoring experiments and
> what must happen next.

> **New product direction:** the VLM-first coaching experience is implemented as
> a separate application. See [vlm-match-coach.md](vlm-match-coach.md). The
> OpenCV referee-oriented application described below remains available as the
> earlier experimental system.

## What is this app?

Padel Analyzer is a **padel match analysis and training-review app**. A player,
coach, or club can upload a fixed-camera recording of a match, review what
happened, inspect suggested events and highlights, and correct uncertain
decisions.

It can support training by helping people study positioning, rallies, mistakes,
and important moments after a match. It is not currently a drill planner,
coaching curriculum, or fully automatic electronic referee.

## ELI5

The app is being built with two main parts:

1. **The eyes** try to find the court, players, and tiny moving ball in a video.
2. **The referee brain** turns reliable observations into padel events and score
   decisions.

The referee brain is substantially implemented and tested. The eyes are still
the limiting factor: a fast ball can be blurred, hidden, or confused with a
racket, shoe, head, light, fence detail, or glass reflection.

Because of that limitation, uncertain point decisions are sent to a human
review queue instead of silently changing the score.

## Current user workflow

The single-camera web app currently supports:

1. Creating a match and entering player names.
2. Uploading a fixed-court recording.
3. Automatically detecting the court or calibrating it manually.
4. Running ball, player, and event analysis in the background.
5. Watching the original or annotated recording.
6. Seeking to suggested rallies, events, and highlights.
7. Confirming or rejecting uncertain decisions and adding manual points.
8. Exporting the event table as CSV or the complete analysis as JSON.

Saved matches, analysis results, and corrections survive an API restart.

## What is reliable today?

- The web workflow, court calibration, saved matches, review queue, corrections,
  highlights, and exports are implemented.
- The padel rules engine covers detailed serve and rally behavior and has focused
  automated tests.
- Player and ball tracking produce useful analysis candidates, but neither
  should be treated as an official match record without review.
- The current ball-model result is approximately 63.5% precision and recall
  within 15 pixels on one small held-out Panasonic rally. This is a limited
  same-camera experiment, not complete scoring accuracy or proof that the model
  generalizes to other courts.

## Why isn't it an automatic referee yet?

Every later decision depends on seeing the correct ball:

```text
ball detection
  -> bounce, hit, wall, and net detection
  -> rally interpretation
  -> point winner
  -> score
```

A wrong or missing ball position can make the whole chain wrong. One camera also
cannot directly measure the exact 3D position of an airborne ball. The app can
make cautious physics-based estimates, but occlusions, reflections, glass
contacts, and close line decisions can remain ambiguous.

Multiple synchronized cameras can recover real 3D positions when they are
calibrated and both identify the correct ball. That machinery has been proven in
small engineering tests, but it remains research tooling rather than the main
product workflow.

## Current product position

The honest description is:

> A smart padel match recorder, analysis assistant, and training-review tool
> with human confirmation for uncertain decisions.

The app should not yet be marketed as a hands-off automatic padel referee or as
having validated official scoring accuracy.

## Development priority

The next priority is improving the app's eyesight:

1. Collect more independently reviewed padel-ball labels across different
   rallies, cameras, courts, players, and lighting conditions.
2. Maintain a suggestion-free gold test set containing blur, occlusions,
   reflections, spare balls, shoes, rackets, and other hard negatives.
3. Improve the ball detector and keep changes only when they beat the frozen
   baseline on untouched data.
4. Measure complete rally and point-decision accuracy, not only pixel error.
5. Expand automatic scoring only after those measurements demonstrate that it
   is dependable.

For operating details, see [smart-recording-webapp.md](smart-recording-webapp.md).
For the current perception evidence, see
[phase1_single_camera.md](phase1_single_camera.md).
