# Current VLM scoring status

This page is the short, plain-English summary of the work so far. For commands,
raw measurements, and experiment details, see
[scoring-architecture-experiments.md](scoring-architecture-experiments.md).

## ELI5: what are we building?

We are testing whether a Mac can watch a fixed-camera padel recording and turn
it into useful rally notes and a tentative score.

The app is **not training an AI model**. It runs an already-trained vision
language model (VLM) locally and asks it questions about chronological images
from the video. It is also **not production-ready as an automatic referee**:
the models can describe obvious play, but cannot yet identify every rally end
or winner reliably enough to change the official score without review.

There are currently two related applications:

- The original web app uses conventional computer vision for court, player,
  ball, event, and scoring experiments.
- The newer `vlm_coach` app samples a video, sends ordered images to local Qwen,
  validates the returned JSON, and builds rally notes and a match story.

## How our direction changed

We began with a long conventional pipeline: find the court and tiny ball, infer
hits and bounces, then calculate the score. Ball detection is too fragile to be
the sole source of truth from one distant camera.

We then tested the opposite extreme: give batches of images or native video to
a VLM and ask it to understand everything. This is simpler, but the small local
models sometimes invent events, contradict themselves, or miss the exact rally
boundary.

The best current direction is a **hybrid**. Cheap signals find moments worth
looking at; the VLM interprets only those moments; ordinary Python applies the
padel scoring rules after a winner has been confirmed.

```text
video
  -> OpenCV finds low-motion candidate gaps
  -> audio quietness removes some unlikely gaps
  -> chronological frames are packed into 2x2 panels
  -> Qwen 0.8B performs quick rally/reset/unclear triage
  -> unclear cases go to Qwen 2B or a person
  -> Qwen 2B reviews the complete rally and suggests the winner
  -> deterministic Python validates and updates the score
```

Court calibration is implemented in the original app, but the new VLM scoring
experiment does not currently depend on it. OpenCV is still useful as a cheap
candidate sensor; it is no longer expected to understand the whole point by
itself.

## What we tested and learned

| Experiment | Result | Meaning |
| --- | --- | --- |
| Qwen 2B, 10 seconds, 8 images | 21.376 s | Runs on the M2/8 GB Mac, but slower than real time. |
| Qwen 0.8B, 10 seconds, 10 images | 12.817 s | Nearly real time and useful for first-pass triage. |
| Native-video Qwen, 6 seconds | About 53 s | Slow, with contradictory or invalid evidence; do not use for now. |
| 2x2 panels, 10 frames | 12.59 s vs 15.95 s | About 21% faster than sending separate images, with the same decision in this test. |
| 2x2 panels, 14 frames | 13.02 s vs 18.95 s | About 31% faster in this test. |
| OpenCV plus audio | Kept 16 of 43 gaps | Reduced VLM calls by about 63%, but accuracy is not known yet. |
| Molmo2 4B 4-bit | Out of memory | Too large for this 8 GB Mac; it was removed. |

Images are passed to Qwen in chronological order. More images can help show
motion, but 60 full-resolution frames at once would add memory and visual-token
cost without guaranteeing better reasoning. The practical experiment is about
one sampled frame per second, grouped into labelled chronological panels, with
overlapping windows and a small structured state carried into the next window.
We should carry facts such as “rally still active”, not the model's previous
prose, because prose caused it to copy and reinforce earlier mistakes.

Audio is only an experimental filter. Quietness can suggest a pause, but speech,
footsteps, glass, music, and crowd noise can confuse it. We do not call audio
peaks “ball hits” without a trained detector.

## What works today

- Qwen 0.8B and 2B run locally through MLX on the target Mac.
- Frame count and video-window duration are configurable.
- Prompts explain the fixed camera: large/lower players are the near side and
  small/upper players are the far side.
- Extended thinking is disabled and responses use short, validated JSON.
- Frames have timestamps and are kept in sequence.
- OpenCV, audio, panels, rolling state, scoring validation, and evaluation
  utilities have been implemented.
- A browser tool is ready for creating human rally labels.
- The VLM/scoring test suite currently has 46 passing tests.

## What is not solved

- We have not measured rally-boundary recall or winner accuracy against enough
  independent human labels.
- A small VLM cannot reliably see the tiny ball, exact contact, bounce, wall,
  foul, or close line call in every sampled frame.
- The audio threshold is not trained or calibrated.
- The visible scoreboard changes by game, not by point, so it cannot provide
  point-by-point ground truth.
- The current output is suitable for experiments and assisted review, not an
  unattended official score.

## What you can do now

Run a quick local benchmark from the repository root:

```bash
backend/.venv/bin/python -m vlm_coach.benchmark \
  --provider mlx \
  --model qwen3.5:0.8b \
  --video data/test_footage/padel_test.mp4 \
  --start 0 \
  --duration 10 \
  --frames 10
```

Use the VLM coach web app to produce review notes and a match story:

```bash
bash run_vlm_coach.sh
```

Treat all suggested winners and scores as items to confirm, not facts.

## The next useful step

The bottleneck is no longer another prompt tweak. It is ground truth. Label at
least 20–50 real rallies, including the exact end time and near/far winner:

```bash
open data/labels/rally_review/index.html
```

In the labeler, use `S` for rally start, `E` for rally end, and `A` to add it.
Labels autosave in the browser; export them to
`data/labels/rally_review/labels.json`. Then compare the candidate gate and VLM
predictions:

```bash
backend/.venv/bin/python -m vlm_coach.evaluate_rally_labels \
  --labels data/labels/rally_review/labels.json \
  --predictions data/experiments/rolling_0_8b_10_130.json \
  --candidates data/experiments/hybrid_candidates_full_match.json \
  --tolerance 2
```

That report tells us whether misses came from OpenCV/audio or from Qwen. Only
after measuring those stages should we decide whether to improve the prompt,
train an audio event detector, change frame sampling, or restore more ball
tracking.
