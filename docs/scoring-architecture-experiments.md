# Scoring architecture experiments

> For the short, plain-English conclusion, see
> [current-vlm-scoring-status.md](current-vlm-scoring-status.md). This document
> remains the detailed experiment log and command reference.

The scoring experiment separates three jobs that should not be conflated:

1. propose a candidate rally boundary;
2. adjudicate whether a completed point and winner are visible;
3. update the score deterministically.

The VLM is never asked to calculate `15`, `30`, `40`, games, or sets. The
existing `PadelScoringEngine` should receive only confirmed point winners.

## Camera contract

Every VLM prompt states that the fixed camera is behind one baseline. Players
lower/larger in the image are `NEAR`; players upper/smaller are `FAR`. The team
assigned to each side is explicit input and must change when teams change ends.

## Modes

Run commands from the repository root. Start with `--limit 1` or `--limit 2` so
an experiment cannot unexpectedly analyze a full recording.

### Pure OpenCV

```bash
backend/.venv/bin/python -m vlm_coach.scoring_experiment \
  --mode opencv --video data/test_footage/padel_test.mp4 --limit 3
```

This measures cheap motion-based candidate boundaries. It deliberately emits no
winner because motion cannot safely award a point.

### Pure VLM

```bash
backend/.venv/bin/python -m vlm_coach.scoring_experiment \
  --mode vlm --video data/test_footage/padel_test.mp4 --limit 1 \
  --window 6 --stride 3 --frames 12
```

Qwen 2B classifies fixed overlapping windows and then adjudicates merged rally
candidates. No motion detector selects the windows.

### Hybrid: OpenCV then Qwen 2B

```bash
backend/.venv/bin/python -m vlm_coach.scoring_experiment \
  --mode hybrid --video data/test_footage/padel_test.mp4 --limit 1 --frames 12
```

OpenCV proposes a candidate; Qwen 2B decides whether a rally ended and whether a
winner is supported.

### Two VLMs: Qwen 0.8B then Qwen 2B

```bash
backend/.venv/bin/python -m vlm_coach.scoring_experiment \
  --mode multi --video data/test_footage/padel_test.mp4 --limit 2 \
  --window 6 --stride 3 --frames 10
```

Qwen 0.8B scouts fixed windows. Qwen 2B sees only merged positive candidates.

### Full cascade: OpenCV, Qwen 0.8B, Qwen 2B

```bash
backend/.venv/bin/python -m vlm_coach.scoring_experiment \
  --mode cascade --video data/test_footage/padel_test.mp4 --limit 2 --frames 10
```

OpenCV cheaply proposes motion windows, Qwen 0.8B rejects non-rallies, and Qwen
2B adjudicates retained candidates. This is the most likely production shape on
an 8 GB Mac if the 0.8B scout proves accurate enough.

### Targeted gap cascade

The newer cascade asks the small model only whether low-motion gaps are inside a
rally or between points. It merges uncertain gaps and sends completed merged
rallies to 2B:

```bash
backend/.venv/bin/python -m vlm_coach.gap_experiment \
  --video /path/to/continuous-match.mp4 \
  --model qwen3.5:0.8b --offset 0 --limit 0 \
  --judge-model qwen3.5:2b --judge-frames 12 \
  --output data/experiments/gap-cascade.json
```

`--judge-model` intentionally requires every gap to be processed. Missing gap
decisions would make merged rally boundaries meaningless.

## Decision criteria

Do not select a mode from prose quality. Label a small set of clips with true
start, end, point validity, and winner, then compare:

- rally boundary precision and recall;
- winner accuracy only when the system claims sufficient confidence;
- unknown/review rate;
- false automatic point awards (the most costly error);
- inference seconds per minute of video;
- images processed per confirmed point.

The preferred threshold should optimize false-award safety, not maximum
automation. A low-confidence or internally inconsistent verdict must enter the
review queue and must not update the authoritative score.

## Initial measured findings

These are engineering probes, not accuracy claims; the clips do not yet have a
reviewed rally/winner ground-truth file.

| Path | Probe | Result |
|---|---|---|
| OpenCV | 66 s highlight clip | 8 motion candidates in 0 VLM seconds; cannot identify winners |
| OpenCV | 60 s continuous clip | 4 candidates, including a 21.7 s active-play window |
| Pure 2B VLM | one 6 s window, 10 images | detected active rally; 22.5 s scout plus 14.5 s judge |
| 0.8B then 2B | one 6 s window, 10 images | 12.5 s scout plus 19.9 s judge; scout was internally inconsistent |
| OpenCV then 2B | one 21.2 s candidate, 12 endpoint-weighted images | 25.4 s; produced a consistent winner claim, still requiring ground-truth verification |

## Rolling temporal-input R&D (2026-07-18)

The storyboard extractor already sends file paths in chronological order. The
original continuity weakness was elsewhere: only `previous_phase` was carried
between calls, and ordering was not visible inside each JPEG. The rolling
experiment now burns `FRAME NN / TIME SS.ss` onto each image, uses overlapping
windows, and carries a bounded machine-state JSON object into the next call.
It deliberately does not carry the previous natural-language explanation.

Run it with:

```bash
backend/.venv/bin/python -m vlm_coach.rolling_experiment \
  --video data/test_footage/padel_test.mp4 \
  --model qwen3.5:0.8b --duration 30 \
  --window 12 --overlap 3 --frames 10 --limit 3
```

Measured results on the M2 8 GB Mac:

| Input | Model | Clip coverage | Result |
|---|---:|---:|---|
| ordered JPEG, 12 s window, 10 frames | 0.8B | 30 s / 3 windows | 25.8 s total; repeatedly called each window a new rally and produced contradictory endpoint phases |
| ordered JPEG, 12 s window, 10 frames | 2B | 30 s / 3 windows | 50.3 s total; more conservative, but incorrectly carried `same_rally` after an idle state |
| ordered JPEG, 6 s window, 10 frames | 0.8B | 18 s / 4 windows | 30.9 s total; detected endings, but copied nearly identical prose/evidence across three calls when previous prose was fed back |
| ordered JPEG, 6 s window, 10 frames, machine state only | 0.8B | 14 s / 3 windows | 25.3 s total; copying disappeared, but confidence remained unjustifiably 1.0 and transitions still conflicted |
| MLX native video, requested 1.5 FPS | 0.8B | one 6 s clip | 53.0 s; contradictory ending claim and citations outside the expected sampled-frame range |

The short edited highlight clip is not valid scoring ground truth, so these runs
measure speed and internal consistency rather than winner accuracy. They do prove:

- input order is not the primary failure;
- feeding model prose back causes anchoring/self-copying;
- 60 frames per minute is still only 1 FPS and can miss contact/bounce evidence;
- more visual input does not repair a weak temporal state machine;
- native Qwen3.5 video in MLX-VLM 0.6.5 works, but this path is currently much
  slower than multi-image input and did not produce safer structured evidence.

### Revised architecture

Python must own the rally state and reject impossible transitions. A VLM window
should answer one narrow question from current pixels; previous state can constrain
the allowed transition but cannot act as evidence. Use 6-second windows with about
10 images as the starting density, two seconds of overlap, and no prior prose.
OpenCV should watch full-rate motion and trigger denser sampling around suspected
serves/stops. Only after Python assembles a complete rally should the 2B model see
the ending and attempt a winner verdict.

The next accuracy experiment needs a continuous fixed-camera source with manually
labelled rally start, end, and winner timestamps. Compare 0.8B and 2B on the same
20-50 labelled points using boundary precision/recall, winner accuracy, false point
awards, seconds per video minute, and images processed per confirmed point.

### Continuous-match ground-truth workflow

The Explore Padel source begins with an edited intro/title sequence. The review
schema excludes 0-10 seconds and the browser starts at 10 seconds. OpenCV's 45
activity regions are intentionally not inserted as point labels: several regions
span 20-70 seconds and contain multiple possible points.

Open the review tool:

```bash
open data/labels/rally_review/index.html
```

Use `S` for rally start, `E` for visible ending, and `A` to add the reviewed
rally. Choose near/team A, far/team B, or unknown; unknown is preferable to a
guess. After 20-50 rallies, download `labels.json` and replace
`data/labels/rally_review/labels.json` with it.

The browser autosaves the draft locally. Machine gap hints are hidden by default;
complete the unbiased continuous pass first, then reveal hints for a second-pass
audit. All 43 OpenCV gaps are available as navigation hints, including the 27
filtered by audio, so reviewers can inspect filter mistakes without silently
treating retained candidates as truth.

Generate rolling predictions with timestamps retained for evidence scoring:

```bash
backend/.venv/bin/python -m vlm_coach.rolling_experiment \
  --video data/datasets/explore_padel/explore_padel_full_match_1080p.mp4 \
  --model qwen3.5:0.8b --start 10 --duration 120 \
  --window 6 --overlap 2 --frames 10 \
  --output data/experiments/rolling_0_8b_10_130.json
```

Evaluate claimed rally endings (overlapping-window duplicates are collapsed):

```bash
backend/.venv/bin/python -m vlm_coach.evaluate_rally_labels \
  --labels data/labels/rally_review/labels.json \
  --predictions data/experiments/rolling_0_8b_10_130.json \
  --candidates data/experiments/hybrid_candidates_full_match.json \
  --tolerance 2
```

The report separates three failure locations: rallies missed by the original
OpenCV candidate generator, rallies proposed by OpenCV but filtered by audio, and
rallies retained by the candidate gate but missed by the VLM. Prediction scope is
derived from the supplied rolling windows, so a two-minute benchmark is not
incorrectly penalized for unprocessed portions of the full match.

The first saved 0.8B continuous-source probe is
`data/experiments/rolling_0_8b_continuous_0_30.json`. It processed seven
overlapping windows in 51.36 seconds. It produced only one ending claim, near
15.6 seconds, even though visual inspection showed continued athletic play. The
model explanation itself said the rally continued. This motivated an evidence
invariant in `RollingObservation`: a point ending now requires cited active-play
frames followed by strictly later reset/stopped frames. Python rejects unordered
or missing transition evidence regardless of model confidence.

### Per-frame state R&D

The holistic rolling schema still allowed contradictory summaries and flags. A
narrower experiment (`vlm_coach.frame_state_experiment`) instead requires exactly
one independent `active_play | reset | unclear` label per supplied frame. Python
then requires at least two confident active frames followed by at least two reset
frames before proposing a boundary.

```bash
backend/.venv/bin/python -m vlm_coach.frame_state_experiment \
  --video data/datasets/explore_padel/explore_padel_full_match_1080p.mp4 \
  --model qwen3.5:0.8b --start 14 --duration 6 --frames 10
```

On 14-20 seconds, the 0.8B model returned all ten required entries as active play
in 15.95 seconds; this agreed with the inspected contact sheet and avoided the
holistic prompt's false ending. On 29-38 seconds it classified all twelve frames
as active in 16.55 seconds, also agreeing with visual inspection and showing that
the original OpenCV low-motion gap was not automatically a point boundary.

On 54-65 seconds it again labelled all fourteen frames active (18.95 seconds),
despite a contact sheet that appears to contain a low-motion/reset interval. This
case remains deliberately unscored because still images cannot prove whether the
players are waiting inside a live rally or between points. It is a priority human
review example. The result suggests per-frame classification improves output
consistency, but may have high active-play sensitivity and poor reset specificity.
Ground-truth review is required before choosing it as the 0.8B scout.

### Panel and audio fusion R&D

Two external results informed the next experiments:

- [Video Panels (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/papers/Doorenbos_Video_Panels_for_Long_Video_Understanding_CVPR_2026_paper.pdf)
  combines four consecutive frames into a 2x2 image, trading spatial resolution
  for denser temporal coverage and reporting gains across multiple VLMs.
- [Multi-Modal Hit Detection in Padel (CVPRW 2024)](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/papers/Decorte_Multi-Modal_Hit_Detection_and_Positional_Analysis_in_Padel_Competitions_CVPRW_2024_paper.pdf)
  reports 92% average hit-detection F1 with a trained audio SED network using
  40-bin log-Mel inputs. The paper explicitly uses audio hit times to narrow the
  frames sent to ball analysis.

Our panel implementation packs frames left-to-right then top-to-bottom while
retaining visible frame/time labels. On the same 0.8B classifications:

| Window | Separate images | 2x2 panels | Decision |
|---|---:|---:|---|
| 14-20 s, 10 frames | 15.95 s / 10 inputs | 12.59 s / 3 inputs | all active |
| 54-65 s, 14 frames | 18.95 s / 14 inputs | 13.02 s / 4 inputs | all active |

Panels preserved the decision while reducing measured inference by about 21% and
31%, respectively. They are now the preferred visual representation for boundary
triage on this Mac. This does not prove an accuracy gain; reviewed labels remain
the accuracy gate.

`vlm_coach.audio_probe` implements an intentionally untrained spectral-flux probe.
Its impulses include speech, footsteps, glass and music, so they must never be
called hits. Long quiet intervals are useful as a second candidate gate:

```bash
backend/.venv/bin/python -m vlm_coach.hybrid_candidate_probe \
  --video data/datasets/explore_padel/explore_padel_full_match_1080p.mp4 \
  --limit 0 --output data/experiments/hybrid_candidates_full_match.json
```

Across the continuous source, OpenCV proposed 43 gaps. Requiring a >=2-second
audio-quiet interval with >=1.5 seconds overlapping the visual gap retained 16
(37.2%) and filtered 27 (62.8%). Crucially, it rejected the visually inspected
false gap at 32.22-36.90 seconds and retained 56.70-62.82 seconds for review.
This can reduce panel/VLM gap calls by roughly 63%, but recall must be calculated
against reviewed rallies before it becomes a production gate.

The retained 56.70-62.82 candidate was sent to 0.8B as three panels. It returned
`unclear` in 11.18 seconds, which is the correct safe behavior in the absence of
human truth. The production routing implied by current evidence is therefore:

```text
OpenCV low-motion gap
  -> audio quietness agrees?
     no: keep rally open
     yes: panel-based 0.8B triage
       -> clear supported boundary: assemble candidate
       -> unclear: 2B or human review
```

Audio quietness and VLM output remain evidence gates only. Neither may update the
score without a completed-rally winner verdict that passes Python validation.

### Independent ground-truth options checked

The source video scoreboard was sampled at ten-second intervals. It shows team
names and games (for example, 0-0 changing to 0-1 around 110 seconds), but no
15/30/40 point score. It can later provide a game-total consistency check, not
individual rally boundaries or winners.

The public dataset accompanying the CVPRW padel-audio paper remains online. Its
Nextcloud share advertises an approximately 2.236 GB archive containing the
research release. The project did not automatically download that archive because
the server exposes it as one large bundle and it is not ground truth for our
specific Explore Padel recording. Its 2,377 annotated hits remain a promising
future route for training the small audio SED model after reviewing license,
contents, and storage expectations.

At this point, further threshold/prompt tuning on unlabelled footage would optimize
against anecdotes. The next scientific gate is the reviewed `labels.json`; until
then, the measured 62.8% call reduction is an efficiency result only, not an
accuracy claim.
| Audio impulses | 60 s continuous clip | 520 raw peaks; commentary/music/noise makes impulses unusable without further clustering |
| 0.8B gap classifier | two low-motion gaps | correctly avoided premature splitting on the first prompt, but later missed/contradicted reset labels; not authoritative |
| 2B gap classifier | one suspected continuation + one visible reset | called both `between_points` at 75% and 95%; human video review is required to score accuracy |

Prompt/schema iterations found four safety requirements:

1. All decision fields must be required. Optional defaults caused apparently
   valid empty/zero decisions.
2. A winner requires a completed rally, confidence >= 0.75, and supplied-frame
   citations. Invalid frame IDs invalidate automatic scoring.
3. `decisive_team` plus `decisive_outcome` must agree with `winner`; Python rejects
   contradictions such as "Team A made the error" plus `winner=team_a`.
4. Scoring storyboards need dense coverage of the final four seconds and a
   near-end frame. Uniform endpoint-exclusive sampling routinely missed resets.

The current evidence favors OpenCV for cheap proposals plus one targeted 2B call.
The 0.8B scout has not yet saved time reliably: on positive windows its inference
and model-switch cost is additive, while its classifications have contradicted
its explanations. It may still be useful after evaluation as a high-recall filter
for long inactive recordings.

Three compact review clips and a ground-truth template live under
`data/labels/scoring_gap_review/`. Fill `human_label` with `active_rally`,
`between_points`, or `unclear` before using these probes as accuracy evidence.

After review, evaluate a model without silently treating unreviewed clips as
truth:

```bash
backend/.venv/bin/python -m vlm_coach.evaluate_gap_labels \
  data/labels/scoring_gap_review/labels.json --model-key qwen_2b
```

The command exits without an accuracy value when no reviewed labels exist.
