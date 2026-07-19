# PadelTracker100 research notes

Official record: [PadelTracker100 on Zenodo](https://zenodo.org/records/17020011)
(DOI `10.5281/zenodo.17020011`). The annotations are published under CC BY
4.0.

## What it contains

PadelTracker100 provides annotations for nearly 100,000 frames selected from
two professional World Padel Tour Barcelona Master Final matches from 2022.
The source footage is 1920x1080 at 30 FPS and uses one standard broadcast
camera view. The annotations include:

- ball bounding boxes in COCO-style JSON;
- player positions and 17-keypoint poses in COCO-style JSON;
- shot events in semicolon-delimited CSV files.

The official archive contains `labels.zip`, a small `scripts.zip`, and an
exploration notebook. It does **not** contain the match videos.

## Source-video blocker

The annotation filenames identify two original YouTube sources:

- `Mdq42o4jdg0`
- `tCuZ6i-aVbY`

Both videos currently report as private. Public searches for the corresponding
2022 Barcelona finals found highlights and other recordings, but not verified
frame-identical copies. A different upload cannot safely be paired with these
labels: broadcast edits, intros, frame rate conversion, or even a single
missing frame will shift every annotation.

Before importing PadelTracker100, obtain authorized copies of the exact source
videos from the dataset authors (`PadelTracker100@gmail.com`) and verify frame
alignment against the COCO `file_name`/frame IDs.

## How it fits this project

Once the exact videos are available:

1. Verify dimensions, frame rate, frame count, and several labelled frames by
   visual overlay or frame hashes.
2. Convert ball boxes to the project's ball-label format while retaining the
   original box and visibility metadata.
3. Import pose and shot-event labels as separate training/evaluation targets.
4. Split by rally or match, never by random neighboring frames, to avoid
   temporal leakage.
5. Report it as a **single-camera** benchmark. It supplies neither synchronized
   multi-camera geometry nor metric depth ground truth.

The downloaded Explore Padel match is therefore appropriate for detector
generalization tests and new manual labels, but it cannot substitute for either
PadelTracker100 source video.
