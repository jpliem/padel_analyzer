#!/usr/bin/env python3
"""Add weak ball suggestions without changing human-review label states."""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend", "src"))

import cv2


def _center(bbox):
    if bbox is None:
        return None
    return [round((bbox[0] + bbox[2]) / 2.0, 2),
            round((bbox[1] + bbox[3]) / 2.0, 2)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("labels")
    parser.add_argument("--detector", choices=("fast", "tracknet"), default="fast")
    parser.add_argument("--tracknet-model", default="models/tracknet_padel.pt")
    parser.add_argument("--output")
    args = parser.parse_args()
    with open(args.labels, encoding="utf-8") as handle:
        doc = json.load(handle)
    video = doc["video"]
    video = video if os.path.isabs(video) else os.path.join(ROOT, video)
    if args.detector == "fast":
        from cv.detectors.fast_ball import FastBallDetector
        detector = FastBallDetector(yolo_fallback=None)
        nominal_confidence = 0.25
    else:
        from cv.detectors.tracknet import TrackNetBallDetector
        model = args.tracknet_model
        model = model if os.path.isabs(model) else os.path.join(ROOT, "backend", model)
        detector = TrackNetBallDetector(model_path=model, yolo_fallback=None)
        nominal_confidence = 0.50

    by_frame = {int(item["frame"]): item for item in doc["labels"]}
    cap = cv2.VideoCapture(video)
    suggested = 0
    if args.detector == "tracknet":
        # Each sparse label needs its own real consecutive temporal window.
        # Running the network on every intervening 4K frame is unnecessary.
        from eval_ball_labels import tracknet_prediction

        for index, (frame_number, item) in enumerate(sorted(by_frame.items()), 1):
            prediction = tracknet_prediction(cap, detector, frame_number)
            if prediction is None:
                item.pop("suggestion", None)
            else:
                item["suggestion"] = {
                    "center": list(prediction), "source": args.detector,
                    "confidence": nominal_confidence,
                    "reviewed": False,
                }
                suggested += 1
            if index % 25 == 0:
                print(f"  suggested {index}/{len(by_frame)}", flush=True)
    else:
        last_frame = max(by_frame)
        for frame_number in range(last_frame + 1):
            ok, frame = cap.read()
            if not ok:
                break
            prediction = _center(detector.detect(frame, frame_number))
            item = by_frame.get(frame_number)
            if item is not None:
                if prediction is None:
                    item.pop("suggestion", None)
                else:
                    item["suggestion"] = {
                        "center": prediction, "source": args.detector,
                        "confidence": nominal_confidence,
                        "reviewed": False,
                    }
                    suggested += 1
    cap.release()
    output = args.output or args.labels
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(doc, handle, indent=2)
    # Keep the self-contained browser labeler in sync with the suggestions.
    from prepare_ball_label_set import HTML
    labeler_path = os.path.join(os.path.dirname(os.path.abspath(output)), "index.html")
    with open(labeler_path, "w", encoding="utf-8") as handle:
        handle.write(HTML.replace("LABELS_JSON", json.dumps(doc)))
    print(f"added {suggested}/{len(by_frame)} weak suggestions; label states unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
