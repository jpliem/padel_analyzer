#!/usr/bin/env python3
"""Inventory independent PadelVic camera views and report evaluation readiness.

The views are intentionally treated independently; approximate PadelVic sync is
not used as frame-accurate multiview ground truth.
"""

import argparse
import json
import os

import cv2


CAMERAS = ("panasonic_final.mp4", "gopro.mp4", "samsung.mp4", "iphone.mp4")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-dir", default="data/datasets/padelvic/cameras")
    parser.add_argument("--out")
    args = parser.parse_args()
    rows = []
    for name in CAMERAS:
        path = os.path.join(args.camera_dir, name)
        row = {"camera": name, "path": path, "available": os.path.isfile(path)}
        if row["available"]:
            cap = cv2.VideoCapture(path)
            row.update({
                "readable": cap.isOpened(),
                "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            })
            cap.release()
        rows.append(row)
    payload = {
        "evaluation_mode": "independent_single_camera",
        "multiview_triangulation_claimed": False,
        "views": rows,
        "ready_views": sum(bool(r.get("available") and r.get("readable")) for r in rows),
    }
    print(json.dumps(payload, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    return 0 if payload["ready_views"] == len(CAMERAS) else 2


if __name__ == "__main__":
    raise SystemExit(main())
