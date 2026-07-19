#!/usr/bin/env python3
"""Backend-working-directory launcher for the root VLM probe script."""

import os
import importlib.util

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOURCE = os.path.join(ROOT, "scripts", "vlm_ball_probe.py")
spec = importlib.util.spec_from_file_location("_root_vlm_ball_probe", SOURCE)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

extract_video_points = module.extract_video_points
summarize_against_gt = module.summarize_against_gt
load_gt = module.load_gt
load_label_gt = module.load_label_gt

if __name__ == "__main__":
    raise SystemExit(module.main())
