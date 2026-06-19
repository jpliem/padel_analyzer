#!/usr/bin/env python
"""Benchmark ready-to-use ball detector models on PADELVIC synthetic GT.

This is a thin runner around `eval_synthetic.py` that compares the local
ready weights in `backend/models/` using the same clip, frame limit, and output
schema.
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_eval(name, args, out_dir):
    out = os.path.join(out_dir, f"{name}.json")
    cmd = [
        sys.executable,
        os.path.join(ROOT, "scripts", "eval_synthetic.py"),
        "--clip", args.clip,
        "--csv", args.csv,
        "--max-frames", str(args.max_frames),
        "--out", out,
    ]
    if name == "yolo":
        cmd += ["--detector", "yolo"]
    elif name == "fast":
        cmd += ["--detector", "fast"]
    else:
        cmd += [
            "--detector", "tracknet",
            "--tracknet-model", f"models/{name}.pt",
            "--tracknet-conf", str(args.tracknet_conf),
            "--no-yolo-fallback",
        ]
    print(f"\n=== {name} ===")
    subprocess.run(cmd, check=True, cwd=ROOT)
    return json.load(open(out))["summary"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="data/datasets/padelvic/synthetic/001-1250-17462.mkv")
    ap.add_argument("--csv", default="data/datasets/padelvic/synthetic/001-1250-17462.csv")
    ap.add_argument("--max-frames", type=int, default=100)
    ap.add_argument("--tracknet-conf", type=float, default=0.3)
    ap.add_argument("--out-dir", default="/tmp/ball_model_benchmark")
    ap.add_argument("--models", default="yolo,fast,tracknet_padel,tracknet_tennis,tracknetv2")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    summaries = []
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        try:
            summaries.append(run_eval(name, args, args.out_dir))
        except Exception as exc:
            summaries.append({"detector": name, "error": str(exc)})

    summary_path = os.path.join(args.out_dir, "summary.json")
    json.dump(summaries, open(summary_path, "w"), indent=2)

    print("\n=== model benchmark summary ===")
    for row in summaries:
        name = row.get("tracknet_model") or row.get("detector")
        if row.get("error"):
            print(f"  {name}: ERROR {row['error']}")
            continue
        print(
            f"  {name}: detection={row.get('detection_rate')} "
            f"median={row.get('median_error_px')} pck50={row.get('pck', {}).get('<50px')}"
        )
    print(f"\n  summary -> {summary_path}")


if __name__ == "__main__":
    main()
