#!/usr/bin/env python
"""Report card — grade rally/point detection against PADELVIC real-match GT.

Ground truth: `derived/PadelVic_Panasonic_labeling.xlsx` sheet `Plays` lists 399
rallies as Start frame / End frame (panasonic_final.mp4, 50 fps).

Prediction: a pipeline run's results.json (from cli_analyze.py) contains an
`events` list. A predicted rally = from a SERVE event to the next POINT_END.

We only grade GT rallies that fall inside the processed frame window (so a
partial run isn't punished for rallies it never watched). Matching is by
temporal overlap (IoU) between predicted and GT intervals.

Example:
    python scripts/eval_rallies.py \
        --results /tmp/panasonic_rallies.json \
        --xlsx data/datasets/padelvic/derived/PadelVic_Panasonic_labeling.xlsx \
        --max-frame 3000
"""
import sys
import os
import argparse
import json

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IOU_MATCH_THRESHOLD = 0.3  # min temporal IoU to count a GT rally as detected


def load_gt_rallies(xlsx_path: str, max_frame: int | None):
    import pandas as pd
    df = pd.read_excel(xlsx_path, sheet_name="Plays")
    df = df.rename(columns=lambda c: str(c).strip())
    rallies = []
    for _, row in df.iterrows():
        s, e = row.get("Start frame"), row.get("End frame")
        if pd.isna(s) or pd.isna(e):
            continue
        s, e = int(s), int(e)
        if max_frame is not None and e > max_frame:
            continue
        rallies.append((s, e))
    return sorted(rallies)


def predicted_rallies(events: list, max_frame: int | None):
    """Pair SERVE → next POINT_END into [start, end] intervals."""
    evs = sorted(events, key=lambda e: e.get("frame_number", 0))
    out = []
    open_serve = None
    for e in evs:
        t = e.get("event_type")
        f = e.get("frame_number", 0)
        if max_frame is not None and f > max_frame:
            break
        if t == "SERVE":
            if open_serve is None:
                open_serve = f
        elif t == "POINT_END":
            if open_serve is not None:
                out.append((open_serve, f))
                open_serve = None
    return out


def interval_iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description="Grade rally/point detection vs PADELVIC GT.")
    ap.add_argument("--results", required=True, help="results.json from cli_analyze.py")
    ap.add_argument("--xlsx", required=True, help="PadelVic_Panasonic_labeling.xlsx")
    ap.add_argument("--max-frame", type=int, default=None,
                    help="Only grade GT rallies ending at/below this frame "
                    "(set to the run's processed frame count)")
    ap.add_argument("--out", help="Write detail JSON here")
    args = ap.parse_args()

    results = args.results if os.path.isabs(args.results) else os.path.join(_ROOT, args.results)
    xlsx = args.xlsx if os.path.isabs(args.xlsx) else os.path.join(_ROOT, args.xlsx)
    for p in (results, xlsx):
        if not os.path.exists(p):
            print(f"ERROR: not found: {p}", file=sys.stderr)
            return 1

    with open(results) as f:
        data = json.load(f)
    events = data.get("events", [])
    frames_processed = data.get("frames_processed")
    max_frame = args.max_frame or frames_processed

    gt = load_gt_rallies(xlsx, max_frame)
    pred = predicted_rallies(events, max_frame)

    # Event-type tally (diagnostic — what did the pipeline even emit?)
    tally = {}
    fault_detail = {}
    for e in events:
        tally[e.get("event_type")] = tally.get(e.get("event_type"), 0) + 1
        if e.get("event_type") == "FAULT":
            d = (e.get("metadata") or {}).get("detail", "?")
            fault_detail[d] = fault_detail.get(d, 0) + 1

    # Greedy match GT → best-IoU prediction.
    used = set()
    matched = []
    for gi, g in enumerate(gt):
        best_j, best_iou = None, 0.0
        for pj, p in enumerate(pred):
            if pj in used:
                continue
            iou = interval_iou(g, p)
            if iou > best_iou:
                best_iou, best_j = iou, pj
        if best_j is not None and best_iou >= IOU_MATCH_THRESHOLD:
            used.add(best_j)
            p = pred[best_j]
            matched.append({"gt": g, "pred": p, "iou": round(best_iou, 3),
                            "start_err": p[0] - g[0], "end_err": p[1] - g[1]})

    n_gt, n_pred, n_match = len(gt), len(pred), len(matched)
    recall = n_match / n_gt if n_gt else None
    precision = n_match / n_pred if n_pred else None
    ious = [m["iou"] for m in matched]

    print("\n=== rally/point detection report card ===")
    print(f"  window: frames 0–{max_frame}")
    print(f"  event tally emitted by pipeline: {tally or '(none)'}")
    if fault_detail:
        print(f"  fault reasons: {fault_detail}")
    print(f"  GT rallies in window:    {n_gt}")
    print(f"  predicted rallies:       {n_pred}")
    print(f"  matched (IoU>={IOU_MATCH_THRESHOLD}):    {n_match}")
    print(f"  recall (GT found):       {recall:.1%}" if recall is not None else "  recall: n/a")
    print(f"  precision (pred real):   {precision:.1%}" if precision is not None else "  precision: n/a (0 predicted)")
    if ious:
        print(f"  mean IoU (matched):      {sum(ious)/len(ious):.2f}")
        print(f"  mean start-frame error:  {sum(m['start_err'] for m in matched)/n_match:+.0f}")
        print(f"  mean end-frame error:    {sum(m['end_err'] for m in matched)/n_match:+.0f}")
    if n_gt and n_pred == 0:
        print("  -> pipeline emitted no SERVE→POINT_END pairs in this window.")

    if args.out:
        out = args.out if os.path.isabs(args.out) else os.path.join(_ROOT, args.out)
        with open(out, "w") as f:
            json.dump({"window_max_frame": max_frame, "tally": tally,
                       "gt_rallies": gt, "pred_rallies": pred,
                       "matched": matched,
                       "recall": recall, "precision": precision}, f)
        print(f"\n  detail -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
