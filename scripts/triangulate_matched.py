#!/usr/bin/env python
"""Colour+motion ball candidates + cross-camera epipolar matching -> 3D.

#1 (colour+motion) gives a few ball-coloured MOVING candidates per camera
(excludes static court lines and non-ball movers). #2 (epipolar/geometry):
among all candidate pairs (one per camera), the REAL ball is the pair whose
two sightlines actually intersect (low reprojection error) AND lands on-court
at a plausible height. Heads get excluded by colour; remaining noise gets
excluded by cross-camera geometry. No single camera has to be right alone.

Example:
    python scripts/triangulate_matched.py --start 2 --window 40 --rate 25 --out /tmp/tri_matched.json
"""
import sys, os, argparse, json, math
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "backend", "src"))
import cv2, numpy as np
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
from ball_motion_color import ball_candidates


def _is_on_court(x, y, z):
    return (-1 <= x <= 11) and (-1 <= y <= 21) and (-0.3 <= z <= 8)


def _point_distance(a, b):
    return math.dist((a["x"], a["y"], a["z"]), (b["x"], b["y"], b["z"]))


def _candidate_xy(candidate):
    if isinstance(candidate, dict):
        return candidate["x"], candidate["y"]
    return candidate[0], candidate[1]


def _candidate_r(candidate):
    if isinstance(candidate, dict):
        return candidate.get("r", 4.0)
    return candidate[2] if len(candidate) > 2 else 4.0


def filter_candidates_near_boxes(candidates, boxes, x_margin_frac=0.45, y_margin_frac=0.12):
    """Remove candidates inside or beside player boxes where rackets/bodies dominate."""
    if not boxes:
        return list(candidates)
    kept = []
    for c in candidates:
        x, y = _candidate_xy(c)
        reject = False
        for x1, y1, x2, y2 in boxes:
            w = max(float(x2) - float(x1), 1.0)
            h = max(float(y2) - float(y1), 1.0)
            ex1 = float(x1) - x_margin_frac * w
            ex2 = float(x2) + x_margin_frac * w
            ey1 = float(y1) - y_margin_frac * h
            ey2 = float(y2) + y_margin_frac * h
            if ex1 <= x <= ex2 and ey1 <= y <= ey2:
                reject = True
                break
        if not reject:
            kept.append(c)
    return kept


def filter_player_boxes(boxes, image_shape, max_height_frac=0.45, min_height_frac=0.06,
                        max_width_frac=0.18):
    """Keep plausible on-court player boxes, dropping mural/poster detections."""
    h, w = image_shape[:2]
    kept = []
    for box in boxes:
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        bw = max(x2 - x1, 1.0)
        bh = max(y2 - y1, 1.0)
        if bh > h * max_height_frac:
            continue
        if bh < h * min_height_frac:
            continue
        if bw > w * max_width_frac:
            continue
        if y2 < h * 0.45:
            continue
        kept.append([x1, y1, x2, y2])
    return kept


def choose_best_match(candidates_a, candidates_b, Pa, Pb, max_reproj,
                      previous_point=None, continuity_weight=1.0):
    """Pick the cross-camera candidate pair using geometry plus continuity.

    Reprojection error remains the primary quality gate. When several pairs are
    geometrically plausible, temporal continuity biases the choice toward the
    previous 3D ball location instead of allowing frame-to-frame teleports.
    """
    from cv.triangulation import triangulate, reprojection_errors

    best = None
    for ca in candidates_a:
        xa, ya = _candidate_xy(ca)
        for cb in candidates_b:
            xb, yb = _candidate_xy(cb)
            obs = [(Pa, (xa, ya)), (Pb, (xb, yb))]
            X = triangulate(obs)
            if X is None:
                continue
            err = max(reprojection_errors(X, obs))
            x, y, z = float(X[0]), float(X[1]), float(X[2])
            if err > max_reproj or not _is_on_court(x, y, z):
                continue

            continuity = 0.0
            if previous_point is not None:
                continuity = math.dist(
                    (x, y, z),
                    (previous_point["x"], previous_point["y"], previous_point["z"]),
                )
            score = err + continuity_weight * continuity
            item = {
                "x": round(x, 2),
                "y": round(y, 2),
                "z": round(z, 2),
                "reproj_px": round(err, 1),
                "on_court": True,
                "match_score": round(score, 3),
                "pixel_a": [round(float(xa), 1), round(float(ya), 1)],
                "pixel_b": [round(float(xb), 1), round(float(yb), 1)],
            }
            if best is None or score < best["match_score"]:
                best = item
    return best


def draw_debug_frame(frame, candidates, selected_pixel=None, label=""):
    out = frame.copy()
    for c in candidates:
        x, y = _candidate_xy(c)
        r = _candidate_r(c)
        cv2.circle(out, (int(x), int(y)), max(int(r) + 8, 12), (0, 255, 255), 2)
    if selected_pixel:
        sx, sy = selected_pixel
        cv2.circle(out, (int(sx), int(sy)), 18, (0, 0, 255), 3)
        cv2.putText(out, "selected", (int(sx) + 20, int(sy) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.putText(out, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    return out


def draw_boxes(frame, boxes):
    out = frame.copy()
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    return out


def write_debug_pair(debug_dir, idx, t, frame_a, frame_b, candidates_a, candidates_b, match,
                     boxes_a=(), boxes_b=()):
    os.makedirs(debug_dir, exist_ok=True)
    selected_a = match.get("pixel_a") if match else None
    selected_b = match.get("pixel_b") if match else None
    a = draw_debug_frame(draw_boxes(frame_a, boxes_a), candidates_a, selected_a,
                         f"panasonic t={t:.2f}s candidates={len(candidates_a)}")
    b = draw_debug_frame(draw_boxes(frame_b, boxes_b), candidates_b, selected_b,
                         f"gopro t={t:.2f}s candidates={len(candidates_b)}")
    h = min(a.shape[0], b.shape[0])
    aw = int(a.shape[1] * h / a.shape[0])
    bw = int(b.shape[1] * h / b.shape[0])
    a = cv2.resize(a, (aw, h))
    b = cv2.resize(b, (bw, h))
    combo = np.hstack([a, b])
    scale = min(1.0, 1800 / combo.shape[1])
    if scale < 1.0:
        combo = cv2.resize(combo, (int(combo.shape[1] * scale), int(combo.shape[0] * scale)))
    cv2.imwrite(os.path.join(debug_dir, f"match_{idx:04d}_t{t:.2f}.jpg"), combo)


def clean_3d_track(points, max_speed_mps=45.0, interpolate_rate=25.0,
                   max_interpolate_gap=0.16):
    """Reject physically implausible jumps and linearly fill short gaps."""
    accepted = []
    rejected = []
    velocity = None

    for p in sorted((q for q in points if q.get("on_court", True)), key=lambda q: q["t"]):
        current = dict(p)
        if not accepted:
            current["track_state"] = "accepted"
            accepted.append(current)
            continue

        last = accepted[-1]
        dt = current["t"] - last["t"]
        if dt <= 0:
            continue

        predicted = last
        if velocity is not None:
            predicted = {
                "x": last["x"] + velocity[0] * dt,
                "y": last["y"] + velocity[1] * dt,
                "z": last["z"] + velocity[2] * dt,
            }

        speed = _point_distance(current, last) / dt
        pred_error = math.dist(
            (current["x"], current["y"], current["z"]),
            (predicted["x"], predicted["y"], predicted["z"]),
        )
        allowed_error = max(1.0, max_speed_mps * dt)
        if speed > max_speed_mps and pred_error > allowed_error:
            current["track_state"] = "rejected_teleport"
            current["on_court"] = False
            rejected.append(current)
            continue

        prev = accepted[-1]
        dt_prev = current["t"] - prev["t"]
        if dt_prev > 0:
            velocity = (
                (current["x"] - prev["x"]) / dt_prev,
                (current["y"] - prev["y"]) / dt_prev,
                (current["z"] - prev["z"]) / dt_prev,
            )
        current["track_state"] = "accepted"
        accepted.append(current)

    if not accepted:
        return []

    step = 1.0 / interpolate_rate if interpolate_rate and interpolate_rate > 0 else None
    clean = [accepted[0]]
    if step is None:
        clean.extend(accepted[1:])
        return clean

    for prev, cur in zip(accepted, accepted[1:]):
        gap = cur["t"] - prev["t"]
        if step < gap <= max_interpolate_gap:
            n = int(round(gap / step))
            for i in range(1, n):
                t = round(prev["t"] + i * step, 3)
                if t >= cur["t"]:
                    continue
                frac = (t - prev["t"]) / gap
                clean.append({
                    "t": t,
                    "x": round(prev["x"] + (cur["x"] - prev["x"]) * frac, 2),
                    "y": round(prev["y"] + (cur["y"] - prev["y"]) * frac, 2),
                    "z": round(prev["z"] + (cur["z"] - prev["z"]) * frac, 2),
                    "reproj_px": round(max(prev.get("reproj_px", 0.0), cur.get("reproj_px", 0.0)), 1),
                    "on_court": True,
                    "track_state": "interpolated",
                })
        clean.append(cur)
    return clean


def build_cam(video, calib):
    from cv.camera_model import CameraModel
    d = json.load(open(calib)); cam = CameraModel()
    cam.calibrate(d["camera_model_keypoints"], image_width=d.get("image_width", 1920),
                  image_height=d.get("image_height", 1080))
    return cam


def make_player_detector():
    from cv.detectors.yolo import UnifiedYoloDetector, YoloPlayerDetector
    return YoloPlayerDetector(UnifiedYoloDetector())


def read3(cap, fno):
    out = []
    for f in (fno - 1, fno, fno + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(f, 0)); ok, im = cap.read()
        out.append(im if ok else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-a", default="data/datasets/padelvic/cameras/panasonic_final.mp4")
    ap.add_argument("--calib-a", default="/tmp/panasonic_cammodel2.json")
    ap.add_argument("--video-b", default="data/datasets/padelvic/cameras/gopro.mp4")
    ap.add_argument("--calib-b", default="/tmp/gopro_cammodel.json")
    ap.add_argument("--sync", default="/tmp/sync_pana_gopro.json")
    ap.add_argument("--start", type=float, default=2.0)
    ap.add_argument("--window", type=float, default=40.0)
    ap.add_argument("--rate", type=float, default=25.0)
    ap.add_argument("--max-reproj", type=float, default=15.0)
    ap.add_argument("--continuity-weight", type=float, default=1.0)
    ap.add_argument("--max-speed-mps", type=float, default=45.0)
    ap.add_argument("--interpolate-rate", type=float, default=25.0)
    ap.add_argument("--max-interpolate-gap", type=float, default=0.16)
    ap.add_argument("--hsv-lo", default="22,50,110")
    ap.add_argument("--hsv-hi", default="48,255,255")
    ap.add_argument("--min-circularity", type=float, default=0.35)
    ap.add_argument("--min-fill-ratio", type=float, default=0.30)
    ap.add_argument("--max-aspect-ratio", type=float, default=2.2)
    ap.add_argument("--min-radius", type=float, default=2.5)
    ap.add_argument("--max-radius", type=float, default=18.0)
    ap.add_argument("--debug-dir", help="write side-by-side candidate/selection frames")
    ap.add_argument("--debug-every", type=int, default=10)
    ap.add_argument("--debug-limit", type=int, default=30)
    ap.add_argument("--exclude-players", action=argparse.BooleanOptionalAction, default=False,
                    help="reject candidates inside expanded YOLO person boxes")
    ap.add_argument("--player-x-margin-frac", type=float, default=0.45)
    ap.add_argument("--player-y-margin-frac", type=float, default=0.12)
    ap.add_argument("--out", default="/tmp/tri_matched.json")
    args = ap.parse_args()
    lo = [int(v) for v in args.hsv_lo.split(",")]; hi = [int(v) for v in args.hsv_hi.split(",")]

    os.chdir(os.path.join(_ROOT, "backend"))
    from cv.triangulation import triangulate, reprojection_errors
    va = os.path.join(_ROOT, args.video_a); vb = os.path.join(_ROOT, args.video_b)
    cam_a, cam_b = build_cam(va, args.calib_a), build_cam(vb, args.calib_b)
    Pa, Pb = cam_a.projection_matrix(), cam_b.projection_matrix()
    offset = json.load(open(args.sync)).get("offset_seconds", 0.0) if args.sync else 0.0
    capa, capb = cv2.VideoCapture(va), cv2.VideoCapture(vb)
    fpsa = capa.get(cv2.CAP_PROP_FPS) or 50.0; fpsb = capb.get(cv2.CAP_PROP_FPS) or 60.0
    player_a = make_player_detector() if args.exclude_players else None
    player_b = make_player_detector() if args.exclude_players else None

    n = int(args.window * args.rate)
    raw_pts = []; matched = 0; debug_written = 0
    for i in range(n):
        t = args.start + i / args.rate
        pa = read3(capa, int(t * fpsa)); pb = read3(capb, int((t - offset) * fpsb))
        if pa[1] is None or pb[1] is None:
            break
        ca = ball_candidates(pa[0], pa[1], pa[2], lo, hi, 2.0, 400.0,
                             min_circularity=args.min_circularity,
                             min_fill_ratio=args.min_fill_ratio,
                             max_aspect_ratio=args.max_aspect_ratio,
                             min_radius=args.min_radius,
                             max_radius=args.max_radius,
                             return_metrics=True)
        cb = ball_candidates(pb[0], pb[1], pb[2], lo, hi, 2.0, 400.0,
                             min_circularity=args.min_circularity,
                             min_fill_ratio=args.min_fill_ratio,
                             max_aspect_ratio=args.max_aspect_ratio,
                             min_radius=args.min_radius,
                             max_radius=args.max_radius,
                             return_metrics=True)
        boxes_a = []
        boxes_b = []
        if args.exclude_players:
            det_a = player_a.detect(pa[1], i * 2)
            det_b = player_b.detect(pb[1], i * 2 + 1)
            boxes_a = filter_player_boxes(det_a[:, :4].tolist(), pa[1].shape) if len(det_a) else []
            boxes_b = filter_player_boxes(det_b[:, :4].tolist(), pb[1].shape) if len(det_b) else []
            ca = filter_candidates_near_boxes(
                ca, boxes_a, args.player_x_margin_frac, args.player_y_margin_frac
            )
            cb = filter_candidates_near_boxes(
                cb, boxes_b, args.player_x_margin_frac, args.player_y_margin_frac
            )
        if not ca or not cb:
            continue
        best = choose_best_match(ca, cb, Pa, Pb, args.max_reproj,
                                 previous_point=raw_pts[-1] if raw_pts else None,
                                 continuity_weight=args.continuity_weight)
        if (args.debug_dir and args.debug_every > 0 and i % args.debug_every == 0 and
                debug_written < args.debug_limit):
            write_debug_pair(args.debug_dir, i, t, pa[1], pb[1], ca, cb, best, boxes_a, boxes_b)
            debug_written += 1
        if best:
            matched += 1
            best["t"] = round(t, 3)
            raw_pts.append(best)
        if i % 100 == 0:
            print(f"\r  t={t:.1f}s matched={matched}", end="", file=sys.stderr, flush=True)
    capa.release(); capb.release(); print(file=sys.stderr)

    pts = clean_3d_track(raw_pts, max_speed_mps=args.max_speed_mps,
                         interpolate_rate=args.interpolate_rate,
                         max_interpolate_gap=args.max_interpolate_gap)
    zs = [p["z"] for p in pts]
    print("\n=== colour+motion candidates + epipolar match -> 3D ===")
    print(f"  samples {n} @ {args.rate}Hz")
    print(f"  raw matched ball (both cams agree, on-court): {matched}")
    print(f"  clean track points: {len(pts)}")
    if zs:
        print(f"  ball height z: min {min(zs):.2f} median {np.median(zs):.2f} max {max(zs):.2f} m")
        spread = (max(p['x'] for p in pts)-min(p['x'] for p in pts),
                  max(p['y'] for p in pts)-min(p['y'] for p in pts))
        print(f"  x,y spread: {spread[0]:.1f} x {spread[1]:.1f} m  (wide = moving ball, tiny = stuck on a head)")
    json.dump({"cam_a": "panasonic", "cam_b": "gopro", "raw_points": raw_pts, "points": pts}, open(args.out, "w"))
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
