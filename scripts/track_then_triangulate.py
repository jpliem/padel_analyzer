#!/usr/bin/env python
"""Per-camera 2D ball tracking, THEN triangulate the dense tracks.

The bottleneck for multi-camera 3D was correspondence: the ball is rarely
detected in BOTH cameras the same instant. Fix: track the ball in each camera
separately with a pixel-space Kalman filter that COASTS through missed frames
(the ball flies a smooth path, it doesn't teleport). Each camera then has a
ball pixel almost every frame -> triangulating the two dense tracks yields a
continuous 3D trajectory instead of a few lonely points.

Pipeline per shared match-time sample:
  read both cams -> detect ball (head-filtered) -> feed each camera's pixel
  Kalman -> smoothed pixel per camera -> triangulate -> reproj-gate -> 3D.

Example:
    python scripts/track_then_triangulate.py --start 2 --window 40 --rate 25 \
        --out /tmp/tri_dense.json
"""
import sys, os, argparse, json
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "backend", "src"))
import cv2, numpy as np
from filterpy.kalman import KalmanFilter

MAX_PIX_JUMP = 400.0   # px between samples; faster = bad detection, reject
MAX_COAST = 6          # consecutive missed samples to coast before going lost


class PixelBallTracker:
    """Constant-velocity Kalman in image space; coasts through gaps."""
    def __init__(self, dt=1.0):
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], float)
        kf.H = np.array([[1,0,0,0],[0,1,0,0]], float)
        kf.P *= 500.; kf.R = np.eye(2)*4.; kf.Q = np.eye(4)*1.0
        self.kf = kf; self.init = False; self.miss = 0; self.prev = None

    def step(self, det):
        """det = (px,py) or None. Returns (px,py, is_coasting) or None if lost."""
        if not self.init:
            if det is None:
                return None
            self.kf.x = np.array([det[0], det[1], 0, 0], float)
            self.init = True; self.miss = 0; self.prev = det
            return (det[0], det[1], False)
        self.kf.predict()
        use = det is not None
        if use and self.prev is not None:
            if np.hypot(det[0]-self.prev[0], det[1]-self.prev[1]) > MAX_PIX_JUMP:
                use = False  # implausible jump -> treat as miss
        if use:
            self.kf.update(np.array(det, float)); self.miss = 0; self.prev = det
            return (float(self.kf.x[0]), float(self.kf.x[1]), False)
        self.miss += 1
        if self.miss > MAX_COAST:
            return None  # lost
        return (float(self.kf.x[0]), float(self.kf.x[1]), True)  # coasting


def make_detectors():
    from cv.detectors.yolo import UnifiedYoloDetector, YoloBallDetector, YoloPlayerDetector
    from cv.detectors.tracknet import TrackNetBallDetector
    u = UnifiedYoloDetector()
    ball = TrackNetBallDetector(model_path="models/tracknet_padel.pt", conf_threshold=0.3,
                                yolo_fallback=YoloBallDetector(u))
    return ball, YoloPlayerDetector(u)


def detect_ball(ball_det, players_det, frame, i):
    bbox = ball_det.detect(frame, i)
    if not bbox or len(bbox) < 4:
        return None
    cx, cy = (bbox[0]+bbox[2])/2.0, (bbox[1]+bbox[3])/2.0
    boxes = players_det.detect(frame, i)
    for x1, y1, x2, y2 in (boxes[:, :4] if len(boxes) else []):
        if x1 <= cx <= x2 and y1 <= cy <= y1 + 0.35*(y2-y1):
            return None  # ball candidate is in a player's head region -> reject
    return (cx, cy)


def build_cam(video, calib):
    from cv.camera_model import CameraModel
    d = json.load(open(calib)); cam = CameraModel()
    cam.calibrate(d["camera_model_keypoints"], image_width=d.get("image_width", 1920),
                  image_height=d.get("image_height", 1080))
    return cam


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
    ap.add_argument("--max-reproj", type=float, default=25.0)
    ap.add_argument("--out", default="/tmp/tri_dense.json")
    args = ap.parse_args()

    os.chdir(os.path.join(_ROOT, "backend"))
    from cv.triangulation import triangulate, reprojection_errors
    va = os.path.join(_ROOT, args.video_a); vb = os.path.join(_ROOT, args.video_b)
    cam_a, cam_b = build_cam(va, args.calib_a), build_cam(vb, args.calib_b)
    Pa, Pb = cam_a.projection_matrix(), cam_b.projection_matrix()
    offset = json.load(open(args.sync)).get("offset_seconds", 0.0) if args.sync else 0.0

    capa, capb = cv2.VideoCapture(va), cv2.VideoCapture(vb)
    fpsa = capa.get(cv2.CAP_PROP_FPS) or 50.0
    fpsb = capb.get(cv2.CAP_PROP_FPS) or 60.0
    ball_a, players_a = make_detectors()
    ball_b, players_b = make_detectors()
    trk_a, trk_b = PixelBallTracker(), PixelBallTracker()

    n = int(args.window * args.rate)
    det_a_n = det_b_n = both_track = kept = 0
    pts = []
    for i in range(n):
        t = args.start + i / args.rate
        capa.set(cv2.CAP_PROP_POS_FRAMES, int(t * fpsa))
        capb.set(cv2.CAP_PROP_POS_FRAMES, int((t - offset) * fpsb))
        oka, fa = capa.read(); okb, fb = capb.read()
        if not (oka and okb):
            break
        da = detect_ball(ball_a, players_a, fa, i)
        db = detect_ball(ball_b, players_b, fb, i)
        det_a_n += da is not None; det_b_n += db is not None
        sa = trk_a.step(da); sb = trk_b.step(db)
        if sa is None or sb is None:
            continue
        both_track += 1
        X = triangulate([(Pa, (sa[0], sa[1])), (Pb, (sb[0], sb[1]))])
        if X is None:
            continue
        err = max(reprojection_errors(X, [(Pa, (sa[0], sa[1])), (Pb, (sb[0], sb[1]))]))
        if err > args.max_reproj:
            continue
        x, y, z = float(X[0]), float(X[1]), float(X[2])
        onc = (-1 <= x <= 11) and (-1 <= y <= 21) and (-0.5 <= z <= 8)
        kept += 1
        pts.append({"t": round(t, 3), "x": round(x, 2), "y": round(y, 2), "z": round(z, 2),
                    "reproj_px": round(err, 1), "coast": bool(sa[2] or sb[2]), "on_court": onc})
        if i % 100 == 0:
            print(f"\r  t={t:.1f}s detA={det_a_n} detB={det_b_n} both-track={both_track} kept={kept}",
                  end="", file=sys.stderr, flush=True)
    capa.release(); capb.release()
    print(file=sys.stderr)

    onc = [p for p in pts if p["on_court"]]
    zs = [p["z"] for p in onc]
    print("\n=== per-camera-track -> dense triangulation ===")
    print(f"  samples {n} @ {args.rate}Hz")
    print(f"  ball detected: camA {det_a_n}  camB {det_b_n}  (per-camera, sparse)")
    print(f"  both tracks live (after coasting): {both_track}")
    print(f"  triangulated & reproj-gated: {kept}")
    print(f"  on-court: {len(onc)} ({100*len(onc)/max(len(pts),1):.0f}% of kept)")
    if zs:
        print(f"  ball height z: min {min(zs):.2f} median {np.median(zs):.2f} max {max(zs):.2f} m")
    json.dump({"cam_a": "panasonic", "cam_b": "gopro", "offset_s": offset, "points": pts},
              open(args.out, "w"))
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
