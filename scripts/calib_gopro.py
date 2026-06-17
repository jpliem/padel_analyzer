#!/usr/bin/env python
"""Build gopro calibration from 4 manually-read court corners, then verify.

Corners are the painted-court outer rectangle in FULL-RES pixels, given in the
order: near-left, near-right, far-right, far-left  (matching the court frame
near=y0, far=y20, left=x0). Builds a homography -> derives the 12 keypoints ->
CameraModel, writes /tmp/gopro_cammodel.json, and draws the overlay so we can
check the green lines land on the real court. Iterate the --corners until good.

Example:
    python scripts/calib_gopro.py --corners 330,1680 2180,1660 1730,600 710,600
"""
import sys, os, argparse, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend", "src"))
import cv2, numpy as np

COURT_W, COURT_L, NET_H = 10.0, 20.0, 0.92
SVC_NEAR, SVC_FAR, CTR = 6.95, 13.05, 5.0
LINES = [
    ((0,0,0),(COURT_W,0,0)), ((0,COURT_L,0),(COURT_W,COURT_L,0)),
    ((0,0,0),(0,COURT_L,0)), ((COURT_W,0,0),(COURT_W,COURT_L,0)),
    ((0,SVC_NEAR,0),(COURT_W,SVC_NEAR,0)), ((0,SVC_FAR,0),(COURT_W,SVC_FAR,0)),
    ((CTR,SVC_NEAR,0),(CTR,SVC_FAR,0)), ((0,10,0),(COURT_W,10,0)),
]


def main():
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--corners", nargs=4, required=True,
                    help="px 'x,y' for near-left near-right far-right far-left")
    ap.add_argument("--frame", type=int, default=1500)
    args = ap.parse_args()
    os.chdir(os.path.join(ROOT, "backend"))
    from cv.camera_model import CameraModel
    from cv.court_calibration import KEYPOINT_COURT_COORDS_12

    corners_px = np.array([[float(v) for v in c.split(",")] for c in args.corners], dtype=np.float32)
    corners_court = np.array([[0,0],[COURT_W,0],[COURT_W,COURT_L],[0,COURT_L]], dtype=np.float32)
    H, _ = cv2.findHomography(corners_px, corners_court)  # px -> court
    Hinv = np.linalg.inv(H)
    kp_m = np.array(KEYPOINT_COURT_COORDS_12[:12], dtype=np.float32).reshape(-1, 1, 2)
    kp_px = cv2.perspectiveTransform(kp_m, Hinv).reshape(-1, 2)

    video = f"{ROOT}/data/datasets/padelvic/cameras/gopro.mp4"
    cap = cv2.VideoCapture(video); cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read(); cap.release()
    w, h = frame.shape[1], frame.shape[0]

    cam = CameraModel()
    cam.calibrate(kp_px.tolist(), image_width=w, image_height=h)
    json.dump({"camera_model_keypoints": kp_px.tolist(), "image_width": w, "image_height": h},
              open("/tmp/gopro_cammodel.json", "w"))
    print("wrote /tmp/gopro_cammodel.json; PnP solved:", cam.rvec is not None)

    def proj(p):
        u, v = cam.court_to_pixel(*p)
        return int(u), int(v)
    for a, b in LINES:
        cv2.line(frame, proj(a), proj(b), (0, 255, 0), 3)
    for c, lab in [((0,0,0),"NL"),((COURT_W,0,0),"NR"),((COURT_W,COURT_L,0),"FR"),((0,COURT_L,0),"FL")]:
        cv2.circle(frame, proj(c), 12, (0,0,255), -1)
        cv2.putText(frame, lab, (proj(c)[0]+12, proj(c)[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
    cv2.imwrite("/tmp/calib_gopro.png", cv2.resize(frame, (1280, int(1280*h/w))))
    print("  -> /tmp/calib_gopro.png  (check green lines sit on the real court)")


if __name__ == "__main__":
    main()
