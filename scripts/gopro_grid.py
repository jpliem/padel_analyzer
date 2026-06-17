"""Save a gopro frame with a pixel grid to read true court-corner coordinates."""
import cv2, numpy as np
ROOT = "/Users/jonathan/Documents/Github/padel_analyzer"
cap = cv2.VideoCapture(f"{ROOT}/data/datasets/padelvic/cameras/gopro.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 1500)
ok, f = cap.read(); cap.release()
H, W = f.shape[:2]
disp_w = 1400
s = disp_w / W
img = cv2.resize(f, (disp_w, int(H * s)))
# grid every 100 px in DISPLAY coords; label is the FULL-RES coord
for x in range(0, img.shape[1], 100):
    cv2.line(img, (x, 0), (x, img.shape[0]), (0, 255, 0), 1)
    cv2.putText(img, str(int(x / s)), (x + 2, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
for y in range(0, img.shape[0], 100):
    cv2.line(img, (0, y), (img.shape[1], y), (0, 255, 0), 1)
    cv2.putText(img, str(int(y / s)), (2, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
cv2.imwrite("/tmp/gopro_grid.png", img)
print(f"full-res {W}x{H}, grid labels are full-res px -> /tmp/gopro_grid.png")
