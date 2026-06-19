#!/usr/bin/env python
"""Prepare a real-frame 2D ball labeling set from PADELVIC video.

Writes sampled frames, a starter `labels.json`, and a small browser labeler.
The browser labeler lets you click the ball center, mark absent/uncertain, and
download an updated JSON file for `scripts/eval_ball_labels.py`.

Example:
    python scripts/prepare_ball_label_set.py --start 2 --end 60 --step 10 \
        --out-dir data/labels/padelvic_panasonic_ball
"""
import argparse
import json
import os

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


HTML = """<!doctype html>
<meta charset="utf-8">
<title>Padel Ball Labeler</title>
<style>
body { margin: 0; font: 14px system-ui, sans-serif; background: #111; color: #eee; }
#bar { display: flex; gap: 12px; align-items: center; padding: 10px; background: #222; position: sticky; top: 0; }
button { padding: 8px 10px; }
#wrap { position: relative; display: inline-block; }
#frame { max-width: 100vw; display: block; }
#dot { position: absolute; width: 18px; height: 18px; border: 3px solid red; border-radius: 50%; transform: translate(-50%, -50%); pointer-events: none; display: none; }
</style>
<div id="bar">
  <button onclick="prev()">Prev</button>
  <button onclick="next()">Next</button>
  <button onclick="absent()">Absent</button>
  <button onclick="uncertain()">Uncertain</button>
  <button onclick="download()">Download labels.json</button>
  <span id="status"></span>
</div>
<div id="wrap"><img id="frame"><div id="dot"></div></div>
<script>
let doc = LABELS_JSON;
let i = 0;
const img = document.getElementById('frame');
const dot = document.getElementById('dot');
const status = document.getElementById('status');
function show() {
  const l = doc.labels[i];
  img.src = l.image;
  const b = l.ball || {};
  dot.style.display = b.visible && b.x != null ? 'block' : 'none';
  if (b.visible && b.x != null) {
    const sx = img.clientWidth / img.naturalWidth;
    const scale = l.image_scale || 1;
    dot.style.left = (b.x * scale * sx) + 'px';
    dot.style.top = (b.y * scale * sx) + 'px';
  }
  status.textContent = `${i+1}/${doc.labels.length} frame=${l.frame} t=${l.t.toFixed(3)}s ${b.visible ? 'visible' : 'absent'}${b.uncertain ? ' uncertain' : ''}`;
}
img.onclick = ev => {
  const r = img.getBoundingClientRect();
  const sx = img.naturalWidth / r.width;
  const sy = img.naturalHeight / r.height;
  const scale = doc.labels[i].image_scale || 1;
  doc.labels[i].ball = {
    visible: true,
    x: +(((ev.clientX - r.left) * sx) / scale).toFixed(2),
    y: +(((ev.clientY - r.top) * sy) / scale).toFixed(2),
    uncertain: false
  };
  show();
};
function prev(){ i = Math.max(0, i-1); show(); }
function next(){ i = Math.min(doc.labels.length-1, i+1); show(); }
function absent(){ doc.labels[i].ball = {visible: false}; show(); }
function uncertain(){ doc.labels[i].ball = {...(doc.labels[i].ball || {}), uncertain: true}; show(); }
function download(){
  const data = JSON.stringify(doc, null, 2);
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([data], {type: 'application/json'}));
  a.download = 'labels.json';
  a.click();
}
document.onkeydown = ev => {
  if (ev.key === 'ArrowRight' || ev.key === ' ') next();
  if (ev.key === 'ArrowLeft') prev();
  if (ev.key === 'a') absent();
  if (ev.key === 'u') uncertain();
};
show();
</script>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/datasets/padelvic/cameras/panasonic_final.mp4")
    ap.add_argument("--out-dir", default="data/labels/padelvic_panasonic_ball")
    ap.add_argument("--start", type=float, default=2.0)
    ap.add_argument("--end", type=float, default=60.0)
    ap.add_argument("--step", type=int, default=10, help="sample every N frames")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--width", type=int, default=1600)
    args = ap.parse_args()

    video = args.video if os.path.isabs(args.video) else os.path.join(ROOT, args.video)
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(ROOT, args.out_dir)
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 50.0
    start_frame = int(round(args.start * fps))
    end_frame = int(round(args.end * fps))
    labels = []
    count = 0
    for frame_no in range(start_frame, end_frame + 1, args.step):
        if args.max_frames and count >= args.max_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = cap.read()
        if not ok:
            continue
        scale = args.width / frame.shape[1]
        preview = cv2.resize(frame, (args.width, int(frame.shape[0] * scale)))
        name = f"frame_{frame_no:06d}.jpg"
        cv2.imwrite(os.path.join(frames_dir, name), preview)
        labels.append({
            "frame": frame_no,
            "t": frame_no / fps,
            "image": f"frames/{name}",
            "image_scale": scale,
            "ball": {"visible": False},
        })
        count += 1
    cap.release()

    doc = {
        "video": args.video,
        "fps": fps,
        "coordinate_space": "original_video_pixels",
        "preview_width": args.width,
        "labels": labels,
    }
    labels_path = os.path.join(out_dir, "labels.json")
    json.dump(doc, open(labels_path, "w"), indent=2)
    html = HTML.replace("LABELS_JSON", json.dumps(doc))
    open(os.path.join(out_dir, "index.html"), "w").write(html)
    print(f"wrote {len(labels)} frames")
    print(f"  labels: {labels_path}")
    print(f"  labeler: {os.path.join(out_dir, 'index.html')}")


if __name__ == "__main__":
    main()
