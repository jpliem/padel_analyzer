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
#suggestion { position: absolute; width: 22px; height: 22px; border: 3px dashed #ffb000; border-radius: 50%; transform: translate(-50%, -50%); pointer-events: none; display: none; }
</style>
<div id="bar">
  <button onclick="prev()">Prev</button>
  <button onclick="next()">Next</button>
  <button onclick="setState('occluded')">Occluded</button>
  <button onclick="setState('outside_frame')">Outside frame</button>
  <button onclick="setState('hard_negative')">Hard negative</button>
  <button onclick="setState('uncertain')">Uncertain</button>
  <button onclick="acceptSuggestion()">Accept suggestion</button>
  <button onclick="download()">Download labels.json</button>
  <span id="status"></span>
</div>
<div id="wrap"><img id="frame"><div id="suggestion"></div><div id="dot"></div></div>
<script>
let doc = LABELS_JSON;
// Resume at the first unfinished item when reopening a partially reviewed set.
let i = Math.max(0, doc.labels.findIndex(label => label.state === 'unreviewed'));
const img = document.getElementById('frame');
const dot = document.getElementById('dot');
const suggestion = document.getElementById('suggestion');
const status = document.getElementById('status');
function show() {
  const l = doc.labels[i];
  img.src = l.image;
  const center = l.center;
  const proposed = l.suggestion && l.suggestion.center;
  dot.style.display = center ? 'block' : 'none';
  if (center) {
    const sx = img.clientWidth / img.naturalWidth;
    const scale = l.image_scale || 1;
    dot.style.left = (center[0] * scale * sx) + 'px';
    dot.style.top = (center[1] * scale * sx) + 'px';
  }
  suggestion.style.display = !center && proposed ? 'block' : 'none';
  if (!center && proposed) {
    const sx = img.clientWidth / img.naturalWidth;
    const scale = l.image_scale || 1;
    suggestion.style.left = (proposed[0] * scale * sx) + 'px';
    suggestion.style.top = (proposed[1] * scale * sx) + 'px';
  }
  status.textContent = `${i+1}/${doc.labels.length} frame=${l.frame} t=${l.t.toFixed(3)}s ${l.state}`;
}
img.onclick = ev => {
  const r = img.getBoundingClientRect();
  const sx = img.naturalWidth / r.width;
  const sy = img.naturalHeight / r.height;
  const scale = doc.labels[i].image_scale || 1;
  doc.labels[i].state = 'visible';
  doc.labels[i].center = [
    +(((ev.clientX - r.left) * sx) / scale).toFixed(2),
    +(((ev.clientY - r.top) * sy) / scale).toFixed(2)
  ];
  show();
};
function prev(){ i = Math.max(0, i-1); show(); }
function next(){ i = Math.min(doc.labels.length-1, i+1); show(); }
function setState(state){ doc.labels[i].state = state; doc.labels[i].center = null; show(); }
function acceptSuggestion(){
  const s = doc.labels[i].suggestion;
  if (!s || !s.center) return;
  doc.labels[i].state = 'visible';
  doc.labels[i].center = [...s.center];
  show();
}
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
  if (ev.key === 'o') setState('occluded');
  if (ev.key === 'x') setState('outside_frame');
  if (ev.key === 'n') setState('hard_negative');
  if (ev.key === 'u') setState('uncertain');
  if (ev.key === 's') acceptSuggestion();
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
    ap.add_argument(
        "--window",
        action="append",
        help="repeatable START_SECONDS:END_SECONDS rally window; overrides --start/--end",
    )
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
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if args.window:
        windows = []
        for value in args.window:
            try:
                start_text, end_text = value.split(":", 1)
                start_seconds, end_seconds = float(start_text), float(end_text)
            except ValueError as exc:
                raise SystemExit(f"invalid --window {value!r}; use START:END") from exc
            if start_seconds < 0 or end_seconds <= start_seconds:
                raise SystemExit(f"invalid --window {value!r}; END must be after START")
            windows.append((start_seconds, end_seconds))
    else:
        windows = [(args.start, args.end)]
    labels = []
    doc = {
        "schema_version": "1.0",
        "video": args.video,
        "fps": fps,
        "coordinate_space": "original_video_pixels",
        "preview_width": args.width,
        "original_video_width": original_width,
        "original_video_height": original_height,
        "labels": labels,
    }
    labels_path = os.path.join(out_dir, "labels.json")
    count = 0
    stop = False
    for start_seconds, end_seconds in windows:
        start_frame = int(round(start_seconds * fps))
        end_frame = int(round(end_seconds * fps))
        sequence_id = f"{os.path.basename(args.video)}:{start_frame}-{end_frame}"
        for frame_no in range(start_frame, end_frame + 1, args.step):
            if args.max_frames and count >= args.max_frames:
                stop = True
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
                "state": "unreviewed",
                "center": None,
                "event_tags": [],
                "sequence_id": sequence_id,
                "camera_id": os.path.basename(args.video),
            })
            count += 1
            # Large 4K videos can be interrupted; keep a valid resumable manifest
            # instead of losing every extracted label entry at process exit.
            if count % 25 == 0:
                with open(labels_path, "w", encoding="utf-8") as handle:
                    json.dump(doc, handle, indent=2)
        if stop:
            break
    cap.release()

    with open(labels_path, "w", encoding="utf-8") as handle:
        json.dump(doc, handle, indent=2)
    html = HTML.replace("LABELS_JSON", json.dumps(doc))
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as handle:
        handle.write(html)
    print(f"wrote {len(labels)} frames")
    print(f"  labels: {labels_path}")
    print(f"  labeler: {os.path.join(out_dir, 'index.html')}")


if __name__ == "__main__":
    main()
