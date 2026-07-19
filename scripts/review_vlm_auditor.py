#!/usr/bin/env python
"""Generate a human-review HTML page for eval_vlm_auditor results.

The eval scores VLM auditors against ball labels, but the labels themselves are
imperfect (inter-annotator ball IoU ~0.68 under motion blur). This page shows
every judged frame with the marker, the expected answer, and the model verdict,
so a human can overrule either side. Click a verdict per case, then Export to
download overrides JSON for re-scoring.

Usage:
    python scripts/review_vlm_auditor.py --report /tmp/vlm_auditor_eval.json
    open /tmp/vlm_auditor_frames/review.html
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

PAGE_STYLE = """
body { font-family: -apple-system, sans-serif; margin: 1.5rem; background: #14161a; color: #e8e8e8; }
h1 { font-size: 1.2rem; } h2 { font-size: 1rem; margin-top: 2rem; color: #9ecbff; }
.case { border: 1px solid #333; border-radius: 8px; padding: 0.8rem; margin: 0.8rem 0; background: #1c1f24; }
.case img { max-width: 100%; border-radius: 4px; }
.meta { display: flex; gap: 1.2rem; flex-wrap: wrap; margin: 0.5rem 0; font-size: 0.9rem; }
.tag { padding: 0.15rem 0.5rem; border-radius: 4px; background: #2a2e35; }
.expected-yes { color: #7ee787; } .expected-no { color: #ff7b72; }
.verdict-match { outline: 1px solid #2ea043; } .verdict-clash { outline: 1px solid #f85149; }
.choices button { margin-right: 0.5rem; padding: 0.3rem 0.7rem; border-radius: 5px; border: 1px solid #444;
  background: #2a2e35; color: #e8e8e8; cursor: pointer; }
.choices button.picked { background: #1f6feb; border-color: #1f6feb; }
#export { position: fixed; top: 1rem; right: 1rem; padding: 0.5rem 1rem; background: #238636;
  color: white; border: none; border-radius: 6px; cursor: pointer; }
"""

PAGE_SCRIPT = """
const picks = {};
function pick(btn, key, value) {
  picks[key] = value;
  btn.parentElement.querySelectorAll('button').forEach(b => b.classList.remove('picked'));
  btn.classList.add('picked');
}
function exportPicks() {
  const blob = new Blob([JSON.stringify(picks, null, 2)], {type: 'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'vlm_auditor_overrides.json';
  a.click();
}
"""


def render_case(model: str, case: dict, frames_dir: Path) -> str:
    verdict = case.get("verdict") or {}
    answer = verdict.get("marker_on_ball", "ERROR")
    expected = "yes/close" if case["condition"] == "correct" else "no"
    matched = (
        (case["condition"] == "correct" and answer in ("yes", "close"))
        or (case["condition"] == "wrong" and answer == "no")
    )
    key = f"{model}|{case['frame']}|{case['condition']}"
    image_rel = Path(case["image"]).name
    hint = html.escape(verdict.get("ball_location_hint", ""))
    note = html.escape(verdict.get("note", ""))
    expected_class = "expected-yes" if case["condition"] == "correct" else "expected-no"
    return f"""
<div class="case {'verdict-match' if matched else 'verdict-clash'}">
  <img src="{image_rel}" loading="lazy" alt="frame {case['frame']}">
  <div class="meta">
    <span class="tag">frame {case['frame']}</span>
    <span class="tag">marker: {case['condition']} ({'on labeled ball' if case['condition'] == 'correct' else 'displaced on purpose'})</span>
    <span class="tag {expected_class}">label says: {expected}</span>
    <span class="tag">model says: <b>{html.escape(str(answer))}</b></span>
    {f'<span class="tag">hint: {hint}</span>' if hint else ''}
    {f'<span class="tag">note: {note}</span>' if note else ''}
  </div>
  <div class="choices">
    Who is right?
    <button onclick="pick(this, '{key}', 'label_right')">Label right</button>
    <button onclick="pick(this, '{key}', 'model_right')">Model right (label wrong)</button>
    <button onclick="pick(this, '{key}', 'cant_tell')">Can't tell</button>
  </div>
</div>"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Build human-review page for auditor eval")
    parser.add_argument("--report", required=True)
    parser.add_argument("--out", help="Output HTML (default: review.html beside the frames)")
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text())
    sections = []
    frames_dir = None
    for model, entry in report.get("models", {}).items():
        cases = [c for c in entry.get("cases", []) if c.get("verdict")]
        if not cases:
            continue
        frames_dir = Path(cases[0]["image"]).parent
        rows = "".join(render_case(model, case, frames_dir) for case in cases)
        summary = entry.get("summary", {})
        sections.append(
            f"<h2>{html.escape(model)} — correct-marker acc "
            f"{summary.get('correct_marker', {}).get('accuracy')}, wrong-marker catch "
            f"{summary.get('wrong_marker', {}).get('accuracy')}</h2>{rows}")

    if frames_dir is None:
        print("no judged cases in report")
        return 1

    out_path = Path(args.out) if args.out else frames_dir / "review.html"
    out_path.write_text(
        f"<!doctype html><meta charset='utf-8'><title>VLM auditor review</title>"
        f"<style>{PAGE_STYLE}</style>"
        f"<button id='export' onclick='exportPicks()'>Export overrides</button>"
        f"<h1>VLM auditor human review — green outline: model matched label; "
        f"red outline: disagreement (judge who is right)</h1>"
        f"{''.join(sections)}<script>{PAGE_SCRIPT}</script>")
    print(f"review page -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
