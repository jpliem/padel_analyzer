#!/usr/bin/env python3
"""Create a self-contained browser workflow for exact rally ground truth."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Padel rally ground truth</title>
<style>
body{font-family:system-ui;margin:0;background:#111827;color:#f3f4f6}main{max-width:1200px;margin:auto;padding:20px}
video{width:100%;max-height:68vh;background:#000}.bar,.fields{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:12px 0}
button,select,input{font:inherit;padding:8px 12px}.primary{background:#22c55e;border:0}.danger{background:#ef4444;border:0;color:white}
table{width:100%;border-collapse:collapse}td,th{padding:7px;border-bottom:1px solid #374151;text-align:left}.muted{color:#9ca3af}
.hints{padding:10px;border:1px solid #374151;background:#1f2937;margin:10px 0}
</style></head><body><main>
<h1>Padel rally ground truth</h1>
<p class="muted">Watch the real continuous match. Mark the first visible serve/action and the moment the point visibly finishes. Unknown is valid—never guess.</p>
<div class="hints"><button onclick="toggleHints()">Show/hide machine gap hints (second pass only)</button>
<span id="hintControls" hidden><button onclick="candidate(-1)">Previous hint</button><button onclick="candidate(1)">Next hint</button><span id="hintText"></span></span></div>
<video id="video" controls preload="metadata" src="__VIDEO__"></video>
<div class="bar"><b id="clock">00:00.00</b>
<button onclick="markStart()">Set start [S]</button><span id="start">—</span>
<button onclick="markEnd()">Set end [E]</button><span id="end">—</span>
<button onclick="seek(-3)">−3s [←]</button><button onclick="seek(3)">+3s [→]</button></div>
<div class="fields">
<label>Winner <select id="winner"><option value="unknown">unknown</option><option value="team_a">near / team A</option><option value="team_b">far / team B</option></select></label>
<label>Ending <select id="ending"><option>unknown</option><option>winner</option><option>forced_error</option><option>unforced_error</option><option>net</option><option>out</option><option>double_bounce</option></select></label>
<label>Certainty <select id="certainty"><option value="certain">certain</option><option value="uncertain">uncertain</option><option value="unusable">unusable/cut</option></select></label>
<input id="notes" size="42" placeholder="Optional evidence/note">
<button class="primary" onclick="addRally()">Add rally [A]</button>
<button onclick="downloadLabels()">Download labels.json</button>
</div>
<p id="status"></p><table><thead><tr><th>#</th><th>start</th><th>end</th><th>winner</th><th>certainty</th><th></th></tr></thead><tbody id="rows"></tbody></table>
<script>
const seed=__SEED__; const storageKey=`padel-rally-labels:${seed.source}`;let draft=null;
try{draft=JSON.parse(localStorage.getItem(storageKey)||'null')}catch(e){}
const video=document.getElementById('video'); let labels=draft?.labels||seed.labels||[]; let start=null,end=null,candidateIndex=-1;
const fmt=t=>{const m=Math.floor(t/60),s=t-m*60;return `${String(m).padStart(2,'0')}:${s.toFixed(2).padStart(5,'0')}`};
video.ontimeupdate=()=>document.getElementById('clock').textContent=fmt(video.currentTime);
video.onloadedmetadata=()=>{if(seed.excluded_ranges?.length)video.currentTime=seed.excluded_ranges[0][1]};
function markStart(){start=+video.currentTime.toFixed(3);document.getElementById('start').textContent=fmt(start)}
function markEnd(){end=+video.currentTime.toFixed(3);document.getElementById('end').textContent=fmt(end)}
function seek(n){video.currentTime=Math.max(0,video.currentTime+n)}
function toggleHints(){hintControls.hidden=!hintControls.hidden;if(!hintControls.hidden&&candidateIndex<0)candidate(1)}
function candidate(delta){const hints=seed.candidate_suggestions||[];if(!hints.length){hintText.textContent=' no hints loaded';return}
 candidateIndex=(candidateIndex+delta+hints.length)%hints.length;const h=hints[candidateIndex];video.currentTime=Math.max(0,h.context_start);
 hintText.textContent=` hint ${candidateIndex+1}/${hints.length}: ${fmt(h.gap_start)}–${fmt(h.gap_end)} · audio gate ${h.audio_supported?'kept':'filtered'}`}
function addRally(){if(start===null||end===null||end<=start){alert('Set a valid start and end');return}
 labels.push({id:labels.length+1,start,end,winner:winner.value,ending:ending.value,certainty:certainty.value,notes:notes.value});
 start=null;end=null;notes.value='';document.getElementById('start').textContent='—';document.getElementById('end').textContent='—';saveDraft();render()}
function saveDraft(){localStorage.setItem(storageKey,JSON.stringify({labels,updated_at:new Date().toISOString()}))}
function removeRow(i){labels.splice(i,1);labels.forEach((x,j)=>x.id=j+1);saveDraft();render()}
function render(){rows.innerHTML=labels.map((x,i)=>`<tr><td>${x.id}</td><td><a href="#" onclick="video.currentTime=${x.start};return false">${fmt(x.start)}</a></td><td>${fmt(x.end)}</td><td>${x.winner}</td><td>${x.certainty}</td><td><button class="danger" onclick="removeRow(${i})">delete</button></td></tr>`).join('');status.textContent=`${labels.length} rallies labelled · draft autosaved in this browser`;}
function downloadLabels(){const payload={...seed,reviewed_at:new Date().toISOString(),labels};const blob=new Blob([JSON.stringify(payload,null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='labels.json';a.click();URL.revokeObjectURL(a.href)}
document.onkeydown=e=>{if(['INPUT','SELECT'].includes(e.target.tagName))return;if(e.key==='s')markStart();if(e.key==='e')markEnd();if(e.key==='a')addRally();if(e.key==='ArrowLeft')seek(-3);if(e.key==='ArrowRight')seek(3)};render();
</script></main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--suggestions", help="Optional hybrid candidate JSON")
    args = parser.parse_args()
    video = Path(args.video).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    labels_path = output / "labels.json"
    seed = {
        "schema_version": 1,
        "source": str(video),
        "camera": "fixed behind near baseline",
        "team_mapping": {"team_a": "near", "team_b": "far"},
        "excluded_ranges": [[0.0, 10.0]],
        "instructions": "Mark exact visible rally start/end; unknown winners are valid.",
        "labels": [],
    }
    if labels_path.exists():
        seed = json.loads(labels_path.read_text())
    else:
        labels_path.write_text(json.dumps(seed, indent=2) + "\n")
    if args.suggestions:
        candidate_payload = json.loads(Path(args.suggestions).read_text())
        seed["candidate_suggestions"] = [{
            "gap_start": item["gap"]["start"], "gap_end": item["gap"]["end"],
            "context_start": item["context"]["start"],
            "context_end": item["context"]["end"],
            "audio_supported": bool(item["audio_supports_boundary_review"]),
        } for item in candidate_payload.get("candidates", [])]
        labels_path.write_text(json.dumps(seed, indent=2) + "\n")
    relative_video = Path(os.path.relpath(video, output))
    html = HTML.replace("__VIDEO__", str(relative_video)).replace(
        "__SEED__", json.dumps(seed).replace("</", "<\\/")
    )
    (output / "index.html").write_text(html)
    print(f"label file: {labels_path}")
    print(f"labeler: {output / 'index.html'}")


if __name__ == "__main__":
    main()
