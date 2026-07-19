from __future__ import annotations

from pathlib import Path
import json
import time
from typing import Callable, Dict, List

from .ollama_client import OllamaClient
from .schemas import MatchStory, RallyAnalysis, model_dump
from .store import MatchStore
from .video import detect_segments, extract_storyboard, probe_video


RALLY_PROMPT = """You are reviewing one candidate segment from a fixed-camera padel match.
The images are chronological and labeled by their order in the request.
Team A is {team_a}; Team B is {team_b}.

Describe only what the images support. Focus on tactical shape, net control,
partner spacing, transitions, visible mistakes, and highlight value. A sparse
storyboard usually cannot prove ball bounces, glass/fence contacts, line calls,
or the point winner. Use unknown and explain uncertainty when evidence is
missing. Never invent a score. Evidence frame numbers refer to zero-based image
order. Keep every explanation to one short sentence. Return only JSON matching
the supplied schema.
"""


STORY_PROMPT = """You are a practical padel coach. Build an evidence-linked match review from
the rally analyses below. Team A is {team_a}; Team B is {team_b}.

Do not convert uncertain rally guesses into facts. Mention patterns only when
supported by multiple rally analyses. Training priorities must cite rally IDs.
Keep every field concise and actionable. Include a limitation that this report was
generated from sparse visual storyboards and is not official scoring.

RALLY ANALYSES:
{rallies}

Return only JSON matching the supplied schema.
"""


class MatchPipeline:
    def __init__(self, store: MatchStore, client: OllamaClient | None = None):
        self.store = store
        self.client = client or OllamaClient()

    def analyze(self, match_id: str, progress: Callable[[int, str], None] | None = None) -> Dict:
        progress = progress or (lambda percent, stage: None)
        record = self.store.load(match_id)
        video_path = self.store.video_path(match_id)
        progress(2, "Reading recording")
        media = probe_video(video_path)
        record["media"] = media
        self.store.update(match_id, media=media)

        progress(7, "Finding active play")
        segments = detect_segments(video_path)
        if not segments:
            raise ValueError("No analysable video segments were found")

        rallies: List[dict] = []
        inference_seconds = 0.0
        for index, segment in enumerate(segments, start=1):
            base = 10 + int((index - 1) / len(segments) * 76)
            progress(base, f"Understanding segment {index} of {len(segments)}")
            relative_dir = Path("storyboards") / f"rally_{index:03d}"
            frames = extract_storyboard(
                video_path, segment, self.store.directory(match_id) / relative_dir,
            )
            frame_timeline = ", ".join(
                f"frame {item['index']}={item['timestamp']:.1f}s" for item in frames
            )
            prompt = (
                RALLY_PROMPT.format(team_a=record["team_a"], team_b=record["team_b"])
                + f"\nStoryboard timeline: {frame_timeline}."
            )
            inference_started = time.perf_counter()
            analysis = self.client.structured(
                record["model"], prompt, RallyAnalysis,
                images=[item["path"] for item in frames],
            )
            rally_inference_seconds = time.perf_counter() - inference_started
            inference_seconds += rally_inference_seconds
            rally = {
                "id": index,
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "storyboard": [
                    {"index": item["index"], "timestamp": item["timestamp"],
                     "url": f"/media/{match_id}/storyboards/rally_{index:03d}/{item['path'].name}"}
                    for item in frames
                ],
                "analysis": model_dump(analysis),
                "inference_seconds": round(rally_inference_seconds, 3),
                "review": None,
            }
            rallies.append(rally)
            record["rallies"] = rallies
            self.store.update(match_id, rallies=rallies)

        progress(89, "Writing the match story")
        story_started = time.perf_counter()
        record["story"] = self.build_story(record)
        story_seconds = time.perf_counter() - story_started
        inference_seconds += story_seconds
        performance = {
            "model": record["model"],
            "video_seconds": media["duration"],
            "storyboards": len(rallies),
            "images_sent": sum(len(rally["storyboard"]) for rally in rallies),
            "vlm_seconds": round(inference_seconds, 3),
            "story_seconds": round(story_seconds, 3),
            "realtime_factor": round(inference_seconds / max(media["duration"], 0.001), 3),
        }
        record["rallies"] = rallies
        record["status"] = "complete"
        record["progress"] = 100
        record["stage"] = "Match review ready"
        record["error"] = None
        # Notify observers before the authoritative final write. Some adapters
        # mark progress callbacks as "analyzing" and must not overwrite the
        # completed record after it has been saved.
        progress(100, "Match review ready")
        record = self.store.update(
            match_id, story=record["story"], rallies=rallies, status="complete",
            performance=performance, progress=100, stage="Match review ready", error=None,
        )
        return record

    def build_story(self, record: Dict) -> dict:
        rallies = record.get("rallies", [])
        compact = []
        for rally in rallies:
            analysis = rally["analysis"]
            observations = analysis.get("coaching_observations", [])[:3]
            compact.append({
                "id": rally["id"], "time": [round(rally["start"], 1), round(rally["end"], 1)],
                "summary": analysis.get("summary", ""),
                "confidence": analysis.get("confidence", 0),
                "ending": analysis.get("ending", {}),
                "observations": [
                    {"team": item.get("team"), "category": item.get("category"),
                     "observation": item.get("observation"), "confidence": item.get("confidence")}
                    for item in observations
                ],
                "human_review": rally.get("review"),
            })
        digest = "\n".join(json.dumps(item, separators=(",", ":")) for item in compact)
        story = self.client.structured(
            record["model"],
            STORY_PROMPT.format(team_a=record["team_a"], team_b=record["team_b"], rallies=digest),
            MatchStory,
        )
        result = model_dump(story)
        valid_rally_ids = {rally["id"] for rally in rallies}
        has_tactical_evidence = any(
            rally["analysis"].get("coaching_observations")
            or rally["analysis"].get("tactical_phases")
            or (
                rally["analysis"].get("ending", {}).get("type") != "unknown"
                and rally["analysis"].get("ending", {}).get("confidence", 0) >= 0.55
            )
            for rally in rallies
        )
        limitation = (
            "Generated from sparse visual storyboards; this is coaching assistance, "
            "not official scoring."
        )
        if not has_tactical_evidence:
            return {
                "headline": "Not enough reliable visual evidence",
                "overview": (
                    "The sampled images did not support a trustworthy tactical match "
                    "story. Review the storyboard or analyze a longer, clearer recording."
                ),
                "team_a_story": "No evidence-backed pattern was identified.",
                "team_b_story": "No evidence-backed pattern was identified.",
                "momentum_notes": [],
                "training_priorities": [],
                "best_rallies": [],
                "limitations": [limitation],
            }

        priorities = []
        for priority in result.get("training_priorities", []):
            cited = [
                rally_id for rally_id in priority.get("evidence_rallies", [])
                if rally_id in valid_rally_ids
            ]
            if cited:
                priority["evidence_rallies"] = cited
                priorities.append(priority)
        result["training_priorities"] = priorities
        result["best_rallies"] = [
            rally_id for rally_id in result.get("best_rallies", [])
            if rally_id in valid_rally_ids
        ]
        if limitation not in result.get("limitations", []):
            result.setdefault("limitations", []).append(limitation)
        return result
