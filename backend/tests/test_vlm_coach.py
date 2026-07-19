from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from vlm_coach import app as coach_app
from vlm_coach.ollama_client import OllamaClient
from vlm_coach.audio_probe import cadence_summary, quiet_intervals, spectral_impulses
from vlm_coach.mlx_client import MlxVlmClient
from vlm_coach.molmo_benchmark import parse_molmo_points
from vlm_coach.evaluate_gap_labels import evaluate_labels
from vlm_coach.hybrid_candidate_probe import overlap_seconds
from vlm_coach.evaluate_rally_labels import (
    evaluate, evaluate_candidate_gate, evaluate_pipeline_stages, predicted_endings,
)
from vlm_coach.scoring import (
    GapDecision, RallyScout, RallyVerdict, RollingObservation,
    derive_frame_state_window,
    fixed_windows, gap_contexts,
    merge_motion_bursts, merge_scout_windows, add_scoring_context,
    reconcile_rolling_observation, rolling_context, rolling_windows,
    sanitize_frame_evidence,
)
from vlm_coach.pipeline import MatchPipeline
from vlm_coach.schemas import (
    CoachingObservation, MatchStory, RallyAnalysis, model_schema,
)
from vlm_coach.store import MatchStore
from vlm_coach.video import (
    ActivitySample, Segment, extract_storyboard, scoring_timestamps,
    segments_from_activity, storyboard_panels,
)


def test_rolling_windows_overlap_without_reordering():
    windows = rolling_windows(30, length=12, overlap=3)
    assert windows == [Segment(0, 12), Segment(9, 21), Segment(18, 30)]


def test_rolling_context_is_bounded_to_previous_decision():
    context = rolling_context({
        "phase_at_end": "rally", "rally_visible": True,
        "observed_change": "Near team approaches net.", "unused": "drop me",
    })
    assert '"phase_at_end":"rally"' in context
    assert "unused" not in context
    assert "approaches net" not in context


def test_rolling_observation_requires_ending_evidence():
    observation = RollingObservation(
        phase_at_start="rally", phase_at_end="point_ending",
        relation_to_previous="same_rally", rally_visible=True,
        point_end_candidate=True, evidence_frames=[],
        active_play_frames=[1, 2], reset_frames=[],
        observed_change="Players stop.", confidence=0.9,
    )
    assert observation.point_end_candidate is False


def test_rolling_observation_accepts_ordered_play_to_reset_evidence():
    observation = RollingObservation(
        phase_at_start="rally", phase_at_end="between_points",
        relation_to_previous="same_rally", rally_visible=True,
        point_end_candidate=True, evidence_frames=[3, 7],
        active_play_frames=[2, 3], reset_frames=[7, 8],
        observed_change="Athletic play changes to relaxed reset.", confidence=0.9,
    )
    assert observation.point_end_candidate is True
    assert observation.rally_visible is True


def test_rolling_observation_rejects_reset_before_play():
    observation = RollingObservation(
        phase_at_start="between_points", phase_at_end="rally",
        relation_to_previous="new_rally", rally_visible=True,
        point_end_candidate=True, evidence_frames=[1, 7],
        active_play_frames=[7], reset_frames=[1],
        observed_change="Reset changes to active play.", confidence=0.9,
    )
    assert observation.point_end_candidate is False


def test_rolling_observation_rejects_uncited_active_play():
    observation = RollingObservation(
        phase_at_start="rally", phase_at_end="rally",
        relation_to_previous="same_rally", rally_visible=True,
        point_end_candidate=False, evidence_frames=[],
        active_play_frames=[], reset_frames=[],
        observed_change="Players appear active.", confidence=1.0,
    )
    assert observation.rally_visible is False


def test_frame_state_derives_ordered_active_to_reset_transition():
    payload = {"frames": [
        {"index": 0, "state": "active_play", "confidence": 0.9},
        {"index": 1, "state": "active_play", "confidence": 0.8},
        {"index": 2, "state": "reset", "confidence": 0.9},
        {"index": 3, "state": "reset", "confidence": 0.8},
    ]}
    result = derive_frame_state_window(payload, 4)
    assert result["coverage_complete"] is True
    assert result["point_end_candidate"] is True
    assert result["active_play_frames"] == [0, 1]
    assert result["reset_frames"] == [2, 3]


def test_frame_state_finds_boundary_even_if_new_play_resumes():
    payload = {"frames": [
        {"index": 0, "state": "active_play", "confidence": 0.9},
        {"index": 1, "state": "active_play", "confidence": 0.9},
        {"index": 2, "state": "reset", "confidence": 0.9},
        {"index": 3, "state": "reset", "confidence": 0.9},
        {"index": 4, "state": "active_play", "confidence": 0.9},
    ]}
    result = derive_frame_state_window(payload, 5)
    assert result["point_end_candidate"] is True


def test_frame_state_rejects_one_frame_reset_noise():
    payload = {"frames": [
        {"index": 0, "state": "active_play", "confidence": 0.9},
        {"index": 1, "state": "active_play", "confidence": 0.9},
        {"index": 2, "state": "reset", "confidence": 0.9},
        {"index": 3, "state": "active_play", "confidence": 0.9},
    ]}
    assert derive_frame_state_window(payload, 4)["point_end_candidate"] is False


def test_rolling_reconciliation_rejects_same_rally_after_idle_state():
    current = reconcile_rolling_observation(
        {"phase_at_end": "between_points", "rally_visible": False},
        {"relation_to_previous": "same_rally", "rally_visible": False,
         "confidence": 0.95},
    )
    assert current["relation_to_previous"] == "unclear"
    assert current["confidence"] == 0.59


def test_sanitizer_does_not_add_fields_to_rolling_schema():
    decision = sanitize_frame_evidence({
        "rally_visible": True, "point_end_candidate": True,
        "evidence_frames": [99], "confidence": 0.9,
    }, 3)
    assert "serve_candidate" not in decision


def test_mlx_structured_rejects_mixed_image_and_video_inputs():
    try:
        MlxVlmClient().structured(
            "qwen3.5:0.8b", "inspect", RollingObservation,
            images=[Path("frame.jpg")], videos=[Path("clip.mp4")],
        )
    except ValueError as exc:
        assert "images or videos" in str(exc)
    else:
        raise AssertionError("mixed visual input must be rejected")


def test_rally_evaluator_collapses_overlap_and_matches_boundaries():
    predictions = {"windows": [
        {"window": {"end": 11}, "frame_timestamps": [8, 9, 10],
         "analysis": {"point_end_candidate": True, "evidence_frames": [2]}},
        {"window": {"end": 12}, "frame_timestamps": [9, 10, 11],
         "analysis": {"point_end_candidate": True, "evidence_frames": [1]}},
        {"window": {"end": 22}, "frame_timestamps": [20, 21, 22],
         "analysis": {"point_end_candidate": True, "evidence_frames": [1]}},
    ]}
    assert predicted_endings(predictions) == [10.0, 21.0]
    report = evaluate({"labels": [
        {"end": 10.5, "certainty": "certain"},
        {"end": 21.4, "certainty": "uncertain"},
        {"end": 30, "certainty": "unusable"},
    ]}, predictions, tolerance=1.0)
    assert report["true_positive"] == 2
    assert report["precision"] == 1.0
    assert report["recall"] == 1.0


def test_candidate_gate_attributes_audio_and_opencv_misses():
    labels = {"labels": [
        {"end": 10, "certainty": "certain"},
        {"end": 20, "certainty": "certain"},
        {"end": 30, "certainty": "certain"},
    ]}
    candidates = {"candidates": [
        {"gap": {"start": 9, "end": 11}, "audio_supports_boundary_review": True},
        {"gap": {"start": 19, "end": 21}, "audio_supports_boundary_review": False},
    ]}
    report = evaluate_candidate_gate(labels, candidates, tolerance=0.5)
    assert report["opencv_candidate_recall"] == 0.6667
    assert report["fused_candidate_recall"] == 0.3333
    assert report["truth_filtered_by_audio"] == [20.0]
    assert report["truth_missed_by_opencv"] == [30.0]


def test_pipeline_stage_report_respects_prediction_scope():
    labels = {"labels": [
        {"end": 10, "certainty": "certain"},
        {"end": 20, "certainty": "certain"},
        {"end": 40, "certainty": "certain"},
    ]}
    candidates = {"candidates": [
        {"gap": {"start": 9, "end": 11}, "audio_supports_boundary_review": True},
        {"gap": {"start": 19, "end": 21}, "audio_supports_boundary_review": True},
    ]}
    predictions = {"windows": [
        {"window": {"start": 0, "end": 25}, "frame_timestamps": [9, 10, 11],
         "analysis": {"point_end_candidate": True, "evidence_frames": [1]}},
    ]}
    report = evaluate_pipeline_stages(labels, candidates, predictions, tolerance=1)
    assert report["reviewed_rallies_in_prediction_scope"] == 2
    assert report["detected_endings"] == [10.0]
    assert report["vlm_misses_after_candidate"] == [20.0]
    assert report["end_to_end_recall"] == 0.5


def _video(path: Path, seconds: float = 2.0, fps: int = 10):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 90))
    assert writer.isOpened()
    for frame_no in range(int(seconds * fps)):
        frame = np.zeros((90, 160, 3), dtype=np.uint8)
        cv2.circle(frame, (20 + frame_no * 3, 45), 8, (80, 220, 230), -1)
        writer.write(frame)
    writer.release()


def test_storyboard_keeps_dense_chronological_indices_and_labels(tmp_path):
    video = tmp_path / "source.mp4"
    _video(video)
    frames = extract_storyboard(
        video, Segment(0, 2), tmp_path / "frames", frame_count=4,
        annotate_timeline=True,
    )
    assert [item["index"] for item in frames] == list(range(len(frames)))
    assert [item["timestamp"] for item in frames] == sorted(
        item["timestamp"] for item in frames
    )
    image = cv2.imread(str(frames[0]["path"]))
    assert image is not None
    assert float(image[:50, :370].mean()) > 0


def test_storyboard_panels_pack_four_frames_row_major(tmp_path):
    video = tmp_path / "source.mp4"
    _video(video)
    frames = extract_storyboard(
        video, Segment(0, 2), tmp_path / "frames", frame_count=5,
        annotate_timeline=True,
    )
    panels = storyboard_panels(frames, tmp_path / "panels")
    assert len(panels) == 2
    first = cv2.imread(str(panels[0]))
    assert first.shape[1] == 768
    assert first.shape[0] > 0


def test_audio_probe_finds_separated_synthetic_impulses():
    sample_rate = 8000
    samples = np.zeros(sample_rate * 2, dtype=np.float32)
    rng = np.random.default_rng(4)
    for timestamp in (0.4, 1.1, 1.7):
        start = int(timestamp * sample_rate)
        samples[start:start + 80] = rng.normal(0, 1, 80)
    result = spectral_impulses(samples, sample_rate)
    assert len(result["times"]) >= 3
    summary = cadence_summary(result["times"], 2.0)
    assert summary["candidate_impulses"] >= 3


def test_audio_quiet_intervals_ignore_short_hit_spacing():
    assert quiet_intervals([0.2, 0.8, 3.1, 3.7], 4.0) == [[0.8, 3.1]]


def test_hybrid_candidate_overlap_is_clipped_to_motion_gap():
    assert overlap_seconds((10, 15), (8, 12)) == 2
    assert overlap_seconds((10, 15), (16, 18)) == 0


def test_activity_samples_become_padded_segments():
    samples = [
        ActivitySample(0, 0), ActivitySample(1, 8), ActivitySample(2, 9),
        ActivitySample(3, 7), ActivitySample(8, 9), ActivitySample(9, 8),
        ActivitySample(10, 9), ActivitySample(11, 8),
    ]
    segments = segments_from_activity(samples, duration=15, threshold=5,
                                      idle_gap=2, min_duration=2, padding=1)
    assert segments == [Segment(0, 4), Segment(7, 12)]


def test_no_motion_falls_back_to_bounded_windows():
    samples = [ActivitySample(i, 0) for i in range(70)]
    segments = segments_from_activity(samples, duration=70, max_duration=35)
    assert segments == [Segment(0, 35), Segment(35, 70)]


def test_store_round_trip_and_atomic_update(tmp_path):
    store = MatchStore(tmp_path)
    match = store.create("League night", "Alice / Bea", "Cara / Dee", "qwen3.5:2b", "game.mp4")
    store.update(match["id"], progress=42, stage="Reviewing")
    loaded = store.load(match["id"])
    assert loaded["progress"] == 42
    assert loaded["team_a"] == "Alice / Bea"
    assert not list(store.directory(match["id"]).glob("*.tmp"))


def test_list_returns_newest_match_first(tmp_path):
    store = MatchStore(tmp_path)
    first = store.create("First", "A", "B", "qwen3.5:2b", "one.mp4")
    second = store.create("Second", "A", "B", "qwen3.5:2b", "two.mp4")
    first_record = store.load(first["id"])
    first_record["created_at"] = "2020-01-01T00:00:00+00:00"
    store.save(first_record)
    assert [item["id"] for item in store.list()] == [second["id"], first["id"]]


def test_ollama_client_retries_invalid_structured_output(monkeypatch):
    responses = ["not json", '{"summary":"Visible coordinated movement"}']
    requests = []

    class Response:
        def __init__(self, content):
            self.content = content

        def raise_for_status(self):
            return None

        def json(self):
            return {"message": {"content": self.content}}

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def post(self, url, json):
            requests.append(json)
            return Response(responses.pop(0))

    monkeypatch.setattr("vlm_coach.ollama_client.httpx.Client", Client)
    result = OllamaClient().structured("qwen3.5:0.8b", "Review this", RallyAnalysis)
    assert result.summary == "Visible coordinated movement"
    assert len(requests) == 2
    assert "previous response" in requests[1]["messages"][0]["content"]


def test_rally_schema_bounds_generated_text():
    schema = model_schema(RallyAnalysis)
    assert schema["properties"]["summary"]["maxLength"] == 360
    evidence = schema["$defs"]["TacticalPhase"]["properties"]["evidence"]
    assert evidence["maxLength"] == 240
    uncertainty = schema["properties"]["uncertainty"]
    assert uncertainty["maxItems"] == 4
    assert uncertainty["items"]["maxLength"] == 240


def test_parse_molmo_multi_image_points():
    raw = '<points coords="1 1 250 500, 2 1 500 750; 2 2 800 200"/>'
    points = parse_molmo_points(raw, [(800, 600), (1000, 500)])
    assert points == [
        {"frame": 0, "object_id": 1, "x": 200.0, "y": 300.0,
         "x_normalized": 0.25, "y_normalized": 0.5},
        {"frame": 1, "object_id": 1, "x": 500.0, "y": 375.0,
         "x_normalized": 0.5, "y_normalized": 0.75},
        {"frame": 1, "object_id": 2, "x": 800.0, "y": 100.0,
         "x_normalized": 0.8, "y_normalized": 0.2},
    ]


def test_fixed_scoring_windows_overlap_and_cover_end():
    assert fixed_windows(10, length=6, stride=3) == [
        Segment(0, 6), Segment(3, 9), Segment(6, 10),
    ]


def test_scoring_storyboard_samples_the_end_densely():
    timestamps = scoring_timestamps(Segment(10, 30), 10)
    assert timestamps[-1] == 29.95
    assert len([value for value in timestamps if value >= 25.95]) >= 4
    assert timestamps[0] == 10


def test_scout_windows_merge_into_one_padded_candidate():
    quiet = RallyScout(
        phase_at_start="waiting", phase_at_end="waiting", rally_visible=False,
        serve_candidate=False, point_end_candidate=False, evidence_frames=[],
        explanation="Players wait.", confidence=0.8,
    )
    active = RallyScout(phase_at_start="serve_started", phase_at_end="rally",
                        rally_visible=True, serve_candidate=True, confidence=0.8,
                        evidence_frames=[1, 2], point_end_candidate=False,
                        explanation="Players move through ready positions.")
    ending = RallyScout(phase_at_start="rally", phase_at_end="point_ending",
                        rally_visible=True, point_end_candidate=True, confidence=0.8,
                        evidence_frames=[2, 3], serve_candidate=False,
                        explanation="Active movement changes to a reset.")
    items = [
        (Segment(0, 6), quiet), (Segment(3, 9), active),
        (Segment(6, 12), ending), (Segment(9, 15), quiet),
    ]
    assert merge_scout_windows(items) == [Segment(1, 14)]


def test_verdict_cannot_award_unsupported_point():
    verdict = RallyVerdict(
        valid_rally=False, point_ended=False, winner="team_b", confidence=0.0,
        decisive_team="team_b", decisive_outcome="won_point",
        ending="unknown", evidence_frames=[], explanation="No visible ending.",
        review_required=False,
    )
    assert verdict.winner == "unknown"
    assert verdict.review_required is True


def test_scout_cannot_trigger_expensive_judge_without_evidence():
    scout = RallyScout(
        phase_at_start="unclear", phase_at_end="unclear", rally_visible=False,
        serve_candidate=False, point_end_candidate=True, evidence_frames=[],
        explanation="Unclear.", confidence=0.0,
    )
    assert scout.point_end_candidate is False


def test_high_confidence_rally_can_trigger_judge_without_citations():
    scout = RallyScout(
        phase_at_start="waiting", phase_at_end="rally", rally_visible=True,
        serve_candidate=False, point_end_candidate=False, evidence_frames=[],
        explanation="Players actively exchange shots.", confidence=0.9,
    )
    assert scout.rally_visible is True


def test_verdict_keeps_evidence_backed_high_confidence_winner():
    verdict = RallyVerdict(
        valid_rally=True, point_ended=True, winner="team_a", confidence=0.81,
        decisive_team="team_a", decisive_outcome="won_point",
        ending="winner", evidence_frames=[8, 9], explanation="Team B failed to return.",
        review_required=False,
    )
    assert verdict.winner == "team_a"


def test_contradictory_decisive_action_blocks_winner():
    verdict = RallyVerdict(
        valid_rally=True, point_ended=True, winner="team_a",
        decisive_team="team_a", decisive_outcome="lost_point",
        ending="forced_error", evidence_frames=[8, 9],
        explanation="Team A made the final error.", confidence=0.9,
        review_required=False,
    )
    assert verdict.winner == "unknown"
    assert verdict.review_required is True


def test_invalid_frame_citation_blocks_automatic_decision():
    decision = {
        "valid_rally": True, "point_ended": True, "winner": "team_a",
        "evidence_frames": [8, 13], "confidence": 0.9, "review_required": False,
    }
    sanitized = sanitize_frame_evidence(decision, frame_count=10)
    assert sanitized["evidence_frames"] == [8]
    assert sanitized["winner"] == "unknown"
    assert sanitized["confidence"] == 0.49
    assert sanitized["review_required"] is True


def test_gap_scout_keeps_valid_citations_when_one_is_invalid():
    decision = {
        "middle_state": "between_points", "evidence_frames": [3, 4, 12],
        "confidence": 0.9,
    }
    sanitized = sanitize_frame_evidence(decision, frame_count=10, strict=False)
    assert sanitized["middle_state"] == "between_points"
    assert sanitized["evidence_frames"] == [3, 4]
    assert sanitized["confidence"] == 0.9


def test_scoring_context_adds_postroll_without_overlapping_next_candidate():
    segments = [Segment(10, 20), Segment(23, 30)]
    assert add_scoring_context(segments, duration=35, postroll=4) == [
        Segment(10, 22.9), Segment(23, 34),
    ]


def test_gap_context_includes_both_motion_shoulders():
    assert gap_contexts([Segment(10, 20), Segment(24, 30)], 40) == [
        (Segment(20, 24), Segment(18, 26)),
    ]


def test_gap_classifier_merges_uncertain_boundaries():
    decision = GapDecision(
        middle_state="between_points", evidence_frames=[],
        explanation="Unclear lull.", confidence=0.9,
    )
    assert decision.same_rally_continues is True
    assert decision.middle_state == "unclear"


def test_motion_bursts_split_only_on_supported_between_points_gap():
    continuation = GapDecision(
        middle_state="active_rally", evidence_frames=[2, 3],
        explanation="Ready positions continue.", confidence=0.9,
    )
    boundary = GapDecision(
        middle_state="between_points", evidence_frames=[3, 4],
        explanation="Players reset.", confidence=0.9,
    )
    bursts = [Segment(10, 20), Segment(24, 30), Segment(36, 42)]
    assert merge_motion_bursts(bursts, [continuation, boundary]) == [
        Segment(10, 30), Segment(36, 42),
    ]


def test_gap_evaluator_refuses_to_invent_accuracy_without_review():
    result = evaluate_labels({"labels": [{
        "id": 1, "human_label": "unreviewed", "qwen_2b": "between_points",
    }]}, "qwen_2b")
    assert result["ready"] is False
    assert result["accuracy"] is None


def test_gap_evaluator_measures_reviewed_predictions_and_abstention():
    result = evaluate_labels({"labels": [
        {"id": 1, "human_label": "between_points", "qwen_2b": "between_points"},
        {"id": 2, "human_label": "active_rally", "qwen_2b": "unclear"},
    ]}, "qwen_2b")
    assert result["accuracy"] == 0.5
    assert result["coverage"] == 0.5


class FakeOllama:
    def health(self):
        return {"available": True, "models": ["qwen3.5:2b"]}

    def structured(self, model, prompt, output_type, images=()):
        if output_type is RallyAnalysis:
            return RallyAnalysis(
                summary="Team A moved forward together while Team B defended.",
                confidence=0.72,
                rally_quality="medium",
                highlight_score=0.64,
                coaching_observations=[CoachingObservation(
                    team="team_a", category="net_control",
                    observation="Team A moved forward together.", confidence=0.72,
                )],
                uncertainty=["The point ending is outside the sampled evidence."],
            )
        assert output_type is MatchStory
        return MatchStory(
            headline="Team A found better net position",
            overview="The reviewed segment shows a coordinated transition.",
            team_a_story="Advanced together.",
            team_b_story="Defended from deep court.",
            limitations=["Sparse storyboard review, not official scoring."],
        )


def test_pipeline_creates_evidence_linked_story(tmp_path, monkeypatch):
    store = MatchStore(tmp_path / "matches")
    record = store.create("Test", "A", "B", "qwen3.5:2b", "clip.mp4")
    video = store.video_path(record["id"])
    _video(video)
    monkeypatch.setattr("vlm_coach.pipeline.detect_segments", lambda path: [Segment(0, 2)])
    pipeline = MatchPipeline(store, FakeOllama())
    def progress(percent, stage):
        store.update(record["id"], status="analyzing", progress=percent, stage=stage)

    result = pipeline.analyze(record["id"], progress)
    assert result["status"] == "complete"
    assert result["rallies"][0]["storyboard"][0]["url"].startswith("/media/")
    assert result["story"]["headline"] == "Team A found better net position"
    assert result["performance"]["storyboards"] == 1
    assert result["performance"]["images_sent"] == 8
    assert store.load(record["id"])["progress"] == 100


def test_story_refuses_to_invent_advice_without_tactical_evidence(tmp_path):
    store = MatchStore(tmp_path / "matches")
    pipeline = MatchPipeline(store, FakeOllama())
    story = pipeline.build_story({
        "model": "qwen3.5:2b", "team_a": "A", "team_b": "B",
        "rallies": [{
            "id": 1, "start": 0, "end": 5, "review": None,
            "analysis": RallyAnalysis(summary="Teams are visible").model_dump(),
        }],
    })
    assert story["headline"] == "Not enough reliable visual evidence"
    assert story["training_priorities"] == []
    assert "not official scoring" in story["limitations"][0]


def test_api_upload_review_and_delete(tmp_path, monkeypatch):
    store = MatchStore(tmp_path / "matches")
    monkeypatch.setattr(coach_app, "store", store)
    monkeypatch.setattr(coach_app, "pipeline", MatchPipeline(store, FakeOllama()))
    monkeypatch.setattr(coach_app, "ollama", FakeOllama())
    source = tmp_path / "upload.mp4"
    _video(source)
    client = TestClient(coach_app.app)

    with source.open("rb") as handle:
        response = client.post("/api/matches", data={
            "name": "Sunday", "team_a": "A", "team_b": "B", "model": "qwen3.5:2b",
        }, files={"file": ("upload.mp4", handle, "video/mp4")})
    assert response.status_code == 200
    match_id = response.json()["id"]

    record = store.load(match_id)
    record["rallies"] = [{"id": 1, "analysis": {}, "review": None}]
    store.save(record)
    reviewed = client.patch(f"/api/matches/{match_id}/rallies/1", json={
        "winner": "team_b", "ending": "forced_error", "note": "Confirmed by player",
    })
    assert reviewed.status_code == 200
    assert store.load(match_id)["rallies"][0]["review"]["winner"] == "team_b"

    deleted = client.delete(f"/api/matches/{match_id}")
    assert deleted.status_code == 200
    assert not store.directory(match_id).exists()
