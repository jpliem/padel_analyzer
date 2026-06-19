from scripts.annotate_current_best import (
    event_tally,
    latest_event_label,
    point_by_frame,
    resolve_repo_path,
    score_overlay_text,
    writer_fps,
)


def test_point_by_frame_indexes_on_court_points_by_video_fps():
    points = [
        {"t": 2.0, "x": 1, "y": 2, "z": 3, "on_court": True},
        {"t": 2.1, "x": 4, "y": 5, "z": 6, "on_court": False},
        {"t": 2.2, "x": 7, "y": 8, "z": 9, "on_court": True},
    ]

    indexed = point_by_frame(points, fps=50.0)

    assert sorted(indexed) == [100, 110]
    assert indexed[100]["z"] == 3
    assert indexed[110]["x"] == 7


def test_score_overlay_text_includes_score_games_sets_and_mode():
    score = {"score": "15 - 0", "games": "1 - 0", "sets": "0 - 0"}

    assert score_overlay_text(score, mode="rally") == "15 - 0 | G 1 - 0 | S 0 - 0 | rally"


def test_event_tally_and_latest_event_label_use_frame_cutoff():
    events = [
        {"event_type": "WALL_HIT", "frame_number": 20},
        {"event_type": "BOUNCE", "frame_number": 40},
        {"event_type": "WALL_HIT", "frame_number": 60},
    ]

    assert event_tally(events) == {"WALL_HIT": 2, "BOUNCE": 1}
    assert latest_event_label(events, frame_number=45) == "BOUNCE @ f40"


def test_resolve_repo_path_leaves_absolute_paths_unchanged():
    assert resolve_repo_path("/tmp/model.pt") == "/tmp/model.pt"


def test_writer_fps_preserves_source_timing():
    assert writer_fps(50.0) == 50.0
    assert writer_fps(0.0) == 30.0
