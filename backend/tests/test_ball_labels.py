from models.ball_labels import SCHEMA_VERSION, group_safe_split, validate_label_document


def _doc(labels):
    return {
        "schema_version": SCHEMA_VERSION,
        "coordinate_space": "original_video_pixels",
        "labels": labels,
    }


def test_visible_requires_center():
    errors = validate_label_document(_doc([{"frame": 1, "state": "visible"}]))
    assert any("center is required" in e for e in errors)


def test_non_visible_forbids_fake_center():
    errors = validate_label_document(_doc([
        {"frame": 1, "state": "outside_frame", "center": [10, 20]},
    ]))
    assert any("center must be null" in e for e in errors)


def test_complete_label_is_valid():
    assert validate_label_document(_doc([{
        "frame": 1, "state": "visible", "center": [10.5, 20.5],
        "event_tags": ["bounce"], "sequence_id": "rally-1", "split": "train",
    }])) == []


def test_split_is_deterministic_and_group_safe():
    first = group_safe_split(["r1", "r2", "r1"])
    assert first == group_safe_split(["r2", "r1"])
    assert set(first.values()) <= {"train", "val", "test"}
