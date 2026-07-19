import json

from scripts.merge_ball_labels import merge_documents


def _manifest(root, name, sequence):
    frames = root / "frames"
    frames.mkdir(parents=True)
    (frames / name).write_bytes(b"image")
    path = root / "labels.json"
    path.write_text(json.dumps({
        "schema_version": "1.0",
        "coordinate_space": "original_video_pixels",
        "video": f"{sequence}.mp4",
        "original_video_width": 100,
        "original_video_height": 50,
        "labels": [{
            "frame": 1,
            "image": f"frames/{name}",
            "image_scale": 0.5,
            "state": "visible",
            "center": [10, 20],
            "sequence_id": sequence,
            "camera_id": sequence,
        }],
    }))
    return path


def test_merge_preserves_distinct_sources_and_resolves_images(tmp_path):
    first = _manifest(tmp_path / "one", "one.jpg", "rally-one")
    second = _manifest(tmp_path / "two", "two.jpg", "rally-two")
    output = tmp_path / "combined" / "labels.json"

    merged = merge_documents([first, second], output)

    assert len(merged["labels"]) == 2
    assert {item["sequence_id"] for item in merged["labels"]} == {
        "rally-one", "rally-two"
    }
    for item in merged["labels"]:
        assert (output.parent / item["image"]).resolve().exists()
