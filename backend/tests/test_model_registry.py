import json
from pathlib import Path

from cv.model_registry import get_ball_model_profile


def test_production_model_is_backed_by_reviewed_label_evidence():
    profile = get_ball_model_profile()
    assert profile["checkpoint"] == "tracknet_padel.pt"
    assert Path(profile["checkpoint_path"]).exists()
    assert profile["evidence"]["visible_labels"] == 52
    assert profile["evidence"]["matched_labels"] == 33
    assert round(33 / 52, 4) == profile["evidence"]["recall"]
    report = Path(__file__).resolve().parents[2] / profile["evidence"]["report"]
    summary = json.loads(report.read_text())["summary"]
    assert profile["evidence"]["precision"] == summary["precision"]
    assert profile["evidence"]["tolerance_px"] == summary["threshold_px"]


def test_custom_checkpoint_does_not_inherit_baseline_accuracy(tmp_path):
    profile = get_ball_model_profile(str(tmp_path / "custom.pt"))
    assert profile["status"] == "unvalidated"
    assert profile["evidence"] is None
