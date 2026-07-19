import subprocess
import sys
from pathlib import Path


def test_known_padelvic_csv_is_rejected_as_ball_ground_truth(tmp_path):
    root = tmp_path / "padelvic" / "synthetic"
    root.mkdir(parents=True)
    clip = root / "clip.mkv"
    csv = root / "clip.csv"
    clip.write_bytes(b"placeholder")
    csv.write_text("0;1;2\n")
    result = subprocess.run([
        sys.executable, str(Path(__file__).parents[2] / "scripts" / "eval_synthetic.py"),
        "--clip", str(clip), "--csv", str(csv), "--max-frames", "1",
    ], capture_output=True, text=True)
    assert result.returncode == 2
    assert "not verified ball centers" in result.stderr
