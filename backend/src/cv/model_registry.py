"""Production model selection and the evidence supporting it.

This is intentionally explicit: a newer checkpoint is not automatically better.
The selected model must be backed by the reviewed-label benchmark.
"""

from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional


_BACKEND_ROOT = Path(__file__).resolve().parents[2]


PRODUCTION_BALL_MODEL = {
    "id": "tracknet-padel-baseline",
    "task": "single_camera_ball_detection",
    "checkpoint": "tracknet_padel.pt",
    "checkpoint_path": str(_BACKEND_ROOT / "models" / "tracknet_padel.pt"),
    "status": "production_candidate",
    "selection_reason": "Best measured checkpoint on the held-out reviewed Panasonic rally.",
    "evidence": {
        "dataset": "PadelVic Panasonic manually reviewed labels",
        "evaluation_split": "held-out rally within the reviewed Panasonic set",
        "visible_labels": 52,
        "matched_labels": 33,
        "precision": 0.6346,
        "recall": 0.6346,
        "tolerance_px": 15,
        "report": "data/labels/padelvic_panasonic_combined/test_baseline.json",
    },
    "rejected_candidates": [
        {
            "id": "tracknet-phase1",
            "reason": "Fine-tuning collapse",
            "precision": 0.0,
            "recall": 0.0,
        },
        {
            "id": "tracknet-phase1-candidate",
            "reason": "Validation result was below the baseline",
            "precision": 0.576271186440678,
            "recall": 0.576271186440678,
        },
    ],
    "limitations": [
        "Evidence is from one Panasonic camera view and does not measure club-to-club generalization.",
        "The benchmark is small and some reviewed labels began as model suggestions.",
        "Detection accuracy is not scoring accuracy.",
        "Monocular video cannot directly recover reliable 3D depth in every rally.",
    ],
}


def get_ball_model_profile(checkpoint_path: Optional[str] = None) -> Dict:
    """Return a serializable profile, preserving evidence for custom weights."""
    profile = deepcopy(PRODUCTION_BALL_MODEL)
    if checkpoint_path:
        selected = Path(checkpoint_path).expanduser().resolve()
        baseline = Path(profile["checkpoint_path"]).resolve()
        profile["checkpoint_path"] = str(selected)
        profile["checkpoint"] = selected.name
        if selected != baseline:
            profile.update({
                "id": f"custom-{selected.stem}",
                "status": "unvalidated",
                "selection_reason": "Custom checkpoint supplied at runtime.",
                "evidence": None,
            })
    return profile
