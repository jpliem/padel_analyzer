from __future__ import annotations

import argparse
import json
from pathlib import Path


def predicted_endings(payload: dict) -> list[float]:
    endings = []
    for item in payload.get("windows", []):
        analysis = item.get("analysis", {})
        if not analysis.get("point_end_candidate"):
            continue
        timestamps = item.get("frame_timestamps", [])
        evidence = analysis.get("reset_frames") or analysis.get("evidence_frames", [])
        cited = [timestamps[index] for index in evidence if 0 <= index < len(timestamps)]
        endings.append(float(max(cited) if cited else item["window"]["end"]))
    # Overlapping windows may cite the same ending. Collapse close duplicates.
    merged = []
    for value in sorted(endings):
        if not merged or value - merged[-1] > 2.0:
            merged.append(value)
        else:
            merged[-1] = (merged[-1] + value) / 2.0
    return merged


def evaluate(labels: dict, predictions: dict, tolerance: float = 2.0) -> dict:
    excluded = [tuple(item) for item in labels.get("excluded_ranges", [])]
    allowed = lambda value: not any(start <= value <= end for start, end in excluded)
    truth = [
        float(item["end"]) for item in labels.get("labels", [])
        if item.get("certainty") != "unusable" and allowed(float(item["end"]))
    ]
    predicted = [value for value in predicted_endings(predictions) if allowed(value)]
    unmatched = set(range(len(predicted)))
    matched_errors = []
    for ending in truth:
        choices = [(abs(predicted[index] - ending), index) for index in unmatched]
        if not choices:
            continue
        error, index = min(choices)
        if error <= tolerance:
            unmatched.remove(index)
            matched_errors.append(error)
    true_positive = len(matched_errors)
    false_positive = len(predicted) - true_positive
    false_negative = len(truth) - true_positive
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(truth) if truth else 0.0
    return {
        "reviewed_rallies": len(truth), "predicted_endings": len(predicted),
        "true_positive": true_positive, "false_positive": false_positive,
        "false_negative": false_negative, "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / (precision + recall), 4)
        if precision + recall else 0.0,
        "mean_boundary_error_seconds": round(sum(matched_errors) / true_positive, 3)
        if true_positive else None,
        "tolerance_seconds": tolerance,
        "excluded_ranges": [list(item) for item in excluded],
    }


def evaluate_candidate_gate(labels: dict, candidates: dict,
                            tolerance: float = 2.0) -> dict:
    truth = [
        float(item["end"]) for item in labels.get("labels", [])
        if item.get("certainty") != "unusable"
    ]
    items = candidates.get("candidates", [])
    all_intervals = [
        (float(item["gap"]["start"]), float(item["gap"]["end"])) for item in items
    ]
    fused_intervals = [
        (float(item["gap"]["start"]), float(item["gap"]["end"])) for item in items
        if item.get("audio_supports_boundary_review")
    ]

    def covered(value: float, intervals: list[tuple[float, float]]) -> bool:
        return any(start - tolerance <= value <= end + tolerance for start, end in intervals)

    opencv_hits = [value for value in truth if covered(value, all_intervals)]
    fused_hits = [value for value in truth if covered(value, fused_intervals)]
    filtered_truth = [
        value for value in truth
        if covered(value, all_intervals) and not covered(value, fused_intervals)
    ]
    missed_opencv = [value for value in truth if not covered(value, all_intervals)]
    matched_candidates = sum(
        any(start - tolerance <= value <= end + tolerance for value in truth)
        for start, end in fused_intervals
    )
    return {
        "reviewed_rallies": len(truth),
        "opencv_candidates": len(all_intervals),
        "audio_supported_candidates": len(fused_intervals),
        "opencv_candidate_recall": round(len(opencv_hits) / len(truth), 4) if truth else 0.0,
        "fused_candidate_recall": round(len(fused_hits) / len(truth), 4) if truth else 0.0,
        "fused_candidate_precision": round(matched_candidates / len(fused_intervals), 4)
        if fused_intervals else 0.0,
        "truth_filtered_by_audio": [round(value, 3) for value in filtered_truth],
        "truth_missed_by_opencv": [round(value, 3) for value in missed_opencv],
        "tolerance_seconds": tolerance,
    }


def evaluate_pipeline_stages(labels: dict, candidates: dict, predictions: dict,
                             tolerance: float = 2.0) -> dict:
    windows = predictions.get("windows", [])
    if not windows:
        return {"reviewed_rallies_in_prediction_scope": 0,
                "error": "predictions contain no windows"}
    scope = (
        min(float(item["window"]["start"]) for item in windows),
        max(float(item["window"]["end"]) for item in windows),
    )
    truth = [
        float(item["end"]) for item in labels.get("labels", [])
        if item.get("certainty") != "unusable" and scope[0] <= float(item["end"]) <= scope[1]
    ]
    fused = [
        (float(item["gap"]["start"]), float(item["gap"]["end"]))
        for item in candidates.get("candidates", [])
        if item.get("audio_supports_boundary_review")
    ]
    predicted = predicted_endings(predictions)
    gate_misses, vlm_misses, detected = [], [], []
    for ending in truth:
        gate_kept = any(
            start - tolerance <= ending <= end + tolerance for start, end in fused
        )
        vlm_found = any(abs(value - ending) <= tolerance for value in predicted)
        if not gate_kept:
            gate_misses.append(ending)
        elif not vlm_found:
            vlm_misses.append(ending)
        else:
            detected.append(ending)
    return {
        "prediction_scope": [round(scope[0], 3), round(scope[1], 3)],
        "reviewed_rallies_in_prediction_scope": len(truth),
        "detected_endings": [round(value, 3) for value in detected],
        "candidate_gate_misses": [round(value, 3) for value in gate_misses],
        "vlm_misses_after_candidate": [round(value, 3) for value in vlm_misses],
        "end_to_end_recall": round(len(detected) / len(truth), 4) if truth else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate rolling VLM rally endings")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--candidates", help="Optional hybrid candidate JSON")
    parser.add_argument("--tolerance", type=float, default=2.0)
    args = parser.parse_args()
    labels = json.loads(Path(args.labels).read_text())
    predictions = json.loads(Path(args.predictions).read_text())
    if not labels.get("labels"):
        raise SystemExit("No reviewed rally labels; label the continuous match first")
    report = {"vlm_boundary": evaluate(labels, predictions, args.tolerance)}
    if args.candidates:
        candidates = json.loads(Path(args.candidates).read_text())
        report["candidate_gate"] = evaluate_candidate_gate(
            labels, candidates, args.tolerance
        )
        report["pipeline_stages"] = evaluate_pipeline_stages(
            labels, candidates, predictions, args.tolerance
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
