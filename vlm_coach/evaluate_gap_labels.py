from __future__ import annotations

import argparse
import json
from pathlib import Path


VALID_LABELS = {"active_rally", "between_points", "unclear"}


def normalize_prediction(value: str | None) -> str:
    if value in VALID_LABELS:
        return value
    return "unclear"


def evaluate_labels(payload: dict, model_key: str) -> dict:
    reviewed = [
        item for item in payload.get("labels", [])
        if item.get("human_label") in VALID_LABELS
        and item.get("human_label") != "unreviewed"
    ]
    predictions = []
    for item in reviewed:
        truth = item["human_label"]
        predicted = normalize_prediction(item.get(model_key))
        predictions.append({
            "id": item.get("id"), "truth": truth, "predicted": predicted,
            "correct": truth == predicted,
        })
    correct = sum(item["correct"] for item in predictions)
    abstained = sum(item["predicted"] == "unclear" for item in predictions)
    return {
        "model": model_key,
        "reviewed": len(reviewed),
        "correct": correct,
        "accuracy": round(correct / len(reviewed), 3) if reviewed else None,
        "abstained": abstained,
        "coverage": round((len(reviewed) - abstained) / len(reviewed), 3) if reviewed else None,
        "predictions": predictions,
        "ready": bool(reviewed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VLM gap labels against review")
    parser.add_argument("labels")
    parser.add_argument("--model-key", default="qwen_2b")
    args = parser.parse_args()
    payload = json.loads(Path(args.labels).read_text())
    result = evaluate_labels(payload, args.model_key)
    print(json.dumps(result, indent=2))
    if not result["ready"]:
        raise SystemExit("No reviewed labels yet; accuracy is intentionally unavailable")


if __name__ == "__main__":
    main()
