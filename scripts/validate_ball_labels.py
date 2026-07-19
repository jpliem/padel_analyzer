#!/usr/bin/env python3
"""Validate labels and optionally assign leakage-safe sequence splits."""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend", "src"))

from models.ball_labels import group_safe_split, validate_label_document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("labels")
    parser.add_argument("--assign-splits", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    with open(args.labels, encoding="utf-8") as handle:
        doc = json.load(handle)

    if args.assign_splits:
        ids = [x.get("sequence_id", "") for x in doc.get("labels", [])]
        if any(not x for x in ids):
            print("cannot split: every label needs sequence_id", file=sys.stderr)
            return 2
        split_map = group_safe_split(ids)
        for item in doc["labels"]:
            item["split"] = split_map[item["sequence_id"]]
        output = args.output or args.labels
        with open(output, "w", encoding="utf-8") as handle:
            json.dump(doc, handle, indent=2)

    errors = validate_label_document(doc)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    counts = {}
    for item in doc["labels"]:
        counts[item["state"]] = counts.get(item["state"], 0) + 1
    print(f"valid: {len(doc['labels'])} labels; states={counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
