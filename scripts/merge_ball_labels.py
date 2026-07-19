#!/usr/bin/env python3
"""Merge reviewed ball-label manifests without losing image provenance."""

import argparse
import copy
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend", "src"))

from models.ball_labels import SCHEMA_VERSION, validate_label_document


def merge_documents(inputs, output):
    output = os.path.abspath(output)
    output_root = os.path.dirname(output)
    merged_labels = []
    sources = []

    for input_path in inputs:
        input_path = os.path.abspath(input_path)
        with open(input_path, encoding="utf-8") as handle:
            doc = json.load(handle)
        errors = validate_label_document(doc)
        if errors:
            raise ValueError(f"invalid labels in {input_path}:\n" + "\n".join(errors))
        input_root = os.path.dirname(input_path)
        sources.append(input_path)
        for original in doc["labels"]:
            label = copy.deepcopy(original)
            image_path = os.path.abspath(os.path.join(input_root, label["image"]))
            label["image"] = os.path.relpath(image_path, output_root)
            label["source_manifest"] = os.path.relpath(input_path, output_root)
            label["source_video"] = doc.get("video")
            label["original_video_width"] = doc.get("original_video_width")
            label["original_video_height"] = doc.get("original_video_height")
            merged_labels.append(label)

    merged = {
        "schema_version": SCHEMA_VERSION,
        "coordinate_space": "original_video_pixels",
        "sources": sources,
        "labels": merged_labels,
    }
    errors = validate_label_document(merged)
    if errors:
        raise ValueError("invalid merged labels:\n" + "\n".join(errors))
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    merged = merge_documents(args.inputs, args.output)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=2)
    print(f"merged {len(merged['labels'])} labels -> {args.output}")


if __name__ == "__main__":
    main()
