#!/usr/bin/env python3
"""Fine-tune the 9-frame TrackNet model on validated padel ball labels.

Only reviewed labels are used. Splits are grouped by sequence/rally, so nearby
frames cannot leak into both training and validation.
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend", "src"))


def heatmap_center(label, preview_width, preview_height, doc,
                   output_width=512, output_height=288):
    """Map an original-pixel label through its own preview into model space."""
    center = label["center"]
    image_scale = label.get("image_scale")
    if image_scale is not None:
        preview_x = float(center[0]) * float(image_scale)
        preview_y = float(center[1]) * float(image_scale)
        return (
            preview_x * output_width / float(preview_width),
            preview_y * output_height / float(preview_height),
        )
    # Backward compatibility for legacy manifests without per-frame scale.
    return (
        float(center[0]) * output_width / float(doc["original_video_width"]),
        float(center[1]) * output_height / float(doc["original_video_height"]),
    )


def weighted_heatmap_loss(output, targets, sample_weights, positive_weight,
                          loss_fn=None):
    """Prevent tiny positive ball heatmaps from being drowned by background."""
    import torch

    if loss_fn is None:
        loss_fn = torch.nn.BCELoss(reduction="none")
    pixel_weights = 1.0 + targets * float(positive_weight)
    return (
        loss_fn(output, targets)
        * pixel_weights
        * sample_weights[:, :, None, None]
    ).mean()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--sigma", type=float, default=3.0)
    parser.add_argument(
        "--positive-weight", type=float, default=250.0,
        help="extra BCE weight at positive heatmap pixels",
    )
    parser.add_argument("--input-width", type=int, default=512)
    parser.add_argument("--input-height", type=int, default=288)
    parser.add_argument(
        "--freeze-backbone", action="store_true",
        help="train only the predictor head; useful for small datasets and CPU",
    )
    args = parser.parse_args()

    import cv2
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset

    from cv.detectors.device import get_device
    from cv.detectors.tracknet import TrackNetV2Model, _state_dict_from_checkpoint
    from models.ball_labels import validate_label_document

    with open(args.labels, encoding="utf-8") as handle:
        doc = json.load(handle)
    errors = validate_label_document(doc)
    if errors:
        raise SystemExit("invalid labels:\n" + "\n".join(errors))
    label_root = os.path.dirname(os.path.abspath(args.labels))

    class SequenceDataset(Dataset):
        def __init__(self, split):
            grouped = {}
            for label in doc["labels"]:
                if label.get("split") == split and label["state"] != "unreviewed":
                    grouped.setdefault(label["sequence_id"], []).append(label)
            self.windows = []
            for labels in grouped.values():
                labels.sort(key=lambda x: x["frame"])
                for i in range(len(labels) - 8):
                    window = labels[i:i + 9]
                    # Uncertain targets are not silently converted to negatives.
                    if all(x["state"] != "uncertain" for x in window[1:]):
                        self.windows.append(window)

        def __len__(self):
            return len(self.windows)

        def __getitem__(self, index):
            frames = []
            targets = []
            weights = []
            for i, label in enumerate(self.windows[index]):
                path = os.path.join(label_root, label["image"])
                frame = cv2.imread(path)
                if frame is None:
                    raise FileNotFoundError(path)
                preview_height, preview_width = frame.shape[:2]
                frame = cv2.resize(
                    frame, (args.input_width, args.input_height)
                ).astype(np.float32) / 255.0
                frames.append(frame)
                if i == 0:
                    continue
                heatmap = np.zeros((args.input_height, args.input_width), np.float32)
                if label["state"] in ("visible", "blurred"):
                    x, y = heatmap_center(
                        label, preview_width, preview_height, doc,
                        output_width=args.input_width,
                        output_height=args.input_height,
                    )
                    yy, xx = np.mgrid[:args.input_height, :args.input_width]
                    heatmap = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) /
                                     (2 * args.sigma ** 2)).astype(np.float32)
                    weights.append(0.6 if label["state"] == "blurred" else 1.0)
                else:
                    weights.append(1.0 if label["state"] == "hard_negative" else 0.5)
                targets.append(heatmap)
            x = np.concatenate(frames, axis=2).transpose(2, 0, 1)
            return (torch.from_numpy(x), torch.from_numpy(np.stack(targets)),
                    torch.tensor(weights, dtype=torch.float32))

    train_data = SequenceDataset("train")
    val_data = SequenceDataset("val")
    if not train_data or not val_data:
        raise SystemExit("need at least one 9-frame train and validation window")

    device = torch.device(get_device())
    model = TrackNetV2Model(in_dim=27, out_dim=8)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(_state_dict_from_checkpoint(checkpoint))
    if args.freeze_backbone:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("predictor.")
    model.to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
    loss_fn = torch.nn.BCELoss(reduction="none")
    loaders = {
        "train": DataLoader(train_data, batch_size=args.batch_size, shuffle=True),
        "val": DataLoader(val_data, batch_size=args.batch_size),
    }
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        metrics = {}
        for split, loader in loaders.items():
            if args.freeze_backbone:
                # Keep frozen BatchNorm statistics fixed while the head learns.
                model.eval()
            else:
                model.train(split == "train")
            total = 0.0
            for inputs, targets, weights in loader:
                inputs, targets, weights = inputs.to(device), targets.to(device), weights.to(device)
                with torch.set_grad_enabled(split == "train"):
                    output = model(inputs)
                    loss = weighted_heatmap_loss(
                        output, targets, weights, args.positive_weight, loss_fn
                    )
                    if split == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                total += float(loss.detach())
            metrics[split] = total / len(loader)
        print(
            f"epoch={epoch} train={metrics['train']:.6f} "
            f"val={metrics['val']:.6f}",
            flush=True,
        )
        if metrics["val"] < best:
            best = metrics["val"]
            torch.save({
                "model": model.state_dict(), "schema_version": doc["schema_version"],
                "val_loss": best, "epoch": epoch, "input_frames": 9,
                "training_input_size": [args.input_width, args.input_height],
                "frozen_backbone": args.freeze_backbone,
                "positive_weight": args.positive_weight,
            }, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
