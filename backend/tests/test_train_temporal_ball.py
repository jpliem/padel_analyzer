import torch

from scripts.train_temporal_ball import heatmap_center, weighted_heatmap_loss


def test_heatmap_center_uses_per_frame_preview_scale():
    label = {"center": [1813.0, 980.0], "image_scale": 1280 / 3626}

    x, y = heatmap_center(label, 1280, 692, doc={})

    assert abs(x - 256.0) < 0.01
    assert abs(y - (980 * (1280 / 3626) * 288 / 692)) < 0.01


def test_heatmap_center_supports_legacy_dimensions():
    label = {"center": [1820.0, 980.0]}
    doc = {"original_video_width": 3640, "original_video_height": 1960}

    assert heatmap_center(label, 1280, 689, doc) == (256.0, 144.0)


def test_weighted_loss_emphasizes_positive_heatmap_pixels():
    output = torch.tensor([[[[0.5, 0.5]]]])
    targets = torch.tensor([[[[1.0, 0.0]]]])
    weights = torch.ones((1, 1))

    plain = weighted_heatmap_loss(output, targets, weights, positive_weight=0)
    weighted = weighted_heatmap_loss(output, targets, weights, positive_weight=9)

    assert weighted > plain * 5
