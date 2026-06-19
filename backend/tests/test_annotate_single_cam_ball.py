from scripts.annotate_single_cam_ball import choose_candidate, candidate_quality, parse_hsv


def test_parse_hsv_converts_comma_triplet_to_ints():
    assert parse_hsv("22,50,110") == [22, 50, 110]


def test_candidate_quality_prefers_round_filled_small_blobs():
    good = {"r": 4.0, "circularity": 0.8, "fill_ratio": 0.7, "aspect_ratio": 1.1}
    bad = {"r": 15.0, "circularity": 0.4, "fill_ratio": 0.35, "aspect_ratio": 2.0}

    assert candidate_quality(good) > candidate_quality(bad)


def test_choose_candidate_prefers_continuity_when_previous_exists():
    candidates = [
        {"x": 100.0, "y": 100.0, "r": 4.0, "circularity": 0.8, "fill_ratio": 0.7, "aspect_ratio": 1.0},
        {"x": 300.0, "y": 300.0, "r": 3.0, "circularity": 0.95, "fill_ratio": 0.9, "aspect_ratio": 1.0},
    ]

    chosen = choose_candidate(candidates, previous={"x": 110.0, "y": 108.0})

    assert chosen["x"] == 100.0
    assert chosen["y"] == 100.0


def test_choose_candidate_returns_none_for_empty_candidates():
    assert choose_candidate([], previous=None) is None
