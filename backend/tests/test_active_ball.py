from cv.active_ball import ActiveBallSelector


def candidate(cx, cy, confidence=0.8):
    return {"bbox": [cx - 5, cy - 5, cx + 5, cy + 5],
            "confidence": confidence, "source": "test"}


def center(selection):
    x1, y1, x2, y2 = selection.bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def test_temporal_continuity_rejects_high_confidence_spare_ball():
    selector = ActiveBallSelector(max_jump_px=80)
    selector.select([candidate(100, 100)], None)
    selector.select([candidate(115, 100)], None)
    selected = selector.select([candidate(130, 101, 0.72), candidate(500, 500, 0.99)], None)
    assert center(selected) == (130, 101)
    assert selected.rejected_count == 1


def test_impossible_jump_becomes_uncertainty_not_a_measurement():
    selector = ActiveBallSelector(max_jump_px=50)
    selector.select([candidate(100, 100)], None)
    selected = selector.select([candidate(500, 400)], None)
    assert selected.bbox is None
    assert selected.state == "uncertain"
    assert "motion gate" in selected.reason


def test_multiple_objects_at_acquisition_are_marked_uncertain():
    selector = ActiveBallSelector()
    selected = selector.select([candidate(100, 100, 0.9), candidate(300, 300, 0.7)], None)
    assert selected.bbox is not None
    assert selected.state == "uncertain"
    assert selected.confidence < 0.9


def test_reacquires_after_sustained_occlusion():
    selector = ActiveBallSelector(max_jump_px=40, reacquire_after=3)
    selector.select([candidate(100, 100)], None)
    for _ in range(3):
        selector.select([], None)
    selected = selector.select([candidate(500, 400)], None)
    assert selected.bbox is not None
    assert selected.state == "acquired"
