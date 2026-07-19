from cv.player_pose import wrist_proximity_confidence


def test_near_visible_wrist_is_strong_evidence():
    points = [[0, 0, 0]] * 17
    points[9] = [105, 100, .9]
    assert wrist_proximity_confidence((100, 100), points, radius_px=50) > .8


def test_hidden_wrists_produce_no_evidence():
    points = [[0, 0, 0]] * 17
    points[9] = [100, 100, .1]
    points[10] = [100, 100, .1]
    assert wrist_proximity_confidence((100, 100), points) == 0
