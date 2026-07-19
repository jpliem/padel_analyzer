import numpy as np

from cv.monocular_trajectory import MonocularTrajectoryEstimator, RayObservation
from cv.visibility import BallVisibilityTracker
from models.court_model import PadelCourtModel
from models.observations import BallVisibility


def test_court_region_classifies_gate_and_recovery_area():
    court = PadelCourtModel({"out_of_court_play_enabled": True})
    assert court.classify_region(5, 10, 1) == "court"
    assert court.classify_region(-0.1, 10, 1) == "left_gate"
    assert court.classify_region(-1.0, 10, 1) == "outside_recovery_left"
    assert court.classify_region(11.0, 10, 1) == "outside_recovery_right"
    assert court.is_recoverable_position(-1.0, 10, 1)
    assert not court.is_recoverable_position(-4.0, 10, 1)


def test_court_infers_gate_and_wall_exit_portals():
    court = PadelCourtModel({"out_of_court_play_enabled": True})
    assert court.infer_exit_portal((1, 10, 1), (-1, 10, 1)) == "left_gate"
    assert court.infer_exit_portal((1, 5, 2), (-1, 5, 2)) == "left_side_over_wall"
    assert court.infer_exit_portal((5, 1, 2), (5, -1, 2)) == "near_back_wall"


def test_visibility_loss_is_explicit_and_recovers():
    tracker = BallVisibilityTracker()
    assert tracker.update(detected=True, detection_confidence=.9).state == BallVisibility.VISIBLE
    hidden = tracker.update(detected=False, occluded=True)
    assert hidden.state == BallVisibility.OCCLUDED
    assert hidden.missing_frames == 1
    recovered = tracker.update(detected=True, detection_confidence=.8)
    assert recovered.state == BallVisibility.VISIBLE
    assert recovered.missing_frames == 0


def test_monocular_ballistic_fit_recovers_synthetic_arc():
    camera = np.array([5.0, -12.0, 7.0])
    p0 = np.array([2.0, 3.0, 1.2])
    v0 = np.array([4.0, 10.0, 5.5])
    gravity = np.array([0.0, 0.0, -9.81])
    observations = []
    for frame, t in enumerate(np.linspace(0.0, 0.8, 17)):
        point = p0 + v0 * t + .5 * gravity * t * t
        ray = point - camera
        ray /= np.linalg.norm(ray)
        observations.append(RayObservation(
            timestamp=float(t), camera_origin=tuple(camera), ray_direction=tuple(ray),
            confidence=1.0, frame_number=frame,
        ))
    fit = MonocularTrajectoryEstimator().fit(observations)
    assert fit is not None
    assert fit.reliable
    assert fit.median_ray_error_m < 1e-6
    assert np.allclose(fit.initial_position, p0, atol=1e-5)
    assert np.allclose(fit.initial_velocity, v0, atol=1e-5)


def test_monocular_fit_requires_temporal_baseline():
    estimator = MonocularTrajectoryEstimator()
    observations = [
        RayObservation(0, (0, 0, 1), (0, 1, 0)),
        RayObservation(.1, (0, 0, 1), (0, 1, 0)),
    ]
    assert estimator.fit(observations) is None



def _synthetic_arc_observations(p0, v0, duration=0.8, n=17):
    camera = np.array([5.0, -12.0, 7.0])
    gravity = np.array([0.0, 0.0, -9.81])
    observations = []
    for frame, t in enumerate(np.linspace(0.0, duration, n)):
        point = np.asarray(p0) + np.asarray(v0) * t + .5 * gravity * t * t
        ray = point - camera
        ray /= np.linalg.norm(ray)
        observations.append(RayObservation(
            timestamp=float(t), camera_origin=tuple(camera), ray_direction=tuple(ray),
            confidence=1.0, frame_number=frame,
        ))
    return observations


def test_monocular_fit_off_court_is_not_reliable():
    """A perfect ballistic arc far outside the court must not be trusted.

    Audit on real footage showed 'reliable' fits at court y=-8 (8 m behind
    the baseline) — wrong-object tracks that were geometrically consistent.
    """
    fit = MonocularTrajectoryEstimator().fit(
        _synthetic_arc_observations(p0=[5.0, -8.0, 1.0], v0=[0.5, 0.5, 5.0]))
    assert fit is not None
    assert not fit.reliable


def test_monocular_fit_unrealistic_height_is_not_reliable():
    fit = MonocularTrajectoryEstimator().fit(
        _synthetic_arc_observations(p0=[5.0, 10.0, 1.0], v0=[0.0, 0.5, 18.0], duration=1.2))
    assert fit is not None
    assert not fit.reliable


def test_monocular_fit_on_court_arc_still_reliable():
    fit = MonocularTrajectoryEstimator().fit(
        _synthetic_arc_observations(p0=[2.0, 3.0, 1.2], v0=[4.0, 10.0, 5.5]))
    assert fit is not None
    assert fit.reliable
