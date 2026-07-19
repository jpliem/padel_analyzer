import numpy as np

from cv.audio_events import AudioImpulseDetector
from cv.player_reid import PlayerReIdentifier
from logic.contact_fusion import ContactEvidence, fuse_contact
from logic.review_ledger import ReviewLedger, ReviewStatus
from models.types import PointReason


def test_reid_requires_similarity_and_separation():
    reid = PlayerReIdentifier(similarity_threshold=.7, margin_threshold=.1)
    reid.register("P1", [1, 0, 0], team_id=1)
    reid.register("P2", [0, 1, 0], team_id=1)
    assert reid.match([.99, .01, 0], allowed_team=1).player_id == "P1"
    assert not reid.match([.7, .7, 0], allowed_team=1).confident


def test_audio_impulse_detects_transient():
    signal = np.zeros(4000)
    signal[2000:2005] = 1.0
    events = AudioImpulseDetector(1000).detect(signal)
    assert len(events) == 1
    assert 1.9 < events[0].timestamp < 2.1


def test_contact_needs_independent_evidence():
    weak = fuse_contact(ContactEvidence(1, "P1", racket_proximity_confidence=1))
    assert weak.requires_review
    strong = fuse_contact(ContactEvidence(1, "P1", audio_confidence=1,
                          direction_change_confidence=1, racket_proximity_confidence=1))
    assert strong.contact_type == "racket_hit"
    assert not strong.requires_review


def test_low_confidence_point_waits_for_review():
    ledger = ReviewLedger()
    record = ledger.propose(100, 1, PointReason.OUT, .6, "vision")
    assert record.status == ReviewStatus.PROPOSED
    assert ledger.replay().get_score_display()["score"] == "0 - 0"
    ledger.resolve(record.id, confirmed=True)
    assert ledger.replay().get_score_display()["score"] == "15 - 0"


def test_correction_replays_score_instead_of_mutating_it():
    ledger = ReviewLedger()
    record = ledger.propose(100, 1, PointReason.OUT, .95, "fusion")
    ledger.correct(record.id, 2, PointReason.MANUAL)
    assert ledger.replay().get_score_display()["score"] == "0 - 15"
    assert ledger.records[0].status == ReviewStatus.SUPERSEDED
