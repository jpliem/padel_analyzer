"""Auditable review ledger with deterministic score replay and correction."""

from dataclasses import dataclass, replace
from enum import Enum
from typing import Dict, List, Optional
from uuid import uuid4

from logic.scoring_engine import PadelScoringEngine
from models.types import PointReason, ServerInfo, TeamId


class ReviewStatus(str, Enum):
    PROPOSED = "proposed"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


@dataclass(frozen=True)
class PointRecord:
    id: str
    frame_number: int
    winner_team: Optional[int]
    reason: PointReason
    confidence: float
    source: str
    status: ReviewStatus
    supersedes: Optional[str] = None
    note: str = ""


class ReviewLedger:
    def __init__(self, *, golden_point: bool = True, sets_to_win: int = 2,
                 first_server: Optional[ServerInfo] = None,
                 team_players: Optional[Dict[TeamId, List[str]]] = None):
        self._scoring_args = dict(
            golden_point=golden_point, sets_to_win=sets_to_win,
            first_server=first_server, team_players=team_players,
        )
        self.records: List[PointRecord] = []

    def propose(self, frame_number: int, winner_team: Optional[int], reason: PointReason,
                confidence: float, source: str, auto_confirm_threshold: float = 0.90) -> PointRecord:
        status = (ReviewStatus.CONFIRMED if winner_team is not None and
                  confidence >= auto_confirm_threshold else ReviewStatus.PROPOSED)
        record = PointRecord(str(uuid4()), frame_number, winner_team, reason,
                             confidence, source, status)
        self.records.append(record)
        return record

    def resolve(self, record_id: str, *, confirmed: bool, winner_team: Optional[int] = None,
                note: str = "") -> PointRecord:
        index = self._index(record_id)
        old = self.records[index]
        winner = old.winner_team if winner_team is None else winner_team
        if confirmed and winner is None:
            raise ValueError("a confirmed point needs a winner")
        updated = replace(old, winner_team=winner, note=note,
                          status=ReviewStatus.CONFIRMED if confirmed else ReviewStatus.REJECTED)
        self.records[index] = updated
        return updated

    def correct(self, record_id: str, winner_team: int, reason: PointReason,
                note: str = "manual correction") -> PointRecord:
        index = self._index(record_id)
        old = self.records[index]
        self.records[index] = replace(old, status=ReviewStatus.SUPERSEDED)
        corrected = PointRecord(str(uuid4()), old.frame_number, winner_team, reason,
                                1.0, "manual", ReviewStatus.CONFIRMED,
                                supersedes=old.id, note=note)
        self.records.append(corrected)
        return corrected

    def replay(self) -> PadelScoringEngine:
        engine = PadelScoringEngine(**self._scoring_args)
        for record in sorted(self.records, key=lambda r: (r.frame_number, self.records.index(r))):
            if record.status == ReviewStatus.CONFIRMED and record.winner_team is not None:
                engine.add_point(record.winner_team, record.reason)
        return engine

    def pending(self) -> List[PointRecord]:
        return [r for r in self.records if r.status == ReviewStatus.PROPOSED]

    def _index(self, record_id: str) -> int:
        for i, record in enumerate(self.records):
            if record.id == record_id:
                return i
        raise KeyError(record_id)

