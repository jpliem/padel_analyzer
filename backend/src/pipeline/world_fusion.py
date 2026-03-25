"""WorldFusion — merges per-camera CameraObservations into a single WorldState."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from models.types import BallPosition, CameraObservation, PlayerPosition, WorldState
from models.court_model import PadelCourtModel

PLAYER_DEDUP_DISTANCE = 2.0  # metres — two detections closer than this are the same player


class WorldFusion:
    """Fuse N per-camera observations into one WorldState per frame."""

    def __init__(self, court_model: PadelCourtModel) -> None:
        self._court = court_model
        self._prev_ball: Optional[BallPosition] = None
        self._prev_time: float = 0.0
        self._has_prev: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fuse(
        self,
        observations: List[CameraObservation],
        camera_weights: Dict[str, float],
    ) -> WorldState:
        """Merge a list of per-camera observations into a WorldState.

        Args:
            observations: One CameraObservation per active camera.
            camera_weights: Per-camera reliability weight (camera_id → weight).
                            Cameras with weight == 0 are ignored.
        """
        # Filter to cameras with positive weight
        valid_obs = [o for o in observations if camera_weights.get(o.camera_id, 0.0) > 0.0]

        # Split into observations with / without a ball detection
        ball_obs = [o for o in valid_obs if o.ball_court is not None]

        ball, cameras = self._fuse_ball(ball_obs, camera_weights)

        # Velocity
        velocity: Optional[Tuple[float, float, float]] = None
        if ball is not None and self._prev_ball is not None and self._has_prev:
            dt = ball.timestamp - self._prev_time
            if dt > 0.0:
                vx = (ball.x - self._prev_ball.x) / dt
                vy = (ball.y - self._prev_ball.y) / dt
                vz = (ball.z - self._prev_ball.z) / dt
                velocity = (vx, vy, vz)

        players = self._fuse_players(valid_obs, camera_weights)

        # Determine frame metadata from valid observations (or fall back to defaults)
        timestamp = 0.0
        frame_number = 0
        if valid_obs:
            timestamp = valid_obs[0].timestamp
            frame_number = valid_obs[0].frame_number

        # Update state for next frame
        if ball is not None:
            self._prev_ball = ball
            self._prev_time = ball.timestamp
            self._has_prev = True

        return WorldState(
            ball=ball,
            ball_velocity=velocity,
            players=players,
            contributing_cameras=cameras,
            timestamp=timestamp,
            frame_number=frame_number,
        )

    def reset(self) -> None:
        """Reset internal state (e.g. at the start of a new point)."""
        self._prev_ball = None
        self._prev_time = 0.0
        self._has_prev = False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fuse_ball(
        self,
        ball_obs: List[CameraObservation],
        camera_weights: Dict[str, float],
    ) -> Tuple[Optional[BallPosition], List[str]]:
        """Compute a fused BallPosition from cameras that detected the ball.

        Returns:
            (ball, contributing_camera_ids)
            If no cameras saw the ball this frame, returns (prev_ball, []).
        """
        if len(ball_obs) == 0:
            return self._prev_ball, []

        if len(ball_obs) == 1:
            obs = ball_obs[0]
            return obs.ball_court, [obs.camera_id]

        # Weighted average: weight = camera_weight * confidence
        total_weight = 0.0
        wx = wy = wz = wspeed = 0.0
        cameras: List[str] = []

        for obs in ball_obs:
            w = camera_weights.get(obs.camera_id, 0.0) * obs.confidence
            if w <= 0.0:
                continue
            bp = obs.ball_court
            wx += w * bp.x
            wy += w * bp.y
            wz += w * bp.z
            wspeed += w * bp.speed
            total_weight += w
            cameras.append(obs.camera_id)

        if total_weight == 0.0:
            return self._prev_ball, []

        # Use timestamp from first observation (all cameras observe same frame)
        ts = ball_obs[0].ball_court.timestamp

        fused = BallPosition(
            x=wx / total_weight,
            y=wy / total_weight,
            z=wz / total_weight,
            speed=wspeed / total_weight,
            timestamp=ts,
        )
        return fused, cameras

    def _fuse_players(
        self,
        observations: List[CameraObservation],
        camera_weights: Dict[str, float],
    ) -> List[PlayerPosition]:
        """Collect all player detections and deduplicate by proximity.

        Two detections within PLAYER_DEDUP_DISTANCE metres are merged via a
        confidence-weighted average of their court positions.
        """
        # Collect all raw detections as (x, y, confidence) tuples
        raw: List[Tuple[float, float, float, str]] = []  # (x, y, conf, camera_id)

        for obs in observations:
            cam_weight = camera_weights.get(obs.camera_id, 0.0)
            if cam_weight <= 0.0:
                continue
            for det in obs.player_detections:
                x = det.get("court_x", 0.0)
                y = det.get("court_y", 0.0)
                conf = det.get("confidence", 0.0)
                raw.append((x, y, conf, obs.camera_id))

        if not raw:
            return []

        # Greedy deduplication: merge detections within PLAYER_DEDUP_DISTANCE
        merged: List[dict] = []  # list of {x, y, total_weight}

        for x, y, conf, _ in raw:
            # Find closest existing cluster
            closest_idx = -1
            closest_dist = float("inf")
            for i, cluster in enumerate(merged):
                dx = cluster["x"] - x
                dy = cluster["y"] - y
                dist = math.hypot(dx, dy)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_idx = i

            if closest_idx >= 0 and closest_dist < PLAYER_DEDUP_DISTANCE:
                # Merge into existing cluster via weighted average
                c = merged[closest_idx]
                new_w = c["total_weight"] + conf
                c["x"] = (c["x"] * c["total_weight"] + x * conf) / new_w
                c["y"] = (c["y"] * c["total_weight"] + y * conf) / new_w
                c["total_weight"] = new_w
            else:
                merged.append({"x": x, "y": y, "total_weight": conf})

        # Convert to PlayerPosition (generic ids since we don't track identity here)
        players: List[PlayerPosition] = []
        for i, c in enumerate(merged):
            players.append(PlayerPosition(
                player_id=f"P{i + 1}",
                x=c["x"],
                y=c["y"],
            ))

        return players
