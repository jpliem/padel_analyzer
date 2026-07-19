"""Appearance-gallery ReID for reconnecting player identities after track loss."""

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ReIdMatch:
    player_id: Optional[str]
    similarity: float
    margin: float
    confident: bool


class PlayerReIdentifier:
    def __init__(self, similarity_threshold: float = 0.72,
                 margin_threshold: float = 0.08, gallery_size: int = 30):
        self.similarity_threshold = similarity_threshold
        self.margin_threshold = margin_threshold
        self.gallery_size = gallery_size
        self._gallery: Dict[str, list] = {}
        self._teams: Dict[str, int] = {}

    @staticmethod
    def _normalise(embedding: Iterable[float]) -> np.ndarray:
        vector = np.asarray(embedding, dtype=float)
        norm = np.linalg.norm(vector)
        if vector.ndim != 1 or norm <= 1e-12:
            raise ValueError("embedding must be a non-zero 1D vector")
        return vector / norm

    def register(self, player_id: str, embedding: Iterable[float],
                 team_id: Optional[int] = None) -> None:
        vector = self._normalise(embedding)
        gallery = self._gallery.setdefault(player_id, [])
        gallery.append(vector)
        del gallery[:-self.gallery_size]
        if team_id is not None:
            self._teams[player_id] = team_id

    def match(self, embedding: Iterable[float], allowed_team: Optional[int] = None,
              excluded_players: Iterable[str] = ()) -> ReIdMatch:
        query = self._normalise(embedding)
        excluded = set(excluded_players)
        candidates = []
        for player_id, gallery in self._gallery.items():
            if player_id in excluded:
                continue
            if allowed_team is not None and self._teams.get(player_id) != allowed_team:
                continue
            # Use the strongest recent view; pose/viewpoint variation makes a
            # single averaged shirt embedding less stable.
            similarity = max(float(np.dot(query, sample)) for sample in gallery)
            candidates.append((similarity, player_id))
        if not candidates:
            return ReIdMatch(None, 0.0, 0.0, False)
        candidates.sort(reverse=True)
        best_score, best_id = candidates[0]
        second = candidates[1][0] if len(candidates) > 1 else 0.0
        margin = best_score - second
        confident = (best_score >= self.similarity_threshold and
                     margin >= self.margin_threshold)
        return ReIdMatch(best_id if confident else None, best_score, margin, confident)


class OsnetAppearanceEncoder:
    """OSNet x0.25 ReID embeddings (MSMT17-pretrained, vendored model).

    512-d appearance vectors far more robust to pose/viewpoint change than the
    HSV baseline. ~0.6M params; runs on MPS or CPU. Weights expected at
    backend/models/osnet_x0_25_msmt17.pth (HuggingFace kaiyangzhou/osnet,
    MIT-licensed deep-person-reid).
    """

    INPUT_SIZE = (256, 128)  # (height, width), the OSNet training resolution
    _MEAN = (0.485, 0.456, 0.406)
    _STD = (0.229, 0.224, 0.225)

    def __init__(self, weights_path: Optional[str] = None, device: Optional[str] = None):
        import os

        import torch

        from .vendor.osnet import osnet_x0_25

        if weights_path is None:
            weights_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__)))), "models", "osnet_x0_25_msmt17.pth")
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self._torch = torch
        self._device = torch.device(device)
        self._model = osnet_x0_25(num_classes=1, pretrained=False)
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state)
        cleaned = {key.replace("module.", ""): value for key, value in state.items()
                   if not key.startswith(("classifier.", "module.classifier."))}
        self._model.load_state_dict(cleaned, strict=False)
        self._model.eval().to(self._device)

    def encode(self, frame: np.ndarray, bbox: Iterable[float]) -> Optional[np.ndarray]:
        import cv2

        x1, y1, x2, y2 = (int(round(v)) for v in bbox)
        h, w = frame.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 - x1 < 4 or y2 - y1 < 8:
            return None
        crop = frame[y1:y2, x1:x2]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = cv2.resize(crop, (self.INPUT_SIZE[1], self.INPUT_SIZE[0]))
        tensor = self._torch.from_numpy(crop).float().div_(255.0)
        tensor = tensor.sub_(self._torch.tensor(self._MEAN)).div_(
            self._torch.tensor(self._STD))
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self._device)
        with self._torch.inference_mode():
            embedding = self._model(tensor)
        vector = embedding.squeeze(0).cpu().numpy().astype(float)
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 1e-12 else None


class PlayerAppearanceEncoder:
    """Small deterministic HSV histogram baseline; replaceable by OSNet later."""

    def encode(self, frame: np.ndarray, bbox: Iterable[float]) -> Optional[np.ndarray]:
        import cv2

        x1, y1, x2, y2 = (int(round(v)) for v in bbox)
        h, w = frame.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 - x1 < 4 or y2 - y1 < 8:
            return None
        crop = frame[y1:y2, x1:x2]
        # Focus on torso/clothing; legs, court, and faces are less stable.
        crop = crop[int(crop.shape[0] * .18):int(crop.shape[0] * .72)]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).reshape(-1)
        norm = np.linalg.norm(hist)
        return hist / norm if norm > 1e-12 else None
