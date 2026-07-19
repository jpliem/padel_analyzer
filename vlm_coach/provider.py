from __future__ import annotations

import importlib.util
import os

from .mlx_client import MlxVlmClient
from .ollama_client import OllamaClient


def build_provider():
    requested = os.environ.get("VLM_COACH_PROVIDER", "auto").strip().lower()
    if requested == "ollama":
        return OllamaClient(os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"))
    if requested == "mlx":
        return MlxVlmClient()
    if requested != "auto":
        raise ValueError("VLM_COACH_PROVIDER must be auto, mlx, or ollama")
    if importlib.util.find_spec("mlx_vlm") is not None:
        return MlxVlmClient()
    return OllamaClient(os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"))

