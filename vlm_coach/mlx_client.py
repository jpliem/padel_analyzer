from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Iterable, Type

from .ollama_client import OllamaError
from .schemas import model_schema, model_validate


MODEL_REPOS = {
    "qwen3.5:0.8b": "mlx-community/Qwen3.5-0.8B-4bit",
    "qwen3.5:2b": "mlx-community/Qwen3.5-2B-4bit",
    "qwen3.5:4b": "mlx-community/Qwen3.5-4B-4bit",
}


def _default_max_tokens(model: str) -> int:
    alias = model.lower()
    if "4b" in alias:
        return 3200
    if "2b" in alias:
        return 2600
    return 1400


class MlxVlmClient:
    """In-process Qwen VLM provider optimized for Apple Silicon."""

    provider_name = "mlx"

    def __init__(self):
        self._loaded_repo = None
        self._model = None
        self._processor = None
        self._config = None
        self._lock = threading.RLock()

    def health(self) -> dict:
        try:
            import importlib.util
            available = importlib.util.find_spec("mlx_vlm") is not None
        except Exception:
            available = False
        return {
            "available": available,
            "provider": self.provider_name,
            "models": list(MODEL_REPOS),
            "loaded_model": self._loaded_repo,
            "error": None if available else "mlx-vlm is not installed",
        }

    def _load(self, alias: str):
        repo = MODEL_REPOS.get(alias, alias)
        if self._loaded_repo == repo:
            return
        try:
            # Standard HTTPS is slower in theory but has proven much more
            # reliable than Xet on this target Mac/network. Downloads resume
            # from the Hugging Face cache across app restarts.
            os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
            from mlx_vlm import load
            from mlx_vlm.utils import load_config
            self._model, self._processor = load(repo)
            self._config = load_config(repo)
            self._loaded_repo = repo
        except Exception as exc:
            raise OllamaError(f"MLX could not load '{repo}': {exc}") from exc

    def structured(
        self,
        model: str,
        prompt: str,
        output_type: Type,
        images: Iterable[Path] = (),
        videos: Iterable[Path] = (),
        video_fps: float = 2.0,
        max_tokens: int | None = None,
    ):
        image_paths = [str(Path(path)) for path in images]
        video_paths = [str(Path(path)) for path in videos]
        if image_paths and video_paths:
            raise ValueError("Use images or videos in one structured call, not both")
        with self._lock:
            self._load(model)
            try:
                from mlx_vlm import generate
                from mlx_vlm.prompt_utils import apply_chat_template
                from mlx_vlm.structured import build_json_schema_logits_processor

                formatted = apply_chat_template(
                    self._processor, self._config, prompt,
                    num_images=len(image_paths), add_generation_prompt=True,
                    enable_thinking=False,
                    **({"video": video_paths, "fps": video_fps} if video_paths else {}),
                )
                tokenizer = getattr(self._processor, "tokenizer", self._processor)
                constraint = build_json_schema_logits_processor(
                    tokenizer, model_schema(output_type)
                )
                result = generate(
                    self._model, self._processor, formatted,
                    image=image_paths or None,
                    video=video_paths or None,
                    fps=video_fps,
                    # Bigger aliases need more room to finish nested JSON.
                    # Let the caller override, otherwise pick a model-aware
                    # default that keeps the object from being cut off.
                    max_tokens=max_tokens or _default_max_tokens(model),
                    # Qwen recommends moderate sampling and a presence penalty
                    # for non-thinking vision tasks. This also prevents long,
                    # repetitive strings from exhausting the JSON token budget.
                    temperature=0.7, top_p=0.8, top_k=20,
                    presence_penalty=1.5, repetition_penalty=1.0,
                    enable_thinking=False, verbose=False,
                    logits_processors=[constraint],
                )
                try:
                    payload = json.loads(result.text)
                except json.JSONDecodeError as exc:
                    preview = result.text[:240].replace("\n", "\\n")
                    raise ValueError(
                        f"model returned incomplete JSON ({len(result.text)} chars): "
                        f"{preview!r}"
                    ) from exc
                return model_validate(output_type, payload)
            except Exception as exc:
                raise OllamaError(f"MLX structured generation failed: {exc}") from exc
