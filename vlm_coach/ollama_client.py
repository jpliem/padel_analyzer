from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Iterable, Type

import httpx

from .schemas import model_schema, model_validate


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    provider_name = "ollama"

    def __init__(self, base_url: str = "http://127.0.0.1:11434", timeout: float = 180.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict:
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=3.0)
            response.raise_for_status()
            payload = response.json()
            return {"available": True, "provider": self.provider_name,
                    "models": [item.get("name") for item in payload.get("models", [])]}
        except Exception as exc:
            return {"available": False, "provider": self.provider_name,
                    "models": [], "error": str(exc)}

    def structured(self, model: str, prompt: str, output_type: Type,
                   images: Iterable[Path] = ()):
        encoded = []
        for image in images:
            encoded.append(base64.b64encode(Path(image).read_bytes()).decode("ascii"))
        current_prompt = prompt
        last_error = None
        raw = ""
        for attempt in range(2):
            payload = {
                "model": model,
                "stream": False,
                "messages": [{"role": "user", "content": current_prompt, "images": encoded}],
                "format": model_schema(output_type),
                # num_predict must cover thinking tokens too: some models (e.g.
                # qwen3-vl) ignore think=false and burn the budget on <think>,
                # leaving content empty.
                "options": {"temperature": 0.1, "num_ctx": 8192, "num_predict": 2048},
                "think": False,
            }
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(f"{self.base_url}/api/chat", json=payload)
                    response.raise_for_status()
            except httpx.ConnectError as exc:
                raise OllamaError("Ollama is not running. Start it with: ollama serve") from exc
            except httpx.HTTPStatusError as exc:
                detail = exc.response.text[:500]
                raise OllamaError(f"Ollama rejected the request: {detail}") from exc
            data = response.json()
            raw = data.get("message", {}).get("content", "")
            try:
                return model_validate(output_type, json.loads(raw))
            except Exception as exc:
                last_error = exc
                current_prompt = (
                    prompt + "\nYour previous response did not validate. Return one complete JSON "
                    "object matching the schema exactly; do not add markdown or commentary."
                )
        raise OllamaError(f"Model returned invalid structured JSON twice: {raw[:500]}") from last_error
