from __future__ import annotations

import base64
import io
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request

from PIL import Image

from config.settings import (
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MAX_IMAGE_SIDE,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT_SECONDS,
    OLLAMA_URL,
)


VALID_LABELS = {"recyclable", "non_recyclable", "uncertain"}
PROMPT = """Judge only visible recyclability cues in this plastic crop.
Do not use metadata, name the object, or infer local recycling rules.
Use uncertain if evidence about rigidity, contamination, or material is ambiguous.
Use recyclable only for clearly clean rigid plastic; non_recyclable only for clearly
flexible film/wrapper, visible contamination, or mixed materials.
Return only JSON: {"predicted_label":"recyclable|non_recyclable|uncertain",
"reason":"brief visible cues, under 12 words","confidence":0.0}
Confidence is self-reported, not measured accuracy."""


class ClassificationError(RuntimeError):
    """Ollama did not return a valid prediction."""


@dataclass(frozen=True)
class Prediction:
    label: str
    reason: str
    confidence: float | None
    inference_seconds: float


class OllamaPlasticClassifier:
    def __init__(
        self,
        *,
        base_url: str = OLLAMA_URL,
        model: str = OLLAMA_MODEL,
        timeout: float = OLLAMA_TIMEOUT_SECONDS,
        max_image_side: int = OLLAMA_MAX_IMAGE_SIDE,
        keep_alive: str = OLLAMA_KEEP_ALIVE,
    ) -> None:
        if timeout <= 0 or max_image_side <= 0:
            raise ValueError("Ollama timeout and maximum image side must be positive.")
        self.endpoint = base_url.rstrip("/") + "/api/generate"
        self.model = model
        self.timeout = timeout
        self.max_image_side = max_image_side
        self.keep_alive = keep_alive

    def classify(self, crop: bytes) -> Prediction:
        encoded = self._encode_crop(crop)
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": PROMPT,
            "images": [encoded],
            "format": "json",
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0, "num_ctx": 2048, "num_predict": 64},
        }
        http_request = request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with request.urlopen(http_request, timeout=self.timeout) as response:
                raw = response.read()
        except (error.URLError, TimeoutError, OSError) as exc:
            raise ClassificationError(f"Ollama request failed: {exc}") from exc
        result = self._parse(raw)
        return Prediction(
            result.label,
            result.reason,
            result.confidence,
            time.perf_counter() - started,
        )

    def _encode_crop(self, crop: bytes) -> str:
        try:
            with Image.open(io.BytesIO(crop)) as source:
                image = source.convert("RGB")
                if max(image.size) > self.max_image_side:
                    scale = self.max_image_side / max(image.size)
                    image = image.resize(
                        (
                            max(1, round(image.width * scale)),
                            max(1, round(image.height * scale)),
                        ),
                        Image.Resampling.LANCZOS,
                    )
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=90, optimize=True)
        except (OSError, ValueError) as exc:
            raise ClassificationError(f"Could not prepare crop: {exc}") from exc
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    @staticmethod
    def _parse(raw: bytes | str) -> Prediction:
        try:
            envelope = json.loads(raw)
            content = envelope["response"] if isinstance(envelope, dict) else envelope
            data = json.loads(content) if isinstance(content, str) else content
            label = data["predicted_label"]
            reason = data["reason"]
            confidence = data.get("confidence")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ClassificationError("Ollama returned invalid JSON.") from exc
        if (
            not isinstance(label, str)
            or label not in VALID_LABELS
            or not isinstance(reason, str)
            or not reason.strip()
        ):
            raise ClassificationError("Ollama returned an invalid prediction.")
        if confidence is not None:
            if (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not 0 <= confidence <= 1
            ):
                raise ClassificationError("Ollama returned invalid confidence.")
            confidence = float(confidence)
        return Prediction(label, reason.strip(), confidence, 0.0)
