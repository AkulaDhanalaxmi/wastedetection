"""Wraps your EXISTING Ollama/Qwen plastic classifier so it can be the fallback. Nothing here replaces it.

You pass in:
  classify_fn(crop_bgr) -> str | dict   your existing function (called ONLY when needed -> Qwen is never
                                         loaded while the custom model works)
  ram_ok_fn() -> bool                   your existing RAM safety check (optional but recommended)
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from plastic_classifier import PlasticResult, _uncertain, normalize_label
from plastic_common import DISPLAY, UNCERTAIN


class QwenFallback:
    def __init__(self, classify_fn: Callable[[Any], Any], ram_ok_fn: Optional[Callable[[], bool]] = None):
        self.classify_fn, self.ram_ok_fn = classify_fn, ram_ok_fn

    def classify(self, crop) -> PlasticResult:
        try:
            if self.ram_ok_fn is not None and not self.ram_ok_fn():
                return _uncertain("Qwen skipped: RAM safety check failed", "qwen_blocked_low_ram", source="qwen_fallback")
            out = self.classify_fn(crop)
        except Exception as e:
            return _uncertain(f"Qwen fallback failed: {e}", "qwen_error", source="qwen_fallback")
        conf, reason = None, ""
        if isinstance(out, dict):
            conf = out.get("confidence")
            reason = str(out.get("reason", ""))
            out = out.get("label") or out.get("classification") or out.get("result")
        label = normalize_label(out)
        return PlasticResult(DISPLAY[label], conf if isinstance(conf, (int, float)) else None, "qwen_fallback",
                             reason or "Qwen fallback classification", None, {}, label == UNCERTAIN)
