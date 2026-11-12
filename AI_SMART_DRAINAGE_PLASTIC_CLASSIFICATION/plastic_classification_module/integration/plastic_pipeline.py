"""YOLO plastic detection -> crop -> custom classifier -> (fallback: existing Qwen) -> dashboard-ready dict."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plastic_classifier import INPUT_ERRORS, MODEL_UNAVAILABLE, PlasticClassifier, PlasticResult, _uncertain  # noqa: E402
from plastic_common import UNCERTAIN  # noqa: E402


def crop_bbox(frame: np.ndarray, bbox_xyxy, pad_fraction: float = 0.10) -> Optional[np.ndarray]:
    """Padded, clamped crop. Returns None for empty/invalid boxes or frames."""
    if frame is None or not isinstance(frame, np.ndarray) or frame.ndim < 2 or frame.size == 0:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    except Exception:
        return None
    if not all(np.isfinite([x1, y1, x2, y2])):
        return None
    h, w = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return None
    x1, x2 = int(max(0, x1 - bw * pad_fraction)), int(min(w, x2 + bw * pad_fraction))
    y1, y2 = int(max(0, y1 - bh * pad_fraction)), int(min(h, y2 + bh * pad_fraction))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


class PlasticRecyclabilityPipeline:
    def __init__(self, classifier: Optional[PlasticClassifier] = None, qwen=None,
                 plastic_names: Iterable[str] = ("plastic", "Plastic"), pad_fraction: float = 0.10,
                 qwen_on_low_confidence: bool = False):
        """qwen: integration.qwen_adapter.QwenFallback (or None). qwen_on_low_confidence=False keeps
        low-confidence custom-model results as Manual Review instead of waking up Qwen (saves RAM)."""
        self.classifier = classifier or PlasticClassifier()
        self.qwen, self.plastic_names = qwen, set(plastic_names)
        self.pad_fraction, self.qwen_on_low = pad_fraction, qwen_on_low_confidence

    def classify_crop(self, crop: Any) -> PlasticResult:
        res = self.classifier.classify(crop)
        if res.error_code in INPUT_ERRORS:            # nothing usable to classify: do not wake Qwen
            return res
        if res.error_code == MODEL_UNAVAILABLE or (res.error_code == "low_confidence" and self.qwen_on_low):
            if self.qwen is None:
                return _uncertain(f"{res.reason}; no Qwen fallback configured", res.error_code or "no_fallback", source="none")
            fb = self.qwen.classify(crop)
            if res.error_code == "low_confidence" and fb.error_code:   # keep custom result if Qwen failed
                return res
            return fb
        return res

    def annotate_detections(self, frame: np.ndarray, detections: list[dict]) -> list[dict]:
        """detections: [{'class_name': 'plastic', 'bbox': [x1,y1,x2,y2], 'confidence': 0.9}, ...].
        Adds 'recyclability' (dict) to plastic detections only; others are returned untouched."""
        for det in detections:
            if det.get("class_name") in self.plastic_names:
                crop = crop_bbox(frame, det.get("bbox"), self.pad_fraction)
                det["recyclability"] = self.classify_crop(crop).to_dict()
        return detections
