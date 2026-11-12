from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from threading import RLock

from app.classification import Prediction


@dataclass
class WasteItem:
    key: str
    material: str
    detection_confidence: float
    category: str
    reason: str
    classification_confidence: float | None = None
    inference_seconds: float | None = None
    classification_source: str | None = None
    classification_state: str = "not_requested"


class SessionCounts:
    def __init__(self) -> None:
        self._lock = RLock()
        self._items: dict[str, WasteItem] = {}
        self._votes: dict[str, deque[tuple[str, float]]] = {}
        self._epoch = 0

    def observe(
        self,
        key: str | None,
        material: str,
        confidence: float,
        crop: bytes | None,
        *,
        confirm_immediately: bool = False,
    ) -> tuple[WasteItem | None, bool]:
        if key is None:
            return None, False
        with self._lock:
            votes = self._votes.setdefault(key, deque(maxlen=5))
            votes.append((material, confidence))
            if not confirm_immediately and len(votes) < 3:
                return None, False
            scores: Counter[str] = Counter()
            for name, score in votes:
                scores[name] += score
            confirmed_material, _ = scores.most_common(1)[0]
            if key not in self._items:
                category = (
                    "PLASTIC_MANUAL_REVIEW"
                    if confirmed_material == "Plastic"
                    else confirmed_material.upper()
                )
                item = WasteItem(
                    key,
                    confirmed_material,
                    confidence,
                    category,
                    "Awaiting local Qwen analysis."
                    if confirmed_material == "Plastic"
                    else f"YOLO detected {confirmed_material}.",
                )
                self._items[key] = item
            item = self._items[key]
            item.detection_confidence = max(item.detection_confidence, confidence)
            if item.material != "Plastic" or crop is None:
                return item, False
            if item.classification_state == "not_requested":
                item.classification_state = "pending"
                item.category = "PLASTIC_MANUAL_REVIEW"
                item.reason = "Awaiting local Qwen analysis."
                return item, True
            return item, False

    def apply_prediction(
        self,
        key: str,
        prediction: Prediction | None,
        error_message: str | None = None,
    ) -> bool:
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return False
            item.inference_seconds = prediction.inference_seconds if prediction else None
            item.classification_confidence = prediction.confidence if prediction else None
            item.classification_source = (
                "Custom" if prediction and prediction.source == "custom_model"
                else "Qwen" if prediction
                else None
            )
            if error_message:
                item.classification_state = "failed"
                item.category = "PLASTIC_MANUAL_REVIEW"
                item.reason = error_message
            elif prediction is not None:
                item.reason = prediction.reason
                if prediction.label == "uncertain":
                    item.classification_state = "uncertain"
                    item.category = "PLASTIC_MANUAL_REVIEW"
                elif prediction.label in {"recyclable", "non_recyclable"}:
                    item.classification_state = "classified"
                    item.category = f"PLASTIC_{prediction.label.upper()}"
                else:
                    item.classification_state = "failed"
                    item.category = "PLASTIC_MANUAL_REVIEW"
                    item.reason = "Unsupported classifier result; manual review required."
            return True

    def snapshot(self) -> dict:
        with self._lock:
            counts = {
                "total": len(self._items),
                "plastic_total": 0,
                "plastic_recyclable": 0,
                "plastic_non_recyclable": 0,
                "plastic_manual_review": 0,
                "metal_total": 0,
                "organic_total": 0,
            }
            for item in self._items.values():
                if item.material == "Plastic":
                    counts["plastic_total"] += 1
                    if item.category == "PLASTIC_RECYCLABLE":
                        counts["plastic_recyclable"] += 1
                    elif item.category == "PLASTIC_NON_RECYCLABLE":
                        counts["plastic_non_recyclable"] += 1
                    else:
                        counts["plastic_manual_review"] += 1
                elif item.material == "Metal":
                    counts["metal_total"] += 1
                elif item.material == "Organic":
                    counts["organic_total"] += 1
            return {
                **counts,
                "items": [self._as_dict(item) for item in self._items.values()],
                "epoch": self._epoch,
            }

    def item(self, key: str) -> dict | None:
        with self._lock:
            item = self._items.get(key)
            return self._as_dict(item) if item else None

    def reset(self) -> int:
        with self._lock:
            self._items.clear()
            self._votes.clear()
            self._epoch += 1
            return self._epoch

    @property
    def epoch(self) -> int:
        with self._lock:
            return self._epoch

    @staticmethod
    def _as_dict(item: WasteItem) -> dict:
        return {
            "track_id": item.key,
            "object_name": item.material,
            "material": item.material,
            "category": item.category,
            "detection_confidence": item.detection_confidence,
            "classification_confidence": item.classification_confidence,
            "classification_inference_seconds": item.inference_seconds,
            "classification_source": item.classification_source,
            "reason": item.reason,
            "classification_state": item.classification_state,
        }
