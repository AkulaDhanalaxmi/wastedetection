from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import cv2
import numpy as np

from config.settings import (
    DETECTION_CONFIDENCE,
    MODEL_PATH,
    OLLAMA_MAX_IMAGE_SIDE,
    TRACKER_CONFIG,
    YOLO_IMAGE_SIZE,
)


EXPECTED_MATERIALS = {"plastic": "Plastic", "metal": "Metal", "organic": "Organic"}


@dataclass(frozen=True)
class RawDetection:
    track_id: str | None
    material: str
    confidence: float
    box: tuple[int, int, int, int]
    crop_jpeg: bytes | None


class WasteDetector:
    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        *,
        model_factory=None,
        confidence: float = DETECTION_CONFIDENCE,
        image_size: int = YOLO_IMAGE_SIZE,
    ) -> None:
        self.model_path = Path(model_path)
        self._model_factory = model_factory
        self.confidence = confidence
        self.image_size = image_size
        self._model = None
        self._class_names: dict[int, str] = {}
        self._lock = Lock()

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            if not self.model_path.is_file():
                raise FileNotFoundError(f"YOLO checkpoint is missing: {self.model_path}")
            factory = self._model_factory
            if factory is None:
                from ultralytics import YOLO

                factory = YOLO
            model = factory(str(self.model_path))
            names = model.names
            normalized = {
                int(class_id): EXPECTED_MATERIALS[str(name).strip().lower()]
                for class_id, name in names.items()
                if str(name).strip().lower() in EXPECTED_MATERIALS
            }
            if set(normalized.values()) != set(EXPECTED_MATERIALS.values()):
                raise ValueError(
                    "Checkpoint must contain Plastic, Metal, and Organic classes; "
                    f"found {names!r}."
                )
            self._class_names = normalized
            self._model = model

    def reset_tracker(self) -> None:
        with self._lock:
            if self._model is None:
                return
            predictor = getattr(self._model, "predictor", None)
            trackers = getattr(predictor, "trackers", [])
            for tracker in trackers:
                tracker.reset()

    def detect(self, frame: np.ndarray, *, tracking: bool) -> list[RawDetection]:
        self.load()
        with self._lock:
            if tracking:
                results = self._model.track(
                    source=frame,
                    persist=True,
                    tracker=TRACKER_CONFIG,
                    conf=self.confidence,
                    classes=list(self._class_names),
                    imgsz=self.image_size,
                    device="cpu",
                    verbose=False,
                )
            else:
                results = self._model.predict(
                    source=frame,
                    conf=self.confidence,
                    classes=list(self._class_names),
                    imgsz=self.image_size,
                    device="cpu",
                    verbose=False,
                )
        result = results[0]
        height, width = frame.shape[:2]
        detections: list[RawDetection] = []
        if result.boxes is None:
            return detections
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            material = self._class_names.get(class_id)
            if material is None:
                continue
            coordinates = box.xyxy[0].tolist()
            x1 = max(0, min(width - 1, int(coordinates[0])))
            y1 = max(0, min(height - 1, int(coordinates[1])))
            x2 = max(x1 + 1, min(width, int(coordinates[2])))
            y2 = max(y1 + 1, min(height, int(coordinates[3])))
            track = box.id[0].item() if tracking and box.id is not None else None
            crop_jpeg = None
            if material == "Plastic" and x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                height, width = crop.shape[:2]
                if max(height, width) > OLLAMA_MAX_IMAGE_SIDE:
                    scale = OLLAMA_MAX_IMAGE_SIDE / max(height, width)
                    crop = cv2.resize(
                        crop,
                        (
                            max(1, round(width * scale)),
                            max(1, round(height * scale)),
                        ),
                        interpolation=cv2.INTER_AREA,
                    )
                success, encoded = cv2.imencode(
                    ".jpg",
                    crop,
                    [cv2.IMWRITE_JPEG_QUALITY, 92],
                )
                if success:
                    crop_jpeg = encoded.tobytes()
                del crop, encoded
            detections.append(
                RawDetection(
                    str(int(track)) if track is not None else None,
                    material,
                    float(box.conf[0].item()),
                    (x1, y1, x2, y2),
                    crop_jpeg,
                )
            )
        return detections
