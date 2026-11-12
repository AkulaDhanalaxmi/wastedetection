from __future__ import annotations

import gc
import importlib
import io
import sys
import time
from pathlib import Path

from PIL import Image

from app.classification import Prediction
from config.settings import CUSTOM_PLASTIC_MODEL_PATH, PLASTIC_MODULE_ROOT


class CustomPlasticClassifier:
    def __init__(
        self,
        *,
        module_root: Path = PLASTIC_MODULE_ROOT,
        weights_path: Path = CUSTOM_PLASTIC_MODEL_PATH,
    ) -> None:
        self.module_root = Path(module_root)
        self.weights_path = Path(weights_path)
        self._classifier = None

    def classify(self, crop: bytes) -> Prediction:
        started = time.perf_counter()
        if not crop:
            return Prediction(
                "uncertain",
                "Plastic crop is empty; manual review required.",
                None,
                time.perf_counter() - started,
                "none",
                "empty_crop",
            )
        try:
            with Image.open(io.BytesIO(crop)) as image:
                prepared = image.convert("RGB")
        except (OSError, ValueError) as exc:
            return Prediction(
                "uncertain",
                f"Plastic crop is invalid ({exc}); manual review required.",
                None,
                time.perf_counter() - started,
                "none",
                "invalid_image",
            )

        if not self.module_root.is_dir():
            self.release()
            return self._unavailable(
                f"Custom classifier module is missing: {self.module_root}",
                started,
            )
        if not self.weights_path.is_file():
            self.release()
            return self._unavailable(
                f"Custom classifier weights are missing: {self.weights_path}",
                started,
            )

        try:
            classifier = self._load_classifier()
            if not classifier.available:
                reason = classifier.unavailable_reason or "Custom classifier could not load its weights."
                self.release()
                return self._unavailable(reason, started)
            result = classifier.classify(prepared)
        except Exception as exc:
            self.release()
            return self._unavailable(
                f"Custom classifier failed ({type(exc).__name__}): {exc}",
                started,
            )

        labels = {
            "Recyclable": "recyclable",
            "Non-Recyclable": "non_recyclable",
            "Uncertain / Manual Review": "uncertain",
        }
        label = labels.get(result.label, "uncertain")
        error_code = result.error_code
        if result.label not in labels and error_code is None:
            error_code = "invalid_model_output"
        return Prediction(
            label,
            result.reason or "Custom classifier returned no explanation.",
            result.confidence,
            time.perf_counter() - started,
            "custom_model",
            error_code,
        )

    def release(self) -> None:
        self._classifier = None
        gc.collect()

    def _load_classifier(self):
        if self._classifier is None:
            module_path = str(self.module_root)
            if module_path not in sys.path:
                sys.path.insert(0, module_path)
            module = importlib.import_module("plastic_classifier")
            self._classifier = module.PlasticClassifier(
                weights=str(self.weights_path),
                config_path=str(self.module_root / "config" / "model_plastic.yaml"),
            )
        return self._classifier

    @staticmethod
    def _unavailable(reason: str, started: float) -> Prediction:
        return Prediction(
            "uncertain",
            reason,
            None,
            time.perf_counter() - started,
            "custom_model",
            "model_unavailable",
        )
