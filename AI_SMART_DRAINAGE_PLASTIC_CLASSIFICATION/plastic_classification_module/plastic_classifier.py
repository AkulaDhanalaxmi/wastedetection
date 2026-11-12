"""Plastic recyclability classifier: crop -> Recyclable / Non-Recyclable / Uncertain-Manual Review.

* Runs a lightweight CNN (PyTorch .pt or ONNX .onnx) on the crop of a YOLO "plastic" detection.
* Returns Uncertain when the input is unusable OR the calibrated confidence is below threshold.
* If the weights file is missing the classifier reports `model_unavailable` (it never guesses);
  integration/plastic_pipeline.py then hands the crop to the existing Qwen classifier.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from plastic_common import (DISPLAY, NON_RECYCLABLE, RECYCLABLE, UNCERTAIN, load_yaml, resolve,
                            softmax, to_model_array)

MODEL_UNAVAILABLE = "model_unavailable"
INPUT_ERRORS = {"empty_crop", "invalid_image", "crop_too_small", "blurry_crop"}


@dataclass
class PlasticResult:
    label: str                      # "Recyclable" | "Non-Recyclable" | "Uncertain / Manual Review"
    confidence: Optional[float]     # calibrated top-class probability, None if no model decision
    source: str                     # "custom_model" | "qwen_fallback" | "none"
    reason: str = ""                # human-readable explanation
    error_code: Optional[str] = None  # model_unavailable / empty_crop / invalid_image / low_confidence / ...
    probabilities: dict = field(default_factory=dict)
    needs_manual_review: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_label(text: Any) -> str:
    """Map free text (e.g. Qwen output) to one of the three internal labels."""
    t = str(text or "").strip().lower().replace("_", " ").replace("-", " ")
    if not t or "uncertain" in t or "manual" in t or "unknown" in t or "unsure" in t:
        return UNCERTAIN
    if t.startswith("non") or "not recycl" in t or "unrecycl" in t or "non recycl" in t:
        return NON_RECYCLABLE
    if "recycl" in t:
        return RECYCLABLE
    return UNCERTAIN


def _uncertain(reason, code, conf=None, probs=None, source="custom_model") -> PlasticResult:
    return PlasticResult(DISPLAY[UNCERTAIN], conf, source, reason, code, probs or {}, True)


class TorchBackend:
    def __init__(self, path: Path, device: str = "cpu"):
        from plastic_models import load_checkpoint
        self.model, meta = load_checkpoint(path, device)
        self.device = device
        self.meta = meta

    def logits(self, batch: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            return self.model(torch.from_numpy(batch).to(self.device)).cpu().numpy()


class OnnxBackend:
    def __init__(self, path: Path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        meta_path = Path(path).with_suffix(".meta.json")
        if not meta_path.exists():
            raise FileNotFoundError(f"ONNX model needs sidecar {meta_path.name} written by train_plastic.py")
        self.meta = json.loads(meta_path.read_text())

    def logits(self, batch: np.ndarray) -> np.ndarray:
        return self.sess.run(None, {self.inp: batch.astype(np.float32)})[0]


class PlasticClassifier:
    def __init__(self, weights: Optional[str] = None, config_path: str = "config/model_plastic.yaml",
                 backend: Any = None, meta: Optional[dict] = None):
        cfg = load_yaml(config_path)
        self.cfg = cfg["inference"]
        self.img_size = cfg["model"]["img_size"]
        self.backend, self.meta = backend, meta or {}
        self.unavailable_reason: Optional[str] = None
        if backend is not None:
            self.meta = meta or getattr(backend, "meta", {}) or {}
        else:
            self._load(weights or self.cfg["weights"])
        if self.backend is not None:
            self.class_names = self.meta.get("class_names", ["non_recyclable", "recyclable"])
            self.img_size = self.meta.get("img_size", self.img_size)
            self.temperature = float(self.meta.get("temperature", 1.0))

    # ---- loading ----
    def _load(self, weights):
        path = resolve(weights)
        if not path.exists():
            self.unavailable_reason = f"weights not found: {path}"
            return
        try:
            if path.suffix == ".onnx":
                self.backend = OnnxBackend(path)
            else:
                self.backend = TorchBackend(path, self.cfg.get("device", "cpu"))
            self.meta = self.backend.meta
        except Exception as e:  # missing torch/onnxruntime, corrupt file, ...
            self.backend = None
            self.unavailable_reason = f"could not load model: {e}"

    @property
    def available(self) -> bool:
        return self.backend is not None

    # ---- input handling ----
    def _to_pil(self, image):
        """Return (PIL.Image | None, error_code | None, reason)."""
        if image is None:
            return None, "empty_crop", "crop is None"
        try:
            if isinstance(image, (str, Path)):
                from plastic_common import open_rgb
                return open_rgb(image), None, ""
            if isinstance(image, Image.Image):
                if 0 in image.size:
                    return None, "empty_crop", "crop has zero size"
                return image.convert("RGB"), None, ""
            if isinstance(image, np.ndarray):
                if image.size == 0 or 0 in image.shape:
                    return None, "empty_crop", "crop has zero size"
                if image.dtype != np.uint8 or image.ndim not in (2, 3):
                    return None, "invalid_image", f"unsupported array dtype/shape {image.dtype}/{image.shape}"
                if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
                    return None, "invalid_image", f"unsupported channels {image.shape[2]}"
                if image.ndim == 2:
                    return Image.fromarray(image).convert("RGB"), None, ""
                if image.shape[2] == 1:
                    return Image.fromarray(image[:, :, 0]).convert("RGB"), None, ""
                rgb = image[:, :, 2::-1] if image.shape[2] == 3 else image[:, :, [2, 1, 0]]  # BGR(A)->RGB
                return Image.fromarray(np.ascontiguousarray(rgb)), None, ""
        except Exception as e:
            return None, "invalid_image", f"cannot decode image: {e}"
        return None, "invalid_image", f"unsupported input type {type(image).__name__}"

    @staticmethod
    def _laplacian_var(im: Image.Image) -> float:
        g = np.asarray(im.convert("L"), dtype=np.float32)
        if min(g.shape) < 3:
            return 0.0
        lap = g[1:-1, 1:-1] * -4 + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
        return float(lap.var())

    def _threshold_for(self, class_name: str) -> float:
        t = (self.cfg.get("thresholds") or {}).get(class_name)
        if t is None:
            t = self.meta.get("suggested_threshold")
        return float(t if t is not None else self.cfg["default_threshold"])

    # ---- main API ----
    def classify(self, image) -> PlasticResult:
        im, code, why = self._to_pil(image)
        if im is None:
            return _uncertain(why, code, source="none")
        if min(im.size) < self.cfg["min_crop_side"]:
            return _uncertain(f"crop {im.size} smaller than {self.cfg['min_crop_side']}px", "crop_too_small", source="none")
        mv = float(self.cfg.get("min_laplacian_var", 0) or 0)
        if mv > 0 and self._laplacian_var(im) < mv:
            return _uncertain("crop too blurry to judge", "blurry_crop", source="none")
        if not self.available:
            return PlasticResult(DISPLAY[UNCERTAIN], None, "none",
                                 self.unavailable_reason or "model not loaded", MODEL_UNAVAILABLE, {}, True)
        batch = to_model_array(im, self.img_size)[None]
        probs = softmax(self.backend.logits(batch), self.temperature)[0]
        pd = {n: float(p) for n, p in zip(self.class_names, probs)}
        k = int(probs.argmax())
        name, conf = self.class_names[k], float(probs[k])
        thr = self._threshold_for(name)
        if conf < thr:
            return _uncertain(f"top class '{name}' confidence {conf:.2f} < threshold {thr:.2f}",
                              "low_confidence", conf, pd)
        return PlasticResult(DISPLAY[name], conf, "custom_model",
                             f"custom model: {name} ({conf:.2f} >= {thr:.2f})", None, pd, False)
