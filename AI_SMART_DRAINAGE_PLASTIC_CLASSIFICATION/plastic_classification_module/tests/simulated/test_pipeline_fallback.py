"""SIMULATED tests of the YOLO-crop -> classifier -> Qwen-fallback routing (fake Qwen function, fake model)."""
import numpy as np
from conftest import make_classifier
from integration.plastic_pipeline import PlasticRecyclabilityPipeline, crop_bbox
from integration.qwen_adapter import QwenFallback
from plastic_classifier import PlasticClassifier


class Counter:
    def __init__(self, ret): self.n, self.ret = 0, ret
    def __call__(self, crop): self.n += 1; return self.ret


def unavailable(tmp_path): return PlasticClassifier(weights=str(tmp_path / "missing.pt"))


def test_qwen_fallback_when_custom_model_unavailable(tmp_path, rgb_crop):
    q = Counter("Recyclable"); p = PlasticRecyclabilityPipeline(unavailable(tmp_path), QwenFallback(q))
    r = p.classify_crop(rgb_crop)
    assert r.label == "Recyclable" and r.source == "qwen_fallback" and q.n == 1


def test_qwen_dict_output_normalised(tmp_path, rgb_crop):
    q = Counter({"classification": "Non-Recyclable", "confidence": 0.7, "reason": "foam"})
    r = PlasticRecyclabilityPipeline(unavailable(tmp_path), QwenFallback(q)).classify_crop(rgb_crop)
    assert r.label == "Non-Recyclable" and r.confidence == 0.7 and r.reason == "foam"


def test_qwen_not_loaded_when_custom_model_decides(rgb_crop):
    q = Counter("Recyclable"); p = PlasticRecyclabilityPipeline(make_classifier([3, -3]), QwenFallback(q))
    assert p.classify_crop(rgb_crop).label == "Non-Recyclable" and q.n == 0


def test_qwen_not_called_for_empty_or_invalid_crop(tmp_path):
    q = Counter("Recyclable"); p = PlasticRecyclabilityPipeline(unavailable(tmp_path), QwenFallback(q))
    assert p.classify_crop(None).error_code == "empty_crop" and p.classify_crop(np.zeros((0, 4, 3), np.uint8)).error_code == "empty_crop"
    assert q.n == 0


def test_qwen_respects_ram_check(tmp_path, rgb_crop):
    q = Counter("Recyclable"); p = PlasticRecyclabilityPipeline(unavailable(tmp_path), QwenFallback(q, ram_ok_fn=lambda: False))
    r = p.classify_crop(rgb_crop)
    assert r.error_code == "qwen_blocked_low_ram" and r.needs_manual_review and q.n == 0


def test_qwen_error_becomes_manual_review(tmp_path, rgb_crop):
    def boom(_): raise RuntimeError("ollama down")
    r = PlasticRecyclabilityPipeline(unavailable(tmp_path), QwenFallback(boom)).classify_crop(rgb_crop)
    assert r.error_code == "qwen_error" and r.needs_manual_review


def test_no_model_and_no_qwen_is_manual_review(tmp_path, rgb_crop):
    r = PlasticRecyclabilityPipeline(unavailable(tmp_path), None).classify_crop(rgb_crop)
    assert r.label == "Uncertain / Manual Review" and r.needs_manual_review and r.source == "none"


def test_low_confidence_stays_manual_review_by_default(rgb_crop):
    q = Counter("Recyclable"); p = PlasticRecyclabilityPipeline(make_classifier([0.1, 0.0]), QwenFallback(q))
    r = p.classify_crop(rgb_crop)
    assert r.label == "Uncertain / Manual Review" and r.source == "custom_model" and q.n == 0


def test_optional_qwen_second_opinion_on_low_confidence(rgb_crop):
    q = Counter("Non-Recyclable"); p = PlasticRecyclabilityPipeline(make_classifier([0.1, 0.0]), QwenFallback(q), qwen_on_low_confidence=True)
    assert p.classify_crop(rgb_crop).source == "qwen_fallback" and q.n == 1


def test_manual_review_result_flags(rgb_crop):
    r = make_classifier([0.0, 0.1]).classify(rgb_crop)
    assert r.needs_manual_review and r.to_dict()["label"] == "Uncertain / Manual Review"


def test_annotate_detections_only_plastic(rgb_crop):
    frame = np.full((200, 300, 3), 127, np.uint8)
    dets = [{"class_name": "plastic", "bbox": [10, 10, 120, 100]}, {"class_name": "metal", "bbox": [0, 0, 50, 50]},
            {"class_name": "plastic", "bbox": [50, 50, 50, 50]}]            # zero-area box
    out = PlasticRecyclabilityPipeline(make_classifier([-3, 3])).annotate_detections(frame, dets)
    assert out[0]["recyclability"]["label"] == "Recyclable" and "recyclability" not in out[1]
    assert out[2]["recyclability"]["error_code"] == "empty_crop"


def test_crop_bbox_clamps_and_rejects():
    f = np.zeros((100, 100, 3), np.uint8)
    c = crop_bbox(f, [-20, -20, 60, 60], 0.1); assert c.shape[0] <= 100 and c.shape[1] <= 100 and c.size > 0
    assert crop_bbox(f, [10, 10, 10, 50]) is None and crop_bbox(f, [float("nan"), 0, 5, 5]) is None
    assert crop_bbox(None, [0, 0, 5, 5]) is None and crop_bbox(f, "bad") is None
