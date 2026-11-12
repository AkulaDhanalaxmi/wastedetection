"""SIMULATED tests: a fake backend returns fixed logits. They verify decision logic and error handling,
NOT the accuracy of any trained model."""
import numpy as np
from conftest import make_classifier
from plastic_classifier import PlasticClassifier, normalize_label


def test_recyclable_prediction(rgb_crop):
    r = make_classifier([-3.0, 3.0]).classify(rgb_crop)
    assert r.label == "Recyclable" and r.source == "custom_model" and r.confidence > 0.95 and not r.needs_manual_review


def test_non_recyclable_prediction(rgb_crop):
    r = make_classifier([3.0, -3.0]).classify(rgb_crop)
    assert r.label == "Non-Recyclable" and not r.needs_manual_review


def test_low_confidence_goes_to_manual_review(rgb_crop):
    r = make_classifier([0.2, 0.0]).classify(rgb_crop)
    assert r.label == "Uncertain / Manual Review" and r.error_code == "low_confidence" and r.needs_manual_review
    assert set(r.probabilities) == {"non_recyclable", "recyclable"}


def test_threshold_from_checkpoint_meta_is_used(rgb_crop):
    strict = make_classifier([0.0, 2.0], suggested_threshold=0.99).classify(rgb_crop)   # p~0.88
    assert strict.error_code == "low_confidence"


def test_per_class_threshold_override(rgb_crop):
    c = make_classifier([-3.0, 3.0]); c.cfg["thresholds"] = {"recyclable": 0.999, "non_recyclable": None}
    assert c.classify(rgb_crop).error_code == "low_confidence"


def test_empty_crops():
    c = make_classifier([-3, 3])
    for bad in (None, np.zeros((0, 10, 3), np.uint8), np.zeros((10, 0, 3), np.uint8)):
        r = c.classify(bad)
        assert r.error_code == "empty_crop" and r.needs_manual_review and r.confidence is None


def test_invalid_images(tmp_path):
    c = make_classifier([-3, 3]); bad_file = tmp_path / "x.jpg"; bad_file.write_bytes(b"not an image")
    for bad in (str(tmp_path / "missing.jpg"), bad_file, np.zeros((50, 50, 3), np.float32), np.zeros((5, 5, 5, 3), np.uint8), 123, "text"):
        assert c.classify(bad).error_code == "invalid_image", bad


def test_tiny_crop_not_classified():
    assert make_classifier([-3, 3]).classify(np.zeros((8, 8, 3), np.uint8)).error_code == "crop_too_small"


def test_grayscale_and_bgra_inputs_accepted():
    c = make_classifier([-3, 3])
    assert c.classify(np.full((64, 64), 128, np.uint8)).label == "Recyclable"
    assert c.classify(np.full((64, 64, 4), 128, np.uint8)).label == "Recyclable"


def test_model_unavailable_is_reported_not_guessed(tmp_path, rgb_crop):
    c = PlasticClassifier(weights=str(tmp_path / "nope.pt"))
    assert not c.available
    r = c.classify(rgb_crop)
    assert r.error_code == "model_unavailable" and r.label == "Uncertain / Manual Review" and r.needs_manual_review


def test_corrupt_weights_file_reported_unavailable(tmp_path, rgb_crop):
    p = tmp_path / "bad.pt"; p.write_bytes(b"garbage")
    assert PlasticClassifier(weights=str(p)).classify(rgb_crop).error_code == "model_unavailable"


def test_normalize_label():
    assert normalize_label("Non-Recyclable") == "non_recyclable"
    assert normalize_label("not recyclable") == "non_recyclable"
    assert normalize_label("RECYCLABLE") == "recyclable"
    assert normalize_label("Uncertain/Manual Review") == "uncertain"
    assert normalize_label(None) == "uncertain" and normalize_label("banana") == "uncertain"
