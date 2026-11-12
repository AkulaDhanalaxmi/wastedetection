import io
import sys
import types
from types import SimpleNamespace

from PIL import Image

from app.custom_classifier import CustomPlasticClassifier


def jpeg_bytes():
    image = Image.new("RGB", (40, 40), "white")
    output = io.BytesIO()
    image.save(output, format="JPEG")
    return output.getvalue()


def test_missing_custom_weights_are_reported_without_loading_module(tmp_path, monkeypatch):
    def fail_import(_name):
        raise AssertionError("The classifier module must not load without trained weights.")

    monkeypatch.setattr("app.custom_classifier.importlib.import_module", fail_import)
    classifier = CustomPlasticClassifier(
        module_root=tmp_path,
        weights_path=tmp_path / "runs" / "plastic" / "default" / "best.pt",
    )

    result = classifier.classify(jpeg_bytes())

    assert result.label == "uncertain"
    assert result.error_code == "model_unavailable"
    assert "weights are missing" in result.reason
    assert "best.pt" in result.reason


def test_invalid_crop_is_manual_review_without_loading_custom_model(tmp_path, monkeypatch):
    def fail_import(_name):
        raise AssertionError("Invalid crops must not load the classifier module.")

    monkeypatch.setattr("app.custom_classifier.importlib.import_module", fail_import)
    classifier = CustomPlasticClassifier(
        module_root=tmp_path,
        weights_path=tmp_path / "best.pt",
    )

    result = classifier.classify(b"not an image")

    assert result.label == "uncertain"
    assert result.error_code == "invalid_image"
    assert result.source == "none"


def test_custom_result_is_mapped_to_application_prediction(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    module_root = tmp_path / "module"
    (module_root / "config").mkdir(parents=True)
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"mock weights")

    class FakeClassifier:
        available = True
        unavailable_reason = None

        def __init__(self, *, weights, config_path):
            assert weights == str(weights_path)
            assert config_path == str(module_root / "config" / "model_plastic.yaml")

        def classify(self, image):
            assert image.mode == "RGB"
            return SimpleNamespace(
                label="Recyclable",
                confidence=0.91,
                reason="Custom model accepted the crop.",
                error_code=None,
            )

    weights_path = weights
    classifier_module = types.ModuleType("plastic_classifier")
    classifier_module.PlasticClassifier = FakeClassifier
    monkeypatch.setitem(sys.modules, "plastic_classifier", classifier_module)
    classifier = CustomPlasticClassifier(
        module_root=module_root,
        weights_path=weights,
    )

    result = classifier.classify(jpeg_bytes())

    assert result.label == "recyclable"
    assert result.confidence == 0.91
    assert result.source == "custom_model"
