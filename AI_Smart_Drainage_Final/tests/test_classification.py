import base64
import io
import json

from PIL import Image

from app.classification import ClassificationError, OllamaPlasticClassifier


def test_qwen_schema_parsing():
    response = {
        "response": json.dumps({
            "predicted_label": "recyclable",
            "reason": "Clean rigid plastic surface.",
            "confidence": 0.81,
        })
    }
    result = OllamaPlasticClassifier._parse(json.dumps(response))
    assert result.label == "recyclable"
    assert result.confidence == 0.81


def test_invalid_qwen_label_is_rejected():
    response = {
        "response": json.dumps({
            "predicted_label": "probably recyclable",
            "reason": "not a valid label",
            "confidence": 0.8,
        })
    }
    try:
        OllamaPlasticClassifier._parse(json.dumps(response))
    except ClassificationError:
        pass
    else:
        raise AssertionError("Invalid classification label was accepted.")


def test_crop_resize_keeps_aspect_ratio_and_jpeg_quality():
    image = Image.new("RGB", (1024, 512), "white")
    source = io.BytesIO()
    image.save(source, format="PNG")
    encoded = OllamaPlasticClassifier(max_image_side=512)._encode_crop(source.getvalue())
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as prepared:
        assert prepared.size == (512, 256)
        assert prepared.format == "JPEG"
