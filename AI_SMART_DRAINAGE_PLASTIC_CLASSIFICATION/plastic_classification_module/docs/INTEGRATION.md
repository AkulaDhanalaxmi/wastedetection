# Integration with the existing Flask app

1. Copy `plastic_classification_module/` into your project root (it does not touch `waste_detection_module`).
2. `pip install -r plastic_classification_module/requirements.txt` (laptop: after `--export-onnx` you only need numpy, Pillow, PyYAML, onnxruntime).
3. Put trained weights at `plastic_classification_module/runs/plastic/default/best.pt` (or `best.onnx` + `best.meta.json`). Without them the pipeline automatically falls back to Qwen.
4. In your Flask app (adapt the two names marked YOURS; I could not see your code):

```python
import sys; sys.path.insert(0, "plastic_classification_module")
from plastic_classifier import PlasticClassifier
from integration.plastic_pipeline import PlasticRecyclabilityPipeline
from integration.qwen_adapter import QwenFallback
from my_existing_qwen_code import classify_plastic_with_qwen, ram_is_safe   # YOURS: existing function + existing RAM check

plastic_pipeline = PlasticRecyclabilityPipeline(
    classifier=PlasticClassifier(),                                   # loads ONLY the small CNN
    qwen=QwenFallback(classify_plastic_with_qwen, ram_is_safe),       # called lazily, only if the CNN is unavailable
    qwen_on_low_confidence=False)                                     # True = ask Qwen when the CNN is unsure (uses more RAM)

# after YOLO inference, for each frame/upload:
dets = [{"class_name": names[int(c)], "bbox": [x1, y1, x2, y2], "confidence": float(s)} for ...]
dets = plastic_pipeline.annotate_detections(frame_bgr, dets)   # adds det["recyclability"] to plastic boxes only
# det["recyclability"] = {"label": "Recyclable" | "Non-Recyclable" | "Uncertain / Manual Review", "confidence", "source", "reason", "error_code", "needs_manual_review", "probabilities"}
```
5. Dashboard: show `label` (+ `source`: `custom_model` / `qwen_fallback` / `none`); highlight `needs_manual_review`.
   If your dashboard expects "Uncertain/Manual Review" spelled differently, edit `DISPLAY` in `plastic_common.py`.
6. Qwen's existing implementation is untouched; `QwenFallback` only wraps your function and your RAM check, and calls neither unless needed. Empty or invalid crops never reach Qwen.
7. Thresholds: `train_plastic.py` stores a val-derived threshold in the checkpoint. Override in `config/model_plastic.yaml -> inference.thresholds`. Higher = more manual review, fewer wrong auto-decisions.
