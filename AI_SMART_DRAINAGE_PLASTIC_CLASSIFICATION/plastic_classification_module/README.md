# plastic_classification_module (Part 2)

Classifies each YOLO "plastic" detection as **Recyclable / Non-Recyclable / Uncertain-Manual Review**.
Self-contained, no files duplicated from Part 1, nothing in `waste_detection_module` modified, existing Qwen classifier preserved as fallback.

```
YOLO plastic box -> crop -> plastic_classifier.py (MobileNetV3, calibrated, thresholded)
                               | unusable crop / low confidence -> Uncertain / Manual Review
                               | model file missing            -> existing Qwen (own RAM check)
```
**Read first:** `docs/DATASET_RESEARCH.md`. No public dataset has trustworthy plastic-recyclability labels, so the
supported path is rule-based manual annotation of real crops. No weights, images or labels are shipped.

| File | Purpose |
|---|---|
| `train_plastic.py` / `evaluate_plastic.py` / `plastic_classifier.py` | train, evaluate (real metrics only), run inference |
| `config/dataset_plastic.yaml`, `model_plastic.yaml`, `recyclability_policy.yaml` | dataset, model and policy configuration |
| `scripts/` | download helper/catalogue, import, crop extraction, annotation, validation, duplicate detection, leakage-free split, stats |
| `integration/` | YOLO-crop pipeline + Qwen fallback adapter |
| `docs/`, `tests/` | research, annotation guide, integration, Colab, testing |

Quick start: `pip install -r requirements.txt && python -m pytest -q`, then follow `run_pipeline.sh` after labelling (see docs/ANNOTATION_GUIDE.md, docs/COLAB.md, docs/INTEGRATION.md).
The default recyclability policy is a conservative starting point, not a regulation: confirm it with your local recycler.
