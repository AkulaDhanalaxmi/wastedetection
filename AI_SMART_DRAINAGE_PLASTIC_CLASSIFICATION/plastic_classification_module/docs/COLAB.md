# Google Colab (GPU) instructions

```python
# 1. Runtime > Change runtime type > GPU. Upload plastic_classification_module.zip, or mount Drive.
!unzip -q AI_SMART_DRAINAGE_PLASTIC_CLASSIFICATION.zip && cd plastic_classification_module && pip -q install pyyaml pytest
%cd plastic_classification_module
!python -m pytest -q                                   # simulated tests (should pass)

# 2. Put your labelled crops + labels in data/ (copy from Drive), then:
!python scripts/validate_plastic_dataset.py --stage labels
!python scripts/prepare_plastic_dataset.py --overwrite
!python scripts/validate_plastic_dataset.py --stage split
!python scripts/dataset_stats.py

# 3. Train (MobileNetV3-Small; ~minutes on a T4 for a few thousand crops) and export ONNX for the laptop
!python train_plastic.py --device cuda --out runs/plastic/default --export-onnx

# 4. Held-out test metrics (real images only)
!python evaluate_plastic.py --weights runs/plastic/default/best.pt --split test

# 5. Download runs/plastic/default/{best.pt,best.onnx,best.meta.json} and reports/ to your laptop
```
annotate_cli.py needs a display: label on your laptop, or edit labels.csv, then upload it.
