#!/usr/bin/env bash
# Order of operations once you have crops + labels.csv (see docs/ANNOTATION_GUIDE.md).
set -euo pipefail
cd "$(dirname "$0")"
python scripts/validate_plastic_dataset.py --stage labels
python scripts/find_duplicates.py
python scripts/prepare_plastic_dataset.py --overwrite
python scripts/validate_plastic_dataset.py --stage split
python scripts/dataset_stats.py
python train_plastic.py --out runs/plastic/default --export-onnx
python evaluate_plastic.py --weights runs/plastic/default/best.pt --split test
