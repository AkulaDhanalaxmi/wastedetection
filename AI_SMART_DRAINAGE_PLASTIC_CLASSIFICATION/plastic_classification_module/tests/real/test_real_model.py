"""REAL-MODEL tests. Skipped unless you have trained weights and a real processed test split.
Run:  PLASTIC_WEIGHTS=runs/plastic/default/best.pt pytest tests/real -s
These do NOT assert a fixed accuracy (that would be fabricated); they assert the evaluation runs on real
images and print real metrics for you to judge."""
import os
from pathlib import Path
import pytest
from plastic_classifier import PlasticClassifier
from plastic_common import load_yaml, resolve
from evaluate_plastic import evaluate, to_markdown

W = os.environ.get("PLASTIC_WEIGHTS")
TEST_DIR = resolve(load_yaml("config/dataset_plastic.yaml")["paths"]["processed_dir"]) / "test"
pytestmark = pytest.mark.skipif(not (W and Path(W).exists() and TEST_DIR.exists()), reason="needs trained weights (PLASTIC_WEIGHTS) and data/plastic_split/test")


def test_real_evaluation_runs_and_metrics_are_finite():
    r = evaluate(PlasticClassifier(weights=W), TEST_DIR)
    print(to_markdown(r, "test"))
    assert r["n"] > 0 and 0.0 <= r["accuracy"] <= 1.0 and 0.0 <= r["ece"] <= 1.0
