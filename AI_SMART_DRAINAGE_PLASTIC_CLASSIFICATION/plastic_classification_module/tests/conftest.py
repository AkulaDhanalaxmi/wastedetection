import csv, sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from plastic_classifier import PlasticClassifier  # noqa: E402
from plastic_common import LABEL_COLUMNS  # noqa: E402


class FakeBackend:
    """SIMULATED model: returns fixed logits. Used only to test decision logic, never to measure accuracy."""
    meta = {"class_names": ["non_recyclable", "recyclable"], "img_size": 64, "temperature": 1.0, "suggested_threshold": None}
    def __init__(self, logits): self.fixed = np.array(logits, dtype=np.float32)
    def logits(self, batch): return np.tile(self.fixed, (len(batch), 1))


def make_classifier(logits, **meta_over):
    meta = {**FakeBackend.meta, **meta_over}
    return PlasticClassifier(backend=FakeBackend(logits), meta=meta)


def synth_image(seed: int, size=(96, 80)) -> Image.Image:
    """Structured pseudo-random image (different seeds -> different dHash). TEST FIXTURE ONLY, never shipped."""
    rng = np.random.default_rng(seed)
    low = rng.integers(0, 255, (6, 6, 3), dtype=np.uint8)
    return Image.fromarray(low).resize(size, Image.BICUBIC)


def write_labels(path: Path, rows: list[dict]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LABEL_COLUMNS); w.writeheader()
        for r in rows: w.writerow({c: r.get(c, "") for c in LABEL_COLUMNS})


@pytest.fixture
def rgb_crop():
    return np.array(synth_image(1))[:, :, ::-1].copy()  # BGR uint8, like an OpenCV crop
