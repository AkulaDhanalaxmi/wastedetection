"""Shared helpers (numpy + Pillow only, no torch) for the plastic classification module."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

MODULE_ROOT = Path(__file__).resolve().parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Internal (file/folder) label names
RECYCLABLE, NON_RECYCLABLE, UNCERTAIN = "recyclable", "non_recyclable", "uncertain"
LABEL_VALUES = (RECYCLABLE, NON_RECYCLABLE, UNCERTAIN)
# Display strings used in the dashboard / returned by the pipeline
DISPLAY = {
    RECYCLABLE: "Recyclable",
    NON_RECYCLABLE: "Non-Recyclable",
    UNCERTAIN: "Uncertain / Manual Review",
}


def load_yaml(path):
    import yaml
    with open(resolve(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else MODULE_ROOT / p


def open_rgb(path) -> Image.Image:
    """Open an image, apply EXIF rotation, convert to RGB. Raises on unreadable files."""
    with Image.open(path) as im:
        im.load()
        im = ImageOps.exif_transpose(im)
        return im.convert("RGB")


def letterbox(im: Image.Image, size: int, fill=(114, 114, 114)) -> Image.Image:
    """Resize keeping aspect ratio, pad to a square. Used identically in training and inference."""
    w, h = im.size
    scale = size / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = im.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def to_model_array(im: Image.Image, size: int) -> np.ndarray:
    """PIL RGB -> float32 CHW, ImageNet-normalised."""
    arr = np.asarray(letterbox(im, size), dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.transpose(2, 0, 1).astype(np.float32)


def md5_file(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(im: Image.Image, hash_size: int = 8) -> int:
    g = im.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# ---------- metrics (numpy only) ----------
def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / max(temperature, 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def confusion_matrix(y_true, y_pred, n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def per_class_prf(cm: np.ndarray) -> list[dict]:
    out = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        out.append({"precision": float(p), "recall": float(r), "f1": float(f1), "support": int(cm[i].sum())})
    return out


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    ece, n = 0.0, len(y)
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.sum() / n * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(ece)


def sweep_thresholds(probs: np.ndarray, y: np.ndarray, grid=None) -> list[dict]:
    """For each confidence threshold: coverage and accuracy on the accepted (non-abstained) samples."""
    grid = grid if grid is not None else np.round(np.arange(0.5, 1.0, 0.05), 2)
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    rows = []
    for t in grid:
        acc_mask = conf >= t
        cov = float(acc_mask.mean()) if len(y) else 0.0
        acc = float((pred[acc_mask] == y[acc_mask]).mean()) if acc_mask.any() else None
        rows.append({"threshold": float(t), "coverage": cov, "accepted_accuracy": acc})
    return rows


def suggest_threshold(rows: list[dict], target_acc: float, min_coverage: float):
    """Smallest threshold meeting target accuracy with enough coverage; None if no threshold qualifies."""
    for r in rows:
        if r["accepted_accuracy"] is not None and r["accepted_accuracy"] >= target_acc and r["coverage"] >= min_coverage:
            return r["threshold"]
    return None


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Grid-search temperature minimising NLL (no torch needed). Returns 1.0 if data are degenerate."""
    if len(y) < 10:
        return 1.0
    best_t, best_nll = 1.0, np.inf
    for t in np.round(np.arange(0.5, 5.01, 0.05), 2):
        p = softmax(logits, float(t))
        nll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1)).mean()
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t


def append_csv_row(csv_path: Path, columns: list[str], row: dict):
    import csv
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in columns})


LABEL_COLUMNS = ["image_path", "label", "label_basis", "rule_id", "source", "group_id",
                 "annotator", "resin_code", "contamination", "notes"]
