"""Evaluate on the held-out split (default: test) with REAL images only. Reports accuracy, balanced accuracy,
macro-F1, per-class P/R/F1, confusion matrix, ECE, threshold sweep, abstention behaviour (what the deployed
thresholds would route to manual review), how the _uncertain hold-out is handled, and per-source / per-basis
breakdowns (so lab-style data cannot hide weak performance on in-domain data).
Usage: python evaluate_plastic.py --weights runs/plastic/default/best.pt --split test"""
from __future__ import annotations

import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np

from plastic_classifier import PlasticClassifier
from plastic_common import (IMG_EXT, confusion_matrix, expected_calibration_error, load_yaml, open_rgb, per_class_prf,
                            resolve, softmax, sweep_thresholds, to_model_array)


def _logits(clf: PlasticClassifier, paths: list[Path], bs: int = 32) -> np.ndarray:
    out = []
    for i in range(0, len(paths), bs):
        batch = np.stack([to_model_array(open_rgb(p), clf.img_size) for p in paths[i:i + bs]])
        out.append(clf.backend.logits(batch))
    return np.concatenate(out) if out else np.zeros((0, len(clf.class_names)))


def _abstain_mask(clf: PlasticClassifier, probs: np.ndarray) -> np.ndarray:
    thr = np.array([clf._threshold_for(clf.class_names[k]) for k in probs.argmax(1)])
    return probs.max(1) < thr


def evaluate(clf: PlasticClassifier, split_dir: Path, manifest: dict | None = None, uncertain_dir: Path | None = None) -> dict:
    if not clf.available:
        raise SystemExit(f"Model unavailable: {clf.unavailable_reason}. Train first (train_plastic.py).")
    names = clf.class_names
    paths, y = [], []
    for i, c in enumerate(names):
        ps = sorted(p for p in (split_dir / c).glob("*") if p.suffix.lower() in IMG_EXT)
        paths += ps; y += [i] * len(ps)
    if not paths:
        raise SystemExit(f"No images under {split_dir}")
    y = np.array(y); probs = softmax(_logits(clf, paths), clf.temperature); pred = probs.argmax(1)
    cm = confusion_matrix(y, pred, len(names)); prf = per_class_prf(cm)
    ab = _abstain_mask(clf, probs); acc_mask = ~ab
    res = {"n": int(len(y)), "classes": names, "temperature": clf.temperature,
           "accuracy": float((pred == y).mean()), "balanced_accuracy": float(np.mean([r["recall"] for r in prf])),
           "macro_f1": float(np.mean([r["f1"] for r in prf])), "per_class": dict(zip(names, prf)), "confusion_matrix": cm.tolist(),
           "ece": expected_calibration_error(probs, y), "threshold_sweep": sweep_thresholds(probs, y),
           "deployed": {"coverage": float(acc_mask.mean()), "accepted_accuracy": float((pred[acc_mask] == y[acc_mask]).mean()) if acc_mask.any() else None,
                        "manual_review_rate": float(ab.mean()),
                        "manual_review_rate_by_true_class": {c: float(ab[y == i].mean()) if (y == i).any() else None for i, c in enumerate(names)},
                        "non_recyclable_called_recyclable_rate": None}}
    if "non_recyclable" in names and "recyclable" in names:
        nr, rc = names.index("non_recyclable"), names.index("recyclable")
        m = (y == nr) & acc_mask
        res["deployed"]["non_recyclable_called_recyclable_rate"] = float((pred[m] == rc).mean()) if m.any() else None
    if manifest:  # breakdowns by source and label basis
        rel = {str(p.relative_to(split_dir.parent)).replace("\\", "/"): k for k, p in enumerate(paths)}
        for key in ("source", "label_basis"):
            groups = defaultdict(list)
            for img, row in manifest.items():
                if img in rel: groups[row[key]].append(rel[img])
            res[f"by_{key}"] = {g: {"n": len(ix), "accuracy": float((pred[ix] == y[ix]).mean())} for g, ix in groups.items()}
    if uncertain_dir and uncertain_dir.exists():
        up = sorted(p for p in uncertain_dir.rglob("*") if p.suffix.lower() in IMG_EXT)
        if up:
            um = _abstain_mask(clf, softmax(_logits(clf, up), clf.temperature))
            res["uncertain_holdout"] = {"n": len(up), "routed_to_manual_review": float(um.mean()),
                                        "note": "These have no ground-truth class; a high routed fraction is desirable, a low one means overconfidence on ambiguous items."}
    return res


def to_markdown(r: dict, split: str) -> str:
    L = [f"# Plastic classifier evaluation ({split}, n={r['n']})", "", f"accuracy {r['accuracy']:.3f} | balanced acc {r['balanced_accuracy']:.3f} | macro-F1 {r['macro_f1']:.3f} | ECE {r['ece']:.3f}", "",
         "| class | precision | recall | F1 | support |", "|---|---|---|---|---|"]
    L += [f"| {c} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['support']} |" for c, v in r["per_class"].items()]
    d = r["deployed"]
    L += ["", f"confusion matrix (rows=true, cols=pred, order {r['classes']}): {r['confusion_matrix']}", "",
          f"deployed thresholds: coverage {d['coverage']:.3f}, accuracy on accepted {d['accepted_accuracy']}, manual-review rate {d['manual_review_rate']:.3f}",
          f"non-recyclable wrongly accepted as recyclable: {d['non_recyclable_called_recyclable_rate']}", "", "| threshold | coverage | accepted accuracy |", "|---|---|---|"]
    L += [f"| {s['threshold']} | {s['coverage']:.3f} | {s['accepted_accuracy']} |" for s in r["threshold_sweep"]]
    for k in ("by_source", "by_label_basis", "uncertain_holdout"):
        if k in r: L += ["", f"{k}: {json.dumps(r[k])}"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default=None); ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--dataset-config", default="config/dataset_plastic.yaml"); ap.add_argument("--model-config", default="config/model_plastic.yaml")
    a = ap.parse_args()
    dcfg = load_yaml(a.dataset_config); proc = resolve(dcfg["paths"]["processed_dir"])
    clf = PlasticClassifier(weights=a.weights, config_path=a.model_config)
    import csv
    mp = proc / "split_manifest.csv"
    manifest = {r["image"]: r for r in csv.DictReader(open(mp, newline="", encoding="utf-8"))} if mp.exists() else None
    res = evaluate(clf, proc / a.split, manifest, proc / "_uncertain")
    rd = resolve(dcfg["paths"]["reports_dir"]); rd.mkdir(parents=True, exist_ok=True)
    (rd / f"eval_{a.split}.json").write_text(json.dumps(res, indent=2)); (rd / f"eval_{a.split}.md").write_text(to_markdown(res, a.split))
    print(to_markdown(res, a.split))


if __name__ == "__main__":
    main()
