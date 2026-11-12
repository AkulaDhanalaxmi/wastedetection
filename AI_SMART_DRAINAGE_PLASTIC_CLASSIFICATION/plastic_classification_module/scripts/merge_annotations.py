"""Merge two annotators' CSVs: agreement -> human_two_annotator_agree, disagreement -> uncertain. Prints Cohen's kappa."""
from __future__ import annotations
import argparse, csv
from collections import Counter
import _path  # noqa: F401
from plastic_common import LABEL_COLUMNS, LABEL_VALUES, load_yaml


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    if n == 0: return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[l] * cb[l] for l in LABEL_VALUES) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def merge(rows_a: list[dict], rows_b: dict, policy_rules: dict):
    merged, la, lb = [], [], []
    for ra in rows_a:
        rb = rows_b.get(ra["image_path"])
        if rb is None: continue
        la.append(ra["label"]); lb.append(rb["label"])
        if ra["label"] == rb["label"] and ra["rule_id"] == rb["rule_id"]:
            merged.append({**ra, "label_basis": "human_two_annotator_agree", "annotator": f"{ra['annotator']}+{rb['annotator']}"})
        elif ra["label"] == rb["label"]:   # same label, different rule: keep label, record first rule
            merged.append({**ra, "label_basis": "human_two_annotator_agree", "annotator": f"{ra['annotator']}+{rb['annotator']}",
                           "notes": (ra.get("notes", "") + f" rules differ: {ra['rule_id']}/{rb['rule_id']}").strip()})
        else:
            merged.append({**ra, "label": "uncertain", "rule_id": "U1_unknown_material", "label_basis": "human_visual_rule",
                           "annotator": f"{ra['annotator']}+{rb['annotator']}", "notes": f"annotators disagreed: {ra['label']} vs {rb['label']}"})
    return merged, cohen_kappa(la, lb)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="config/dataset_plastic.yaml"); x = ap.parse_args()
    rd = lambda p: list(csv.DictReader(open(p, newline="", encoding="utf-8")))
    cfg = load_yaml(x.config)
    merged, kappa = merge(rd(x.a), {r["image_path"]: r for r in rd(x.b)}, {})
    with open(x.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LABEL_COLUMNS, extrasaction="ignore"); w.writeheader(); w.writerows(merged)
    print(f"merged {len(merged)} items, Cohen's kappa = {kappa:.3f}")

if __name__ == "__main__":
    main()
