"""Dataset statistics report from split_manifest.csv (counts per split/class/source/basis)."""
from __future__ import annotations
import argparse, json
from collections import Counter
import _path  # noqa: F401
from plastic_common import load_yaml, resolve
from validate_plastic_dataset import read_csv


def build_stats(manifest: list[dict], classes: list[str]) -> dict:
    split_class = Counter((m["split"], m["label"]) for m in manifest)
    st = {"split_class": {f"{s}/{l}": n for (s, l), n in sorted(split_class.items())},
          "by_source": dict(Counter(m["source"] for m in manifest)),
          "by_basis": dict(Counter(m["label_basis"] for m in manifest)),
          "uncertain_held_out": sum(n for (s, _), n in split_class.items() if s == "_uncertain"),
          "audit_only": sum(n for (s, _), n in split_class.items() if s == "_audit")}
    tr = [split_class.get(("train", c), 0) for c in classes]
    st["train_imbalance_ratio"] = (max(tr) / min(tr)) if tr and min(tr) else None
    warn = []
    if st["train_imbalance_ratio"] is None or st["train_imbalance_ratio"] > 3:
        warn.append("train classes are imbalanced (>3:1) or one class is empty; class weighting is on, but collect more minority data")
    if set(st["by_basis"]) <= {"resin_policy"}:
        warn.append("ALL labels come from resin_policy (lab-style images). Expect weak transfer to drain photos; add human-labelled in-domain crops, especially for val/test")
    st["warnings"] = warn
    return st


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--config", default="config/dataset_plastic.yaml")
    a = ap.parse_args(); cfg = load_yaml(a.config)
    m = read_csv(resolve(cfg["paths"]["processed_dir"]) / "split_manifest.csv")
    st = build_stats(m, cfg["classes"]); rd = resolve(cfg["paths"]["reports_dir"]); rd.mkdir(parents=True, exist_ok=True)
    (rd / "dataset_stats.json").write_text(json.dumps(st, indent=2))
    lines = ["# Dataset statistics", "", "| split/class | images |", "|---|---|"] + [f"| {k} | {v} |" for k, v in st["split_class"].items()]
    lines += ["", f"by source: {st['by_source']}", f"by label basis: {st['by_basis']}", f"train imbalance: {st['train_imbalance_ratio']}"] + [f"- WARNING: {w}" for w in st["warnings"]]
    (rd / "dataset_stats.md").write_text("\n".join(lines)); print("\n".join(lines))

if __name__ == "__main__":
    main()
