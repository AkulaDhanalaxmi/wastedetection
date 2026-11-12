"""Validate labels.csv (and optionally the split folders). Exit code 1 when errors are found."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import _path  # noqa: F401
from plastic_common import LABEL_COLUMNS, LABEL_VALUES, UNCERTAIN, IMG_EXT, load_yaml, open_rgb, resolve


def validate_rows(rows: list[dict], cfg: dict, policy: dict, images_dir: Path, check_images=True):
    errors, warnings = [], []
    missing = [c for c in LABEL_COLUMNS if rows and c not in rows[0]]
    if missing:
        return [f"labels.csv missing columns: {missing}"], warnings
    rules = {r["id"]: r["label"] for r in policy["rules"]}
    resin = {k.upper(): v["label"] for k, v in policy["resin_codes"].items()}
    seen = set()
    for i, r in enumerate(rows, start=2):
        where = f"row {i} ({r['image_path']})"
        if r["image_path"] in seen: errors.append(f"{where}: duplicate image_path")
        seen.add(r["image_path"])
        if r["label"] not in LABEL_VALUES: errors.append(f"{where}: label '{r['label']}' not in {LABEL_VALUES}")
        if r["label_basis"] not in cfg["label_bases"]: errors.append(f"{where}: unknown label_basis '{r['label_basis']}'")
        if not r["group_id"]: errors.append(f"{where}: empty group_id (crops of one photo MUST share a group_id)")
        if not r["source"]: errors.append(f"{where}: empty source")
        basis = r["label_basis"]
        if basis in ("human_visual_rule", "human_two_annotator_agree"):
            rid = r["rule_id"]
            if rid not in rules: errors.append(f"{where}: rule_id '{rid}' not in policy")
            elif rules[rid] != r["label"]: errors.append(f"{where}: rule {rid} implies '{rules[rid]}' but label is '{r['label']}'")
            if not r["annotator"]: errors.append(f"{where}: human label without annotator")
        if basis == "resin_policy":
            code = (r["resin_code"] or "").upper()
            if code not in resin: errors.append(f"{where}: resin_policy label needs a known resin_code (got '{code}')")
            elif resin[code] != r["label"]: errors.append(f"{where}: resin {code} -> policy '{resin[code]}' but label is '{r['label']}'")
        if r["contamination"] == "heavy" and r["label"] == "recyclable":
            errors.append(f"{where}: heavily contaminated item cannot be 'recyclable' (rule NR2)")
        if r["label"] == "non_recyclable" and basis == "community_unverified":
            warnings.append(f"{where}: unverified non_recyclable label -> will be kept in _audit only")
        if check_images:
            p = images_dir / r["image_path"]
            if not p.exists(): errors.append(f"{where}: file not found"); continue
            try:
                w, h = open_rgb(p).size
                if min(w, h) < cfg["preprocess"]["min_side"]: warnings.append(f"{where}: tiny image {w}x{h}")
            except Exception as e:
                errors.append(f"{where}: unreadable image ({e})")
    return errors, warnings


def read_csv(path: Path):
    import csv
    with open(path, newline="", encoding="utf-8") as f: return list(csv.DictReader(f))


def check_split_leakage(manifest_rows: list[dict]):
    """Return list of leakage problems in a split_manifest.csv (md5 or cluster shared across splits)."""
    errs = []
    for key in ("md5", "cluster"):
        seen = {}
        for r in manifest_rows:
            k = r[key]
            if k in seen and seen[k] != r["split"]:
                errs.append(f"{key} {k} appears in both '{seen[k]}' and '{r['split']}'")
            seen.setdefault(k, r["split"])
    return errs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/dataset_plastic.yaml"); ap.add_argument("--stage", choices=["labels", "split"], default="labels")
    a = ap.parse_args()
    cfg = load_yaml(a.config); policy = load_yaml(cfg["paths"]["policy"])
    if a.stage == "labels":
        csv_path = resolve(cfg["paths"]["labels_csv"])
        if not csv_path.exists(): print(f"ERROR: {csv_path} not found. See docs/ANNOTATION_GUIDE.md"); sys.exit(1)
        rows = read_csv(csv_path)
        errs, warns = validate_rows(rows, cfg, policy, resolve(cfg["paths"]["images_dir"]))
        print(f"rows={len(rows)} errors={len(errs)} warnings={len(warns)}")
    else:
        m = resolve(cfg["paths"]["processed_dir"]) / "split_manifest.csv"
        if not m.exists(): print(f"ERROR: {m} not found; run prepare_plastic_dataset.py"); sys.exit(1)
        errs, warns = check_split_leakage(read_csv(m)), []
        print(f"leakage problems={len(errs)}")
    for w in warns[:50]: print("WARN ", w)
    for e in errs[:100]: print("ERROR", e)
    sys.exit(1 if errs else 0)

if __name__ == "__main__":
    main()
