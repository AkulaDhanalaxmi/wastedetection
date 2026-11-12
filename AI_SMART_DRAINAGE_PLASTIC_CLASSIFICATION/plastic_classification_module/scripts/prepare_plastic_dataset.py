"""labels.csv -> de-duplicated, leakage-free train/val/test folders (+ _uncertain, _audit)."""
from __future__ import annotations
import argparse, csv, random, shutil, sys
from collections import Counter, defaultdict
from pathlib import Path
import _path  # noqa: F401
from plastic_common import NON_RECYCLABLE, RECYCLABLE, UNCERTAIN, dhash, load_yaml, md5_file, open_rgb, resolve
from find_duplicates import UnionFind, near_dup_pairs
from validate_plastic_dataset import check_split_leakage, read_csv, validate_rows


def assign_splits(cluster_labels: dict, ratios: dict, seed: int) -> dict:
    """cluster_labels: {cluster_id: [label, ...]} -> {cluster_id: split}. Whole clusters go to one split;
    greedy fill keeps per-class proportions close to the ratios."""
    rng = random.Random(seed)
    totals = Counter(l for ls in cluster_labels.values() for l in ls)
    target = {s: {l: totals[l] * r for l in totals} for s, r in ratios.items()}
    cur = {s: Counter() for s in ratios}
    ids = list(cluster_labels); rng.shuffle(ids); ids.sort(key=lambda c: -len(cluster_labels[c]))
    out = {}
    for cid in ids:
        cnt = Counter(cluster_labels[cid])
        def deficit(s): return sum(n * (target[s][l] - cur[s][l]) / max(target[s][l], 1.0) for l, n in cnt.items())
        best = max(ratios, key=lambda s: (deficit(s), rng.random()))
        out[cid] = best; cur[best].update(cnt)
    return out


def save_processed(src: Path, dst: Path, cfg: dict):
    im = open_rgb(src); ms = cfg["preprocess"]["max_side"]
    if max(im.size) > ms:
        s = ms / max(im.size); im = im.resize((max(1, round(im.size[0] * s)), max(1, round(im.size[1] * s))))
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, "JPEG", quality=cfg["preprocess"]["jpeg_quality"])


def run(cfg: dict, policy: dict, overwrite=False, labels_csv=None, images_dir=None, out_dir=None) -> dict:
    labels_csv = Path(labels_csv or resolve(cfg["paths"]["labels_csv"]))
    images_dir = Path(images_dir or resolve(cfg["paths"]["images_dir"]))
    out = Path(out_dir or resolve(cfg["paths"]["processed_dir"]))
    rows = read_csv(labels_csv)
    errs, warns = validate_rows(rows, cfg, policy, images_dir)
    if errs:
        raise SystemExit("Label validation failed:\n" + "\n".join(errs[:50]))
    if out.exists():
        if not overwrite: raise SystemExit(f"{out} exists; use --overwrite")
        shutil.rmtree(out)
    # 1. hash everything
    items = []
    for r in rows:
        p = images_dir / r["image_path"]
        im = open_rgb(p)
        items.append({**r, "path": p, "md5": md5_file(p), "dh": dhash(im, cfg["dedupe"]["hash_size"])})
    # 2. exact duplicates: keep first; conflicting labels -> ambiguous -> uncertain
    by_md5 = defaultdict(list)
    for it in items: by_md5[it["md5"]].append(it)
    kept, dropped_dups, conflicts = [], 0, 0
    for group in by_md5.values():
        first = group[0]
        if len({g["label"] for g in group}) > 1:
            first = {**first, "label": UNCERTAIN, "notes": (first.get("notes", "") + " [conflicting duplicate labels]").strip()}
            conflicts += 1
        kept.append(first); dropped_dups += len(group) - 1
    # 3. clusters = near-duplicate hashes U group_id (same photo / same session never straddles splits)
    uf = UnionFind(len(kept))
    for a, b in near_dup_pairs([k["dh"] for k in kept], cfg["dedupe"]["max_hamming"]): uf.union(a, b)
    first_by_group = {}
    for i, k in enumerate(kept):
        g = (k["source"], k["group_id"])
        if g in first_by_group: uf.union(i, first_by_group[g])
        else: first_by_group[g] = i
    for i, k in enumerate(kept): k["cluster"] = str(uf.find(i))
    # 4. route: uncertain / audit (untrusted basis) / trainable
    trusted = set(cfg["trusted_bases"])
    trainable = [k for k in kept if k["label"] != UNCERTAIN and k["label_basis"] in trusted]
    side = [k for k in kept if k not in trainable]
    # clusters touching side items are fine (they are not trained on); but near-dups of side items inside
    # trainable clusters are impossible by construction only if clustered together -> drop trainable items
    # that share a cluster with an UNCERTAIN item? No: uncertain items never enter any split, so no leakage.
    cl_labels = defaultdict(list)
    for k in trainable: cl_labels[k["cluster"]].append(k["label"])
    ratios = {s: cfg["split"][s] for s in ("train", "val", "test")}
    split_of = assign_splits(cl_labels, ratios, cfg["split"]["seed"])
    manifest = []
    for k in trainable:
        s = split_of[k["cluster"]]
        name = f"{k['source']}__{k['md5'][:12]}.jpg"
        save_processed(k["path"], out / s / k["label"] / name, cfg)
        manifest.append({"image": f"{s}/{k['label']}/{name}", "split": s, "label": k["label"], "source": k["source"],
                         "label_basis": k["label_basis"], "group_id": k["group_id"], "cluster": k["cluster"], "md5": k["md5"]})
    for k in side:
        folder = "_uncertain" if k["label"] == UNCERTAIN else "_audit"
        name = f"{k['source']}__{k['md5'][:12]}.jpg"
        save_processed(k["path"], out / folder / k["label"] / name, cfg)
        manifest.append({"image": f"{folder}/{k['label']}/{name}", "split": folder, "label": k["label"], "source": k["source"],
                         "label_basis": k["label_basis"], "group_id": k["group_id"], "cluster": k["cluster"], "md5": k["md5"]})
    cols = ["image", "split", "label", "source", "label_basis", "group_id", "cluster", "md5"]
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "split_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(manifest)
    leak = check_split_leakage([m for m in manifest if m["split"] in ("train", "val", "test")])
    if leak: raise SystemExit("LEAKAGE DETECTED:\n" + "\n".join(leak[:20]))
    counts = Counter((m["split"], m["label"]) for m in manifest)
    return {"input_rows": len(rows), "exact_duplicates_removed": dropped_dups, "conflicting_duplicate_sets": conflicts,
            "counts": {f"{s}/{l}": n for (s, l), n in sorted(counts.items())}, "warnings": warns}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/dataset_plastic.yaml"); ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    cfg = load_yaml(a.config); res = run(cfg, load_yaml(cfg["paths"]["policy"]), a.overwrite)
    for k, v in res.items(): print(k, v)
    for s, m in cfg["warn_min_per_class_per_split"].items():
        for l in cfg["classes"]:
            n = res["counts"].get(f"{s}/{l}", 0)
            if n < m: print(f"WARNING: {s}/{l} has only {n} images (< {m}); metrics for this class will be unreliable")

if __name__ == "__main__":
    main()
