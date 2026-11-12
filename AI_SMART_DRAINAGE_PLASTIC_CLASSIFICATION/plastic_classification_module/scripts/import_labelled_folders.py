"""Register an already-labelled class-folder dataset in labels.csv.
Two safe modes:
  --mode resin       folders are RESIN CODES (PET, HDPE, PP, PS, ...) -> label from recyclability_policy.yaml (basis resin_policy)
  --mode community   folders are someone else's 'recyclable'/'non_recyclable' labels -> basis community_unverified
                     (goes to _audit, never into train/val/test until YOU have verified them)
Object-category folders (bottle, bag, wrapper, ...) are NOT accepted: they carry no recyclability information."""
from __future__ import annotations
import argparse, shutil
from pathlib import Path
import _path  # noqa: F401
from plastic_common import IMG_EXT, LABEL_COLUMNS, append_csv_row, load_yaml, resolve

OBJECT_CATEGORY_NAMES = {"bottle", "bag", "wrapper", "cup", "container", "can", "plastic", "plastic_bottle", "plastic_bag", "trash", "garbage"}


def import_folders(root: Path, mode: str, source: str, policy: dict, images_dir: Path, labels_csv: Path, folder_map=None) -> dict:
    resin = {k.upper(): v["label"] for k, v in policy["resin_codes"].items()}
    counts = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        name = d.name.upper() if mode == "resin" else d.name.lower().replace("-", "_").replace(" ", "_")
        if mode == "resin":
            if name not in resin: raise SystemExit(f"Folder '{d.name}' is not a known resin code {sorted(resin)}")
            label, code = resin[name], name
        else:
            label = (folder_map or {}).get(d.name, name)
            if label not in ("recyclable", "non_recyclable", "uncertain"):
                raise SystemExit(f"Folder '{d.name}' is not recyclable/non_recyclable/uncertain (use --folder-map). Object categories are not labels.")
            code = ""
        for p in sorted(x for x in d.rglob("*") if x.suffix.lower() in IMG_EXT):
            rel = f"{source}/{d.name}/{p.name}"
            dst = images_dir / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dst)
            append_csv_row(labels_csv, LABEL_COLUMNS, {"image_path": rel, "label": label, "source": source, "group_id": f"{source}:{p.stem}",
                           "label_basis": "resin_policy" if mode == "resin" else "community_unverified", "resin_code": code,
                           "contamination": "unknown", "annotator": "dataset_metadata"})
            counts[(d.name, label)] = counts.get((d.name, label), 0) + 1
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True); ap.add_argument("--mode", choices=["resin", "community"], required=True)
    ap.add_argument("--source", required=True); ap.add_argument("--config", default="config/dataset_plastic.yaml"); a = ap.parse_args()
    cfg = load_yaml(a.config)
    c = import_folders(Path(a.root), a.mode, a.source, load_yaml(cfg["paths"]["policy"]), resolve(cfg["paths"]["images_dir"]), resolve(cfg["paths"]["labels_csv"]))
    for (f, l), n in c.items(): print(f"{f:>10} -> {l:<15} {n}")
    print("NOTE: group_id defaults to the file stem. If several images show the SAME physical item, set a shared group_id in labels.csv BEFORE splitting.")

if __name__ == "__main__":
    main()
