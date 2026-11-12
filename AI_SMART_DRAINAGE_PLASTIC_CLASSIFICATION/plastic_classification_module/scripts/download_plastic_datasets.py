"""Dataset catalogue + helpers. NOTHING here is a ready-made 'recyclable vs non-recyclable plastic' dataset:
see docs/DATASET_RESEARCH.md. Sub-commands: list | wadaba-check <dir> | taco-hint | roboflow-audit."""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import _path  # noqa: F401
from plastic_common import IMG_EXT

CATALOGUE = [
    {"name": "WaDaBa", "url": "http://wadaba.pcz.pl/", "labels": "resin code (PET, PE-HD, PP, PS, Other)", "images": "4000 (published: PET 2200, PE-HD 600, PP 640, PS 520, Other 40; PVC/PE-LD 0)",
     "recyclability_labels": "NO (resin type only; needs your policy)", "licence": "images downloadable; annotations after signing licence on the site"},
    {"name": "TACO", "url": "http://tacodataset.org/", "labels": "60 object categories, COCO", "images": "~1500 photos",
     "recyclability_labels": "NO (object categories)", "licence": "annotations MIT; Flickr images per-image CC"},
    {"name": "TrashNet / Garbage classification", "url": "https://github.com/garythung/trashnet", "labels": "glass/paper/cardboard/plastic/metal/trash", "images": "2527 (plastic 482)",
     "recyclability_labels": "NO (material category; 'trash' != non-recyclable plastic)", "licence": "MIT (per community review)"},
    {"name": "Roboflow 'recyclable / non-recyclable' community projects", "url": "https://universe.roboflow.com/ (search 'recyclable non-recyclable')", "labels": "author-defined recyclable/non-recyclable", "images": "varies; check each project",
     "recyclability_labels": "UNVERIFIED (hobbyist labelling, rules undocumented)", "licence": "mostly CC BY 4.0; check each version"},
]


def cmd_list():
    for d in CATALOGUE:
        print(f"\n{d['name']}\n  url: {d['url']}")
        for k in ("labels", "images", "recyclability_labels", "licence"): print(f"  {k}: {d[k]}")


def cmd_wadaba_check(folder: Path):
    imgs = [p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXT]
    print(f"{len(imgs)} images under {folder}")
    print("WaDaBa ships images unlabelled in the free download; resin labels come with the annotations (licence form).\n"
          "Arrange images into folders named PET/ HDPE/ PP/ PS/ OTHER/ then run:\n"
          "  python scripts/import_labelled_folders.py --root <that dir> --mode resin --source wadaba")


def cmd_roboflow(workspace, project, version, out):
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key: sys.exit("Set ROBOFLOW_API_KEY first")
    from roboflow import Roboflow  # pip install roboflow  (not tested here: no network)
    ds = Roboflow(api_key=key).workspace(workspace).project(project).version(int(version)).download("folder", location=out)
    print("downloaded to", ds.location, "\nThese labels are UNVERIFIED. Import with --mode community; they go to _audit only.")


def main():
    ap = argparse.ArgumentParser(description=__doc__); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list"); w = sub.add_parser("wadaba-check"); w.add_argument("dir"); sub.add_parser("taco-hint")
    r = sub.add_parser("roboflow-audit")
    for k in ("workspace", "project", "version", "out"): r.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    if a.cmd == "list": cmd_list()
    elif a.cmd == "wadaba-check": cmd_wadaba_check(Path(a.dir))
    elif a.cmd == "taco-hint":
        print("TACO crops: run Part 1 (download_datasets.py + prepare_datasets.py), then\n  python scripts/extract_plastic_crops.py --images <part1>/data/processed/images --labels <part1>/data/processed/labels --source taco\nThen annotate with scripts/annotate_cli.py")
    else: cmd_roboflow(a.workspace, a.project, a.version, a.out)

if __name__ == "__main__":
    main()
