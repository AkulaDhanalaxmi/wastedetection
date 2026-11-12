"""Cut plastic crops out of a YOLO-format dataset (e.g. Part 1's data/processed or your drain photos)
and register them in to_label.csv for annotation. Crops from one photo share group_id (leakage control)."""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import _path  # noqa: F401
from plastic_common import IMG_EXT, LABEL_COLUMNS, append_csv_row, load_yaml, open_rgb, resolve


def parse_yolo_line(line: str):
    """Returns (class_id, cx, cy, w, h) normalised; polygons are reduced to their bounding box."""
    p = line.split()
    if len(p) < 5: return None
    try: v = [float(x) for x in p]
    except ValueError: return None
    cid = int(v[0])
    if len(p) == 5: return cid, v[1], v[2], v[3], v[4]
    xs, ys = v[1::2], v[2::2]
    if not xs or not ys: return None
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    return cid, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0


def extract(images: Path, labels: Path, out_dir: Path, source: str, class_id: int, pad: float, min_px: int, to_label_csv: Path):
    n = 0
    done = set()
    if to_label_csv.exists():
        with open(to_label_csv, newline="", encoding="utf-8") as f: done = {r["image_path"] for r in csv.DictReader(f)}
    for img_path in sorted(p for p in images.rglob("*") if p.suffix.lower() in IMG_EXT):
        lab = labels / (img_path.stem + ".txt")
        if not lab.exists(): continue
        try: im = open_rgb(img_path)
        except Exception: continue
        W, H = im.size
        for k, line in enumerate(lab.read_text().splitlines()):
            b = parse_yolo_line(line)
            if not b or b[0] != class_id: continue
            _, cx, cy, w, h = b
            x1, x2 = (cx - w / 2 * (1 + 2 * pad)) * W, (cx + w / 2 * (1 + 2 * pad)) * W
            y1, y2 = (cy - h / 2 * (1 + 2 * pad)) * H, (cy + h / 2 * (1 + 2 * pad)) * H
            x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(W, int(x2)), min(H, int(y2))
            if x2 - x1 < min_px or y2 - y1 < min_px: continue
            rel = f"{source}/{img_path.stem}_{k}.jpg"
            if rel in done: continue
            dst = out_dir / rel; dst.parent.mkdir(parents=True, exist_ok=True)
            im.crop((x1, y1, x2, y2)).save(dst, "JPEG", quality=95)
            append_csv_row(to_label_csv, LABEL_COLUMNS, {"image_path": rel, "source": source, "group_id": f"{source}:{img_path.stem}"}); n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", required=True); ap.add_argument("--labels", required=True)
    ap.add_argument("--source", required=True, help="short name, e.g. taco / drain_site1")
    ap.add_argument("--config", default="config/dataset_plastic.yaml")
    a = ap.parse_args(); cfg = load_yaml(a.config); c = cfg["crops"]
    n = extract(Path(a.images), Path(a.labels), resolve(cfg["paths"]["images_dir"]), a.source, c["plastic_class_id"],
                c["pad_fraction"], c["min_box_px"], resolve(cfg["paths"]["to_label_csv"]))
    print(f"wrote {n} crops; next: python scripts/annotate_cli.py")

if __name__ == "__main__":
    main()
