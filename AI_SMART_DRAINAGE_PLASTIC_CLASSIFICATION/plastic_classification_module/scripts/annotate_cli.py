"""Keyboard annotation tool (OpenCV window). Shows each unlabelled crop; you pick a policy RULE.
Keys: 1 R1 recyclable | 2 NR1 PS/PVC | 3 NR2 heavy contamination | 4 NR3 multilayer sachet |
      5 U1 unknown material | 6 U2 occluded/blurry | 7 U3 film/bag | 8 U4 PP/conditional |
      s skip | q quit.  Needs a display (not for Colab/headless; there, edit the CSV directly)."""
from __future__ import annotations
import argparse, csv
import _path  # noqa: F401
from plastic_common import LABEL_COLUMNS, append_csv_row, load_yaml, resolve

KEYMAP = {"1": "R1_clean_pet_hdpe", "2": "NR1_ps_or_pvc", "3": "NR2_heavy_contamination", "4": "NR3_multilayer_sachet",
          "5": "U1_unknown_material", "6": "U2_occluded_or_blurry", "7": "U3_film_or_bag", "8": "U4_pp_or_conditional"}


def label_row(base: dict, rule_id: str, rules: dict, annotator: str) -> dict:
    contamination = "heavy" if rule_id == "NR2_heavy_contamination" else "unknown"
    return {**base, "label": rules[rule_id], "label_basis": "human_visual_rule", "rule_id": rule_id,
            "annotator": annotator, "contamination": contamination}


def main():
    import cv2
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--annotator", required=True)
    ap.add_argument("--config", default="config/dataset_plastic.yaml"); a = ap.parse_args()
    cfg = load_yaml(a.config); rules = {r["id"]: r["label"] for r in load_yaml(cfg["paths"]["policy"])["rules"]}
    root, out = resolve(cfg["paths"]["images_dir"]), resolve(cfg["paths"]["labels_csv"])
    done = set()
    if out.exists():
        with open(out, newline="", encoding="utf-8") as f: done = {r["image_path"] for r in csv.DictReader(f) if r["annotator"] == a.annotator}
    with open(resolve(cfg["paths"]["to_label_csv"]), newline="", encoding="utf-8") as f: todo = [r for r in csv.DictReader(f) if r["image_path"] not in done]
    for i, r in enumerate(todo):
        img = cv2.imread(str(root / r["image_path"]))
        if img is None: continue
        cv2.imshow(f"{i+1}/{len(todo)}  1:R1 2:NR1 3:NR2 4:NR3 5:U1 6:U2 7:U3 8:U4 s:skip q:quit", cv2.resize(img, (512, 512)))
        k = chr(cv2.waitKey(0) & 0xFF); cv2.destroyAllWindows()
        if k == "q": break
        if k in KEYMAP: append_csv_row(out, LABEL_COLUMNS, label_row(r, KEYMAP[k], rules, a.annotator))

if __name__ == "__main__":
    main()
