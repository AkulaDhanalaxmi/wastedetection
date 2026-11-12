# Manual annotation workflow

Why: no public dataset labels plastic recyclability reliably (see DATASET_RESEARCH.md).

## Steps
1. `python scripts/extract_plastic_crops.py --images <yolo images> --labels <yolo labels> --source drain_site1`
   creates crops under `data/plastic_crops/<source>/` and rows in `data/annotations/to_label.csv`.
   All crops of the same photo share `group_id`, so they can never straddle train/val/test.
2. Label: `python scripts/annotate_cli.py --annotator <you>` (OpenCV window, keys 1-8 = policy rules), or edit
   `labels.csv` directly (e.g. in Colab/Excel).
3. Recommended: a second person labels the same crops independently (`--annotator <other>`), then
   `python scripts/merge_annotations.py --a a1.csv --b a2.csv --out data/annotations/labels.csv`
   (agree -> `human_two_annotator_agree`; disagree -> `uncertain`; prints Cohen's kappa).
4. `python scripts/validate_plastic_dataset.py --stage labels`

## labels.csv columns
`image_path` (relative to data/plastic_crops), `label` (recyclable | non_recyclable | uncertain), `label_basis`
(resin_policy | human_visual_rule | human_two_annotator_agree | community_unverified), `rule_id` (from
config/recyclability_policy.yaml), `source`, `group_id`, `annotator`, `resin_code`, `contamination`
(none|light|heavy|unknown), `notes`.

## Rules (edit in config/recyclability_policy.yaml)
* Label **recyclable** only with rule R1: rigid PET/HDPE item, material known or unambiguous, clean or lightly soiled.
* Label **non_recyclable** only with NR1 (PS/PVC), NR2 (heavily contaminated) or NR3 (multilayer sachet). Never because the item "looks like rubbish".
* Everything else, and anything you hesitate on: **uncertain** (U1-U4). Uncertain crops are kept in `_uncertain/`
  and are never forced into a class; evaluation reports how many the model routes to manual review.
* "It is a bottle" is not evidence. Shape alone gives no recyclability.
* The validator rejects: rule/label mismatches, heavy contamination marked recyclable, resin_policy labels that
  contradict the policy, missing files, missing group_id.

## How much is enough?
Config warns below 100 train / 30 val / 30 test images per class (`warn_min_per_class_per_split`). These are
sanity floors, not guarantees; more real in-domain data matters more than anything else.
