# Dataset research: is there a public recyclable / non-recyclable PLASTIC dataset?

**Short answer: I found no public dataset whose labels genuinely represent plastic recyclability and that I can
recommend training on as-is.** Findings come from web searches on 12 Nov 2026 (search snippets and published papers).
**I did not download any dataset, so every image count below is quoted from a cited source, not counted by me.**
Re-verify each one when you download it.

| Dataset | Real classes | Images (as published) | Annotation | Licence | Recyclability label? |
|---|---|---|---|---|---|
| **WaDaBa** (wadaba.pcz.pl, Bobulski & Kubanek) | resin code: PET, PE-HD, PP, PS, Other (PVC and PE-LD have 0 images) | ~4000: PET 2200, PE-HD 600, PP 640, PS 520, Other 40 (from 2 papers + a benchmark paper) | image classification; single object on plain background, deformed/soiled variants; free images, annotations after signing a licence form | not stated openly: read the licence form | **No.** Resin type only. Recyclability must come from a policy (`config/recyclability_policy.yaml`). Lab domain, 1% "Other", only PS would map to non-recyclable. |
| **TACO** (tacodataset.org) | 60 object categories (bottle, bag, wrapper, ...) | ~1500 photos | COCO boxes/masks | annotations MIT, Flickr photos per-image CC | **No.** Object categories. Good source of *in-the-wild plastic crops to annotate*. |
| **TrashNet / Garbage classification** | glass, paper, cardboard, plastic, metal, trash | 2527 (plastic 482, trash 137) | classification | MIT (per community review) | **No.** Many tutorials map "plastic -> recyclable" and "trash -> non-recyclable": that is an assumption, not a label. |
| **Roboflow Universe community projects** ("recyclable / non-recyclable" searches) | author-defined; e.g. one project lists Food / Non-Recyclable / Recyclable and is described as made by an eighth grader; another ("Recyclables vs. Non-Recyclables") actually lists object classes (cardboard, paper, bottles, cans); "Recyclable Logos" labels the recycling *symbol* | varies (1k-9k listed) | boxes | mostly CC BY 4.0 | **Unverified.** Labelling rules undocumented; often object-category in disguise. Audit-only. |
| **Drinking Waste Classification** (Kaggle) | aluminium can, glass bottle, PET bottle, HDPE milk bottle | ~10k (review site) | classification | check page | **No** (object category, narrow domain). |

## What this means
* Resin-code data (WaDaBa) is the only *material-grounded* public source, but (a) recyclability still depends on a
  local policy, (b) the photos are lab-style, (c) resin is often not visible in a drain photo. A model trained only on it
  will likely transfer poorly to YOLO crops from drains.
* Object-category data (TACO, TrashNet, drinking waste) must **not** be turned into recyclable/non-recyclable labels.
  `import_labelled_folders.py` refuses folders named like objects.
* Roboflow community labels are imported only as `community_unverified` and land in `_audit`, never in train/val/test.

## The gap and the plan
1. Extract plastic crops from your own YOLO data (TACO via Part 1, plus real drain photos) with `extract_plastic_crops.py`.
2. Label them with the rule-based workflow in `ANNOTATION_GUIDE.md` (evidence + rule id; ambiguous -> uncertain).
3. Optionally add WaDaBa via resin policy for extra PET/HDPE/PS variety (`--mode resin`), keeping val/test dominated by in-domain human-labelled crops.
4. Before relying on the model, check with your local recycler/municipality that the policy file reflects what is accepted.
