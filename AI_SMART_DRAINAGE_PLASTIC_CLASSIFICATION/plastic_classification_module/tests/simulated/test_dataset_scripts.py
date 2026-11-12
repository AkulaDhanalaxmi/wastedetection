"""SIMULATED data-pipeline tests on throw-away synthetic images generated in tmp dirs (nothing shipped)."""
import copy, csv, io
import numpy as np
import pytest
from conftest import make_classifier, synth_image, write_labels
from plastic_common import LABEL_COLUMNS, load_yaml, sweep_thresholds, suggest_threshold, fit_temperature, softmax
from find_duplicates import analyse
from prepare_plastic_dataset import assign_splits, run
from validate_plastic_dataset import check_split_leakage, read_csv, validate_rows
from import_labelled_folders import import_folders
from merge_annotations import cohen_kappa, merge
from extract_plastic_crops import extract, parse_yolo_line
from evaluate_plastic import evaluate

CFG = load_yaml("config/dataset_plastic.yaml"); POL = load_yaml("config/recyclability_policy.yaml")


def row(path, label, rule, group, basis="human_visual_rule", **kw):
    base = {c: "" for c in LABEL_COLUMNS}
    base.update({"image_path": path, "label": label, "label_basis": basis, "rule_id": rule, "source": "t", "group_id": group,
                 "annotator": "a1", "contamination": "none"})
    return {**base, **kw}


def build(tmp_path, n_groups=40):
    imgs = tmp_path / "imgs"; imgs.mkdir(); rows = []; seed = 0
    for g in range(n_groups):
        lab, rule = ("recyclable", "R1_clean_pet_hdpe") if g % 2 == 0 else ("non_recyclable", "NR1_ps_or_pvc")
        for k in range(3):                                   # 3 crops of the same photo -> same group_id
            seed += 1; name = f"g{g}_{k}.jpg"; synth_image(seed).save(imgs / name); rows.append(row(name, lab, rule, f"photo{g}"))
    return imgs, rows


def test_split_has_no_leakage_and_groups_stay_together(tmp_path):
    imgs, rows = build(tmp_path)
    synth_image(999).save(imgs / "orig.jpg"); synth_image(999).save(imgs / "exact_copy.jpg")           # exact duplicate (different name)
    im = synth_image(1000); im.save(imgs / "near_a.jpg"); im.resize((60, 50)).save(imgs / "near_b.jpg", quality=70)  # near duplicate
    rows += [row("orig.jpg", "recyclable", "R1_clean_pet_hdpe", "o"), row("exact_copy.jpg", "recyclable", "R1_clean_pet_hdpe", "oc"),
             row("near_a.jpg", "non_recyclable", "NR1_ps_or_pvc", "na"), row("near_b.jpg", "non_recyclable", "NR1_ps_or_pvc", "nb")]
    write_labels(tmp_path / "labels.csv", rows)
    res = run(CFG, POL, labels_csv=tmp_path / "labels.csv", images_dir=imgs, out_dir=tmp_path / "out")
    assert res["exact_duplicates_removed"] == 1
    m = read_csv(tmp_path / "out" / "split_manifest.csv")
    assert check_split_leakage([r for r in m if r["split"] in ("train", "val", "test")]) == []
    by_group = {}
    for r in m:
        by_group.setdefault(r["group_id"], set()).add(r["split"])
    assert all(len(s) == 1 for s in by_group.values() if s)
    assert {"train", "val", "test"} <= {r["split"] for r in m}
    near = {r["split"] for r in m if r["group_id"] in ("na", "nb")}; assert len(near) == 1          # near-dups never straddle splits


def test_uncertain_and_unverified_kept_out_of_splits(tmp_path):
    imgs, rows = build(tmp_path, 12)
    synth_image(7001).save(imgs / "u.jpg"); synth_image(7002).save(imgs / "c.jpg")
    rows += [row("u.jpg", "uncertain", "U1_unknown_material", "u"), row("c.jpg", "recyclable", "", "c", basis="community_unverified")]
    write_labels(tmp_path / "labels.csv", rows)
    run(CFG, POL, labels_csv=tmp_path / "labels.csv", images_dir=imgs, out_dir=tmp_path / "out")
    m = read_csv(tmp_path / "out" / "split_manifest.csv")
    assert [r["split"] for r in m if r["group_id"] == "u"] == ["_uncertain"] and [r["split"] for r in m if r["group_id"] == "c"] == ["_audit"]


def test_conflicting_duplicate_labels_become_uncertain(tmp_path):
    imgs = tmp_path / "imgs"; imgs.mkdir(); synth_image(5).save(imgs / "a.jpg"); synth_image(5).save(imgs / "b.jpg")
    write_labels(tmp_path / "l.csv", [row("a.jpg", "recyclable", "R1_clean_pet_hdpe", "a"), row("b.jpg", "non_recyclable", "NR1_ps_or_pvc", "b")])
    res = run(CFG, POL, labels_csv=tmp_path / "l.csv", images_dir=imgs, out_dir=tmp_path / "o")
    assert res["conflicting_duplicate_sets"] == 1 and list(res["counts"]) == ["_uncertain/uncertain"]


def test_validation_catches_bad_labels(tmp_path):
    imgs = tmp_path / "i"; imgs.mkdir(); synth_image(1).save(imgs / "a.jpg")
    bad = [row("a.jpg", "maybe", "R1_clean_pet_hdpe", "g"),                                  # invalid label
           row("a.jpg", "recyclable", "NR1_ps_or_pvc", "g"),                                  # rule/label mismatch (+dup path)
           row("missing.jpg", "recyclable", "R1_clean_pet_hdpe", ""),                        # missing file, empty group
           row("a.jpg", "recyclable", "R1_clean_pet_hdpe", "g", contamination="heavy"),
           row("a.jpg", "recyclable", "", "g", basis="resin_policy", resin_code="PS")]       # resin policy says PS -> non_recyclable
    errs, _ = validate_rows(bad, CFG, POL, imgs)
    text = "\n".join(errs)
    for needle in ("not in", "implies", "not found", "empty group_id", "heavily contaminated", "policy 'non_recyclable'", "duplicate image_path"):
        assert needle in text, needle


def test_validation_accepts_clean_rows(tmp_path):
    imgs, rows = build(tmp_path, 4); assert validate_rows(rows, CFG, POL, imgs)[0] == []


def test_import_refuses_object_category_folders(tmp_path):
    for d in ("bottle", "bag"):
        (tmp_path / "ds" / d).mkdir(parents=True); synth_image(3).save(tmp_path / "ds" / d / "x.jpg")
    with pytest.raises(SystemExit):
        import_folders(tmp_path / "ds", "community", "s", POL, tmp_path / "im", tmp_path / "l.csv")
    with pytest.raises(SystemExit):
        import_folders(tmp_path / "ds", "resin", "s", POL, tmp_path / "im", tmp_path / "l.csv")


def test_import_resin_folders_uses_policy(tmp_path):
    for d in ("PET", "PS", "PP"):
        (tmp_path / "ds" / d).mkdir(parents=True); synth_image(hash(d) % 1000).save(tmp_path / "ds" / d / "x.jpg")
    c = import_folders(tmp_path / "ds", "resin", "wadaba", POL, tmp_path / "im", tmp_path / "l.csv")
    assert c == {("PET", "recyclable"): 1, ("PS", "non_recyclable"): 1, ("PP", "uncertain"): 1}
    assert validate_rows(read_csv(tmp_path / "l.csv"), CFG, POL, tmp_path / "im")[0] == []


def test_extract_crops_from_yolo(tmp_path):
    (tmp_path / "im").mkdir(); (tmp_path / "lb").mkdir(); synth_image(1, (200, 100)).save(tmp_path / "im" / "p.jpg")
    (tmp_path / "lb" / "p.txt").write_text("0 0.5 0.5 0.4 0.6\n1 0.2 0.2 0.1 0.1\n0 0.1 0.1 0.01 0.01\n")
    n = extract(tmp_path / "im", tmp_path / "lb", tmp_path / "out", "src", 0, 0.1, 24, tmp_path / "t.csv")
    assert n == 1 and parse_yolo_line("0 0.1 0.1 0.5 0.1 0.5 0.5") is not None
    assert read_csv(tmp_path / "t.csv")[0]["group_id"] == "src:p"


def test_two_annotator_merge_and_kappa():
    a = [row("1", "recyclable", "R1_clean_pet_hdpe", "g"), row("2", "recyclable", "R1_clean_pet_hdpe", "g")]
    b = {"1": {**row("1", "recyclable", "R1_clean_pet_hdpe", "g"), "annotator": "a2"}, "2": {**row("2", "non_recyclable", "NR1_ps_or_pvc", "g"), "annotator": "a2"}}
    merged, kappa = merge(a, b, {})
    assert merged[0]["label_basis"] == "human_two_annotator_agree" and merged[1]["label"] == "uncertain"
    assert cohen_kappa(["recyclable"] * 4 + ["non_recyclable"] * 4, ["recyclable"] * 4 + ["non_recyclable"] * 4) == 1.0


def test_threshold_and_calibration_utils():
    probs = np.array([[0.9, 0.1], [0.6, 0.4], [0.2, 0.8], [0.55, 0.45]]); y = np.array([0, 1, 1, 0])
    rows = sweep_thresholds(probs, y); assert rows[0]["coverage"] == 1.0
    assert suggest_threshold(rows, 0.99, 0.9) is None
    assert fit_temperature(np.zeros((3, 2)), np.zeros(3, int)) == 1.0


def test_assign_splits_roughly_stratified():
    cl = {i: ["recyclable"] * 3 if i % 2 else ["non_recyclable"] * 3 for i in range(100)}
    out = assign_splits(cl, {"train": .7, "val": .15, "test": .15}, 42)
    for s in ("train", "val", "test"):
        labs = {l for c, sp in out.items() if sp == s for l in cl[c]}; assert labs == {"recyclable", "non_recyclable"}


def test_evaluate_logic_with_simulated_backend(tmp_path):
    """Exercises metrics/abstention code only. The fake model always says 'recyclable', so ~50% accuracy is EXPECTED."""
    for c in ("recyclable", "non_recyclable"):
        (tmp_path / "test" / c).mkdir(parents=True)
        for i in range(5): synth_image(hash(c) % 100 + i).save(tmp_path / "test" / c / f"{i}.jpg")
    res = evaluate(make_classifier([-3.0, 3.0]), tmp_path / "test")
    assert res["n"] == 10 and abs(res["accuracy"] - 0.5) < 1e-9 and res["deployed"]["non_recyclable_called_recyclable_rate"] == 1.0
    assert res["per_class"]["recyclable"]["recall"] == 1.0 and res["per_class"]["non_recyclable"]["recall"] == 0.0
